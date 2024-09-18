#!/usr/bin/env/python3
"""Recipe for training a Text-to-Speech system based on tokenized audio

Inspired by WhisperSpeech
https://github.com/collabora/WhisperSpeech

However, this is not an implementation of WhisperSpeech, but rather
a radical simplification of it that uses only an acoustic model


Authors
 * Artem Ploujnikov 2024
"""


import logging
import math
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.dataio.preparation import add_prepared_features
from speechbrain.lobes.models.discrete.Tokotron import RepresentationMode
from speechbrain.utils.audio_tokens import (
    feature_pad_to,
    get_silence_token,
    use_silence_padding,
)
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)

SPECIAL_TOKEN_COUNT = 1


# Brain class for speech recognition training
class TokotronBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Runs all the computation of the Tokotron TTS

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            TTS predictions
        """
        batch = batch.to(self.device)
        tokens, tokens_length = batch.tokens
        audio, audio_length = batch.audio_bos
        if self.compression:
            audio = self.compression_model.compress(audio)
        emb = None
        if self.use_spk_emb:
            emb = {"spk": batch.spk_emb.data.squeeze(1)}

        predictions = self.modules.model(
            input_tokens=tokens,
            input_length=tokens_length,
            audio=audio,
            audio_length=audio_length,
            emb=emb,
        )

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs. We here
        do multi-task learning and the loss is a weighted sum of the ctc + seq2seq
        costs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        batch = batch.to(self.device)
        audio, audio_length = batch.audio_pad
        if self.compression:
            audio = self.compression_model.compress(audio)
        loss_details = self.hparams.compute_cost(
            predictions=predictions,
            audio=audio,
            audio_length=audio_length,
            input_tokens=batch.tokens.data,
            input_length=batch.tokens.lengths,
        )
        self.loss_metric.append(
            batch.uttid,
            predictions=predictions,
            audio=audio,
            audio_length=audio_length,
            input_tokens=batch.tokens.data,
            input_length=batch.tokens.lengths,
            reduction="batch",
        )
        return loss_details.loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        if hasattr(self.modules.vocoder, "model"):
            self.modules.vocoder.model.device = self.device
        self.create_perfect_samples()
        self.loss_metric = sb.utils.metric_stats.MultiMetricStats(
            metric=self.hparams.compute_cost,
            batch_eval=True,
        )
        if (
            self.hparams.audio_emb_pretrained
            and epoch == 1
            and stage == sb.Stage.TRAIN
        ):
            vocabulary = None
            if hasattr(self.hparams.token_model, "vocabulary"):
                vocabulary = self.hparams.token_model.vocabulary
            elif hasattr(self.hparams.token_model, "vocabularies"):
                vocabulary = torch.stack(
                    [
                        torch.from_numpy(voc)
                        for voc in self.hparams.token_model.vocabularies
                    ]
                )
            if vocabulary is not None:
                self.modules.model.init_audio_emb(vocabulary)
        # Load the compression model only if compression is enables
        self.compression = getattr(self.hparams, "compression", False)
        if self.compression:
            self.compression_model = self.hparams.compression_model(
                run_opts={"device": self.device}
            )
            self.modules.model.compression_model = self.compression_model

        # Speaker embeddings are optional and are used only to pretrain
        # multispeaker models
        self.use_spk_emb = getattr(self.hparams, "use_spk_emb", False)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        loss_stats = self.loss_metric.summarize(flat=True)
        stage_stats = {"loss": stage_loss, **loss_stats}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            if self.hparams.lr_annealing_mode == "epoch":
                _, new_lr = self.hparams.lr_annealing(stage_loss)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            lr = self.optimizer.param_groups[0]["lr"]

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]},
                min_keys=["loss"],
            )

            self.create_samples()

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``
        * ``optimizers_step()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        loss = super().fit_batch(batch)
        if self.hparams.lr_annealing_mode == "step":
            self.hparams.lr_annealing(self.optimizer)
        return loss

    def create_samples(self):
        """Writes audio samples at the end of an epoch"""
        epoch = self.hparams.epoch_counter.current
        if epoch % self.hparams.samples_interval != 0:
            return
        if self.debug:
            self.modules.model.decoder.infer_max_decoder_steps = (
                self.hparams.debug_infer_max_audio_length
            )
        sample_loader = sb.dataio.dataloader.make_dataloader(
            self.sample_data,
            **self.hparams.sample_dataloader_opts,
        )
        with self.hparams.progress_report:
            for batch in sample_loader:
                batch = batch.to(self.device)
                tokens, tokens_length = batch.tokens
                emb = None
                if self.use_spk_emb:
                    emb = {"spk": batch.spk_emb.data.squeeze(1)}

                with torch.no_grad():
                    infer_out = self.modules.model.infer(
                        input_tokens=tokens, input_length=tokens_length, emb=emb
                    )
                self.hparams.progress_report.write(
                    ids=batch.uttid,
                    audio=infer_out.wav,
                    length_pred=infer_out.wav_length,
                    length=batch.audio_pad.lengths,
                    tgt_max_length=batch.audio_pad.data.size(1),
                    alignments=infer_out.alignments,
                    p_eos=infer_out.p_eos,
                )

    def create_perfect_samples(self):
        """Creates the best samples that can be created using
        the vocoder provided, for comparison purposes"""
        if not self.hparams.progress_logger["perfect_samples_created"]:
            sample_loader = sb.dataio.dataloader.make_dataloader(
                self.sample_data, **self.hparams.sample_dataloader_opts
            )
            for batch in sample_loader:
                batch = batch.to(self.device)
                sample_audio, length = batch.audio_pad
                with torch.no_grad():
                    vocoder_out = self.modules.vocoder(sample_audio, length)
                if isinstance(vocoder_out, tuple):
                    samples, samples_length = vocoder_out
                else:
                    samples = vocoder_out
                    samples_length = length
                if samples.dim() == 3:
                    samples = samples.squeeze(1)
                max_len = samples.size(1)
                samples_length_abs = (samples_length * max_len).int()
                with self.hparams.progress_logger:
                    for item_id, item_wav, item_length in zip(
                        batch.uttid, samples, samples_length_abs
                    ):
                        item_cut = item_wav[: item_length.item()]
                        self.hparams.progress_logger.save(
                            name=f"{item_id}.wav",
                            content=item_cut.detach().cpu(),
                            mode="audio",
                            folder="_perfect",
                            samplerate=self.hparams.model_sample_rate,
                        )
                    self.hparams.progress_logger["perfect_samples_created"] = (
                        True
                    )
                    self.hparams.progress_logger.clear()


INPUT_FEATURE_MAP = {"text": "label_norm", "phonemes": "phonemes"}


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys that correspond
        to the DynamicItemDataset objects.
    silence_token : dict
        the token used for silence
    """

    representation_mode = RepresentationMode(
        hparams.get("representation_mode", RepresentationMode.DISCRETE)
    )

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_folder = hparams["data_folder"]
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }
    label_encoder = hparams["label_encoder"]
    input_feature = INPUT_FEATURE_MAP[hparams["input"]]

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_norm")
    def text_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        return label.upper()

    @sb.utils.data_pipeline.takes(input_feature)
    @sb.utils.data_pipeline.provides("tokens")
    def tokens_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        return label_encoder.encode_sequence_torch(label)

    use_silence_padding = hparams.get("use_silence_padding", True)
    audio_tokens_per_step = len(hparams["token_model_layers"])
    if use_silence_padding:
        silence_token, silence_emb = get_silence_token(
            hparams["token_model"],
            extract_emb=representation_mode == RepresentationMode.CONTINUOUS,
            model_kwargs=hparams.get("token_model_kwargs"),
        )
    else:
        silence_token = (
            torch.ones(hparams["audio_tokens_per_step"], dtype=torch.int64)
            * hparams["eos_index"]
        )
    silence_token = silence_token.cpu()
    silence_padding = (
        silence_token
        if representation_mode == RepresentationMode.DISCRETE
        else silence_emb
    )
    silence_padding_len = int(math.ceil(hparams["silence_padding"]))
    bos_width = hparams.get("bos_width", 1)
    audio_features = (
        "audio_tokens"
        if representation_mode == RepresentationMode.DISCRETE
        else "audio_ssl"
    )
    audio_bos_prefix = (
        torch.ones(bos_width, audio_tokens_per_step) * hparams["bos_index"]
    )
    if representation_mode == RepresentationMode.CONTINUOUS:
        audio_bos_prefix = audio_bos_prefix.unsqueeze(-1).repeat(
            1, 1, hparams["audio_dim"]
        )

    @sb.utils.data_pipeline.takes(audio_features)
    @sb.utils.data_pipeline.provides("audio_pad", "audio_bos")
    def audio_pipeline(audio):
        audio = torch.from_numpy(audio)
        audio_pad = feature_pad_to(
            audio, len(audio) + silence_padding_len, silence_padding
        )
        yield audio_pad
        audio_bos = torch.cat([audio_bos_prefix, audio_pad], dim=0)
        yield audio_bos

    dynamic_items = [text_pipeline, tokens_pipeline, audio_pipeline]

    init_sequence_encoder(hparams)
    use_spk_emb = hparams.get("use_spk_emb", False)
    prepared_features = [audio_features]
    output_keys = [
        "uttid",
        "tokens",
        "audio_pad",
        "audio_bos",
    ]
    if use_spk_emb:
        prepared_features.append("spk_emb")
        output_keys.append("spk_emb")

    for dataset in data_info:
        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=dynamic_items,
            output_keys=output_keys,
        )

        add_prepared_features(
            dataset=dynamic_dataset,
            save_path=Path(hparams["prepare_save_folder"]) / "features",
            id_key="uttid",
            features=prepared_features,
        )

        datasets[dataset] = dynamic_dataset
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="length")
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="length", reverse=True
        )
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    datasets["sample"] = select_sample(hparams, datasets)
    return datasets, silence_padding


def select_sample(hparams, datasets):
    """Selects a sample of files for sample generation, freezing the sample if
    requested to persist across multiple experiments

    Arguments
    ---------
    hparams : dict
        experiment hyperparameters
    datasets : dict
        a dictionary of datasets

    Returns
    -------
    dataset : speechbrain.dataio.dataset.FilteredSortedDynamicItemDataset
        the sample dataset
    """
    sample_path = hparams.get("sample_path")
    dataset = None
    if sample_path is not None:
        sample_path = Path(sample_path)
        if sample_path.exists():
            with open(sample_path, "r") as sample_file:
                data_ids = [line.strip() for line in sample_file]
                dataset = FilteredSortedDynamicItemDataset(
                    datasets["valid"], data_ids
                )

    if dataset is None:
        dataset = (
            datasets["valid"]
            .batch_shuffle(1)
            .filtered_sorted(select_n=hparams["num_audio_samples"])
        )
        if sample_path is not None:
            with open(sample_path, "w") as sample_file:
                for data_id in dataset.data_ids:
                    print(data_id, file=sample_file)
    return dataset


def init_sequence_encoder(hparams):
    """Initialize a sequence encoder

    Arguments
    ---------
    hparams: dict
        parsed hyperparameters

    Returns
    -------
    encoder: speechbrain.dataio.encoder.TextEncoder
        an encoder instance"""
    encoder = hparams["label_encoder"]
    token_list_file_name = hparams["token_list_file"]
    tokens = read_token_list(token_list_file_name)
    encoder.add_unk()
    encoder.update_from_iterable(tokens, sequence_input=False)
    encoder.expect_len(len(tokens) + SPECIAL_TOKEN_COUNT)
    return encoder


def read_token_list(file_name):
    """Reads a simple text file with tokens (e.g. characters or phonemes) listed
    one per line

    Arguments
    ---------
    file_name: str
        the file name

    Returns
    -------
    result: list
        a list of tokens
    """
    if not Path(file_name).exists():
        raise ValueError(f"Token file {file_name} not found")
    with open(file_name) as token_file:
        return [line.strip("\r\n") for line in token_file if line]


def apply_overfit_test(hparams, dataset):
    """Helper for applying an overfit test conditionally based
    on hyperparameters:

    `overfit_test`: whether or not to apply an overfit test
    `overfit_test_sample_count`: the number of samples to use from the
        original dataset
    `overfit_test_epoch_data_count`: the number of samples per epoch

    The function will accept datasets, (train, valid, test) tuples
    or dictionaries of the form:
    {"train": dataset1, "valid": dataset2, "test": dataset3}

    If a tuple or dictionary is used, the training dataset will be of length
    overfit_test_epoch_data_count wheres the evaluation dataset will be of
    length overfit_test_sample_count.

    Arguments
    ---------
    hparams: dict
        parsed hyperparameters
    dataset: DynamicItemDataset|tuple|dict
        One of the following
        a dataset
        a dictionary ({"train": dataset1, "valid": dataset2, "test": dataset3})
        a (train, valid, test)  tuple of datasets

    Returns
    -------
    result: DynamicItemDataset|tuple|dict
        a dataset or collection of datasets suitable for
        an overfitting test - in the same format as the
        dataset argument (single dataset, dictionary and tuple)
    """
    if hparams["overfit_test"]:
        if isinstance(dataset, tuple):
            dataset_train, _, _ = dataset
            dataset_train = apply_overfit_test(hparams, dataset_train)
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
            )
            result = dataset_train, dataset_eval, dataset_eval
        elif isinstance(dataset, dict):
            dataset_train = apply_overfit_test(hparams, dataset["train"])
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
            )
            result = {
                "train": dataset_train,
                "valid": dataset_eval,
                "test": dataset_eval,
                "sample": dataset_eval,
            }
        else:
            result = dataset.overfit_test(
                hparams["overfit_test_sample_count"],
                hparams["overfit_test_epoch_data_count"],
            )
    else:
        result = dataset
    return result


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from ljspeech_prepare import prepare_ljspeech

    # Data preparation, to be run on only one process.
    representation_mode = RepresentationMode(
        hparams.get("representation_mode", RepresentationMode.DISCRETE)
    )
    audio_features = (
        "audio_tokens"
        if representation_mode == RepresentationMode.DISCRETE
        else "audio_ssl"
    )
    extract_features = [audio_features]
    if hparams.get("use_spk_emb", False):
        extract_features.append("spk_emb")

    if not hparams["skip_prep"]:
        with hparams["freezer"]:
            run_on_main(
                prepare_ljspeech,
                kwargs={
                    "data_folder": hparams["data_folder"],
                    "save_folder": hparams["prepare_save_folder"],
                    "splits": hparams["splits"],
                    "split_ratio": hparams["split_ratio"],
                    "seed": hparams["seed"],
                    "extract_features": extract_features,
                    "extract_features_opts": hparams["extract_features_opts"],
                    "extract_phonemes": hparams["input"] == "phonemes",
                    "model_name": "tokotron",
                    "g2p_src": hparams["g2p_src"],
                    "skip_ignore_folders": hparams[
                        "prepare_skip_ignore_folders"
                    ],
                    "frozen_split_path": hparams.get("frozen_split_path"),
                    "device": run_opts.get("device", "cpu"),
                },
            )

    # We can now directly create the datasets for training, valid, and test
    datasets, silence_padding = dataio_prepare(hparams)

    # Apply overfit test settings
    datasets = apply_overfit_test(hparams, datasets)
    audio_keys = ["audio_pad", "audio_bos"]

    # Trainer initialization
    tts_brain = TokotronBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    tts_brain.sample_data = datasets["sample"]

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    tts_brain.fit(
        tts_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=use_silence_padding(
            hparams["train_dataloader_opts"], silence_padding, audio_keys
        ),
        valid_loader_kwargs=use_silence_padding(
            hparams["valid_dataloader_opts"], silence_padding, audio_keys
        ),
    )

    # Load best checkpoint for evaluation
    test_stats = tts_brain.evaluate(
        test_set=datasets["test"],
        min_key="loss",
        test_loader_kwargs=use_silence_padding(
            hparams["test_dataloader_opts"], silence_padding, audio_keys
        ),
    )

    # Save final checkpoint (fixed name)
    tts_brain.checkpointer.save_checkpoint(name="latest")
