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
import re
import string
import sys
from pathlib import Path

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataio import write_audio
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.lobes.models.discrete.Tokotron import (
    RepresentationMode,
    get_silence_repr,
    get_silence_token,
)
from speechbrain.utils.data_utils import feature_pad
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
        features = self.prepare_features(batch)
        audio_bos, audio_bos_length, _, _, spk_emb = features
        emb = None
        if self.hparams.use_spk_emb:
            emb = {"spk": spk_emb}

        predictions = self.modules.model(
            input_tokens=tokens,
            input_length=tokens_length,
            audio=audio_bos,
            audio_length=audio_bos_length,
            emb=emb,
        )

        return predictions, features

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
        predictions, features = predictions
        _, _, audio_pad, audio_pad_length, _ = features
        loss_details = self.hparams.compute_cost(
            predictions=predictions,
            audio=audio_pad,
            audio_length=audio_pad_length,
            input_tokens=batch.tokens.data,
            input_length=batch.tokens.lengths,
        )
        self.loss_metric.append(
            batch.uttid,
            predictions=predictions,
            audio=audio_pad,
            audio_length=audio_pad_length,
            input_tokens=batch.tokens.data,
            input_length=batch.tokens.lengths,
            reduction="batch",
        )
        return loss_details.loss

    # TODO: Replace this with offline extraction
    def prepare_features(self, batch):
        """Prepares features for training

        Arguments
        ---------
        batch : PaddedBatch
            a batch

        Returns
        -------
        audio_bos : torch.Tensor
            The audio with the BOS marker
        audio_bos_length : torch.Tensor
            The length of the BOS sequence
        audio_pad : torch.Tensor
            The audio with silence padding
        audio_pad_length : torch.Tensor
            The length of padded audio, if applicable
        spk_emb : torch.Tensor
            The speaker embedding (if enabled)
        """
        audio, *_ = self.modules.audio_model(batch.sig.data)
        silence_padding_len = int(math.ceil(hparams["silence_padding"]))
        audio_pad, audio_pad_length = feature_pad(
            audio, batch.sig.lengths, silence_padding_len, self.end_padding
        )
        batch_size = batch.sig.data.size(0)
        audio_bos = torch.cat(
            [
                self.audio_bos_prefix[None, :, :].expand(
                    batch_size, *self.audio_bos_prefix.shape
                ),
                audio_pad,
            ],
            dim=1,
        )
        audio_length_abs = audio_pad_length * audio_pad.size(1)
        audio_bos_length = (
            audio_length_abs + self.hparams.bos_width
        ) / audio_bos.size(1)
        if self.hparams.use_spk_emb:
            mel_spec = self.modules.spk_emb_model.mel_spectogram(
                audio=batch.sig.data
            )
            spk_emb = self.modules.spk_emb_model.encode_mel_spectrogram_batch(
                mel_spec, batch.sig.lengths
            )
        else:
            spk_emb = None
        return audio_bos, audio_bos_length, audio_pad, audio_pad_length, spk_emb

    def get_end_padding(self):
        """Obtains the padding to be added at the end of the audio
        sequence

        Returns
        -------
        end_padding : torch.Tensor
            The padding tensor"""
        if self.hparams.use_silence_padding:
            if representation_mode == RepresentationMode.DISCRETE:
                end_padding = get_silence_token(
                    self.hparams.token_model,
                    model_kwargs=hparams.get("token_model_kwargs"),
                )
            else:
                end_padding = get_silence_repr(
                    self.hparams.audio_model,
                )
        else:
            end_padding = (
                torch.ones(
                    self.hparams.audio_tokens_per_step, dtype=torch.int64
                )
                * hparams["eos_index"]
            )
        return end_padding

    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        loss = super().evaluate_batch(batch, stage)
        if self.is_evaluating:
            self.create_samples(batch)
        loss = loss.detach().cpu()
        return loss

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
        self.loss_metric = sb.utils.metric_stats.MultiMetricStats(
            metric=self.hparams.compute_cost,
            batch_eval=True,
        )
        self.audio_bos_prefix = self.get_bos_prefix()
        self.end_padding = self.get_end_padding()
        self.is_evaluating = (stage == sb.Stage.TEST) or (
            epoch % self.hparams.samples_interval == 0
        )

    def get_bos_prefix(self):
        """Computes the beginning-of-sequence prefix"""
        audio_bos_prefix = (
            torch.ones(
                self.hparams.bos_width, self.hparams.audio_tokens_per_step
            )
            * hparams["bos_index"]
        )
        if representation_mode == RepresentationMode.CONTINUOUS:
            audio_bos_prefix = audio_bos_prefix.unsqueeze(-1).repeat(
                1, 1, self.hparams.audio_dim
            )
        return audio_bos_prefix

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

    @torch.no_grad()
    def create_samples(self, batch):
        """Creates samples for the specified batch

        Arguments
        ---------
        batch : PaddedBatch
            a batch
        """
        epoch = self.hparams.epoch_counter.current
        batch = batch.to(self.device)
        tokens, tokens_length = batch.tokens
        infer_out = self.modules.model.infer(
            input_tokens=tokens, input_length=tokens_length
        )
        samples_folder = Path(self.hparams.progress_folder) / str(epoch)
        samples_folder.mkdir(parents=True, exist_ok=True)
        wav_length_abs = (infer_out.wav_length * infer_out.wav.size(-1)).int()
        for uttid, wav, wav_length in zip(
            batch.uttid, infer_out.wav, wav_length_abs
        ):
            wav = wav[:, : wav_length.item()]
            file_name = samples_folder / f"{uttid}.wav"
            write_audio(
                file_name,
                wav.squeeze(0),
                samplerate=self.hparams.model_sample_rate,
            )

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
        loss : torch.Tensor
            detached loss
        """
        loss = super().fit_batch(batch)
        if self.hparams.lr_annealing_mode == "step":
            self.hparams.lr_annealing(self.optimizer)
        return loss


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

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def sig_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.functional.resample(
            sig,
            orig_freq=hparams["sample_rate"],
            new_freq=hparams["model_sample_rate"],
        )
        return sig

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_norm", "label_norm_eval")
    def text_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        label_norm = label.upper()
        yield label_norm
        label_norm_eval = RE_PUNCTUATION.sub("", label_norm)
        yield label_norm_eval

    @sb.utils.data_pipeline.takes(input_feature)
    @sb.utils.data_pipeline.provides("tokens")
    def tokens_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        return label_encoder.encode_sequence_torch(label)

    dynamic_items = [text_pipeline, tokens_pipeline, sig_pipeline]

    init_sequence_encoder(hparams)
    output_keys = ["uttid", "tokens", "sig"]

    for dataset in data_info:
        dataset_dynamic_items = list(dynamic_items)
        dataset_output_keys = list(output_keys)
        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=dataset_dynamic_items,
            output_keys=dataset_output_keys,
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
    return datasets


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
            with open(sample_path, "r", encoding="utf-8") as sample_file:
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
            with open(sample_path, "w", encoding="utf-8") as sample_file:
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
    file_name = Path(file_name)
    if not file_name.is_absolute() and not file_name.exists():
        file_name = Path(__file__).parent / file_name
    if not file_name.exists():
        raise ValueError(f"Token file {file_name} not found")
    with open(file_name, encoding="utf-8") as token_file:
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


RE_PUNCTUATION = re.compile(
    "|".join(re.escape(char) for char in string.punctuation)
)


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        yaml = fin.read()

    # Load evaluation hyperparameters
    eval_hparams_file = Path(hparams_file).parent / "eval.yaml"
    if eval_hparams_file.exists():
        logger.info(
            "Using evaluation hyperparameters from %s", eval_hparams_file
        )
        with open(eval_hparams_file, encoding="utf-8") as eval_hparams:
            hparams_yaml = eval_hparams.read()
            yaml = "\n".join([yaml, hparams_yaml])
    else:
        logger.info(
            "%s not found - not using evaluation hyperparameters",
            eval_hparams_file,
        )
    hparams = load_hyperpyyaml(yaml, overrides, overrides_must_match=True)

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

    if not hparams["skip_prep"]:
        prepare_opts = {
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["prepare_save_folder"],
            "splits": hparams["splits"],
            "split_ratio": hparams["split_ratio"],
            "seed": hparams["seed"],
            "extract_phonemes": hparams["input"] == "phonemes",
            "model_name": "tokotron",
            "g2p_src": hparams["g2p_src"],
            "skip_ignore_folders": hparams["prepare_skip_ignore_folders"],
            "device": run_opts.get("device", "cpu"),
        }
        run_on_main(prepare_ljspeech, kwargs=prepare_opts)

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # Apply overfit test settings
    datasets = apply_overfit_test(hparams, datasets)

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
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    test_stats = tts_brain.evaluate(
        test_set=datasets["test"],
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    # Save final checkpoint (fixed name)
    tts_brain.checkpointer.save_checkpoint(name="latest")
