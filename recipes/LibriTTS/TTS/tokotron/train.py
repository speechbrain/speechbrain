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
from functools import partial
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.dataio.preparation import add_prepared_features
from speechbrain.inference.eval import SpeechEvaluationMetricStats
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
        audio = self.select_layers(audio)
        spk_emb = (
            batch.spk_emb_random_match.data
            if self.hparams.spk_emb_shuffle
            else batch.spk_emb.data
        )
        spk_emb = spk_emb.squeeze(1)
        predictions = self.modules.model(
            input_tokens=tokens,
            input_length=tokens_length,
            audio=audio,
            audio_length=audio_length,
            emb={"spk": spk_emb},
        )

        return predictions

    def _get_selected_layer_idx(self):
        selected_layers = None
        if (
            hasattr(self.hparams, "select_layers")
            and self.hparams.select_layers
        ):
            layers = self.hparams.select_layers
            model_layers_map = {
                layer: idx
                for idx, layer in enumerate(self.hparams.token_model_layers)
            }
            selected_layers = [model_layers_map[layer] for layer in layers]
        return selected_layers

    def select_layers(self, audio_ssl):
        """Applies layer squishing, if enabled

        Arguments
        ---------
        audio_ssl : torch.Tensor
            SSL features

        Returns
        -------
        audio_ssl : torch.Tensor
            SSL features, squished if enabled
        """
        if self.layer_idx:
            audio_ssl = audio_ssl[:, :, self.layer_idx]
        return audio_ssl

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
        audio = self.select_layers(audio)

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
        loss = loss.detach().cpu()
        if self.is_evaluating():
            with torch.no_grad():
                tokens, tokens_length = batch.tokens
                emb = {"spk": batch.spk_emb.data.squeeze(1)}
                # Produce samples
                infer_out = self.modules.model.infer(
                    input_tokens=tokens, input_length=tokens_length, emb=emb
                )
                with self.hparams.progress_report:
                    # Save samples
                    self.hparams.progress_report.write(
                        ids=batch.uttid,
                        audio=infer_out.wav,
                        length_pred=infer_out.wav_length,
                        length=batch.audio_pad.lengths,
                        tgt_max_length=batch.audio_pad.data.size(1),
                        alignments=infer_out.alignments,
                        p_eos=infer_out.p_eos,
                    )
                    # Evaluate
                    self.evaluation_metric.append(
                        ids=batch.uttid,
                        wav=infer_out.wav,
                        text=batch.label_norm_eval,
                        length=infer_out.wav_length,
                        wav_ref=batch.sig.data,
                        length_ref=batch.sig.lengths,
                    )
        return loss

    def is_evaluating(self):
        """Determines if evaluation should happen in the current epoch"""
        epoch = self.hparams.epoch_counter.current
        return epoch is None or epoch % self.hparams.eval_interval == 0

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
        self.layer_idx = self._get_selected_layer_idx()
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
            # TODO: Clean this up
            if hasattr(self.hparams.token_model, "vocabulary"):
                vocabulary = self.hparams.token_model.vocabulary
            elif hasattr(self.hparams.token_model, "vocabularies"):
                vocabulary = torch.stack(
                    [
                        torch.from_numpy(voc)
                        for voc in self.hparams.token_model.vocabularies
                    ]
                )
            self.modules.model.init_audio_emb(vocabulary)
        # Load the compression model only if compression is enables
        self.compression = getattr(self.hparams, "compression", False)
        pretrained_run_opts = {"device": self.device}
        if self.compression:
            self.compression_model = self.hparams.compression_model(
                run_opts=pretrained_run_opts
            )
            self.modules.model.compression_model = self.compression_model

        # If speaker embedding shuffling is enabled, re-initialize them for the
        # epoch
        if self.hparams.spk_emb_shuffle:
            stage_key = stage.name.lower()
            resample_fn[stage_key](epoch=epoch)

        # Reset the learning rate - if supported. This is useful when fine-tuning
        # a model pre-trained on another dataset
        if (
            stage == sb.Stage.TRAIN
            and self.hparams.reset_annealing_epoch is not None
            and epoch is not None
            and epoch == self.hparams.reset_annealing_epoch
        ):
            self.hparams.lr_annealing.n_steps = 0

        if not hasattr(self, "evaluation_metric"):
            self.evaluation_metric = SpeechEvaluationMetricStats(
                self.hparams, self.device
            )
        if stage != sb.Stage.TRAIN and self.is_evaluating():
            self.hparams.progress_logger.clear()
            self.evaluation_metric.on_evaluation_start()

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit/compiled modules
        # cannot be pickled.
        self._compile()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None and not getattr(
            self, "_ckpt_recovered", False
        ):
            self.checkpointer.recover_if_possible()
            self._ckpt_recovered = True

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

        if stage != sb.Stage.TRAIN and self.is_evaluating():
            self.hparams.progress_report.flush()
            self.evaluation_metric.on_evaluation_end()
            self.save_eval(stage)

    def save_eval(self, stage):
        """Saves evaluation results"""
        with self.hparams.progress_logger:
            stage_folder = Path(stage.name.lower())
            for file_name in self.evaluation_metric.files:
                with open(file_name) as eval_file:
                    content = eval_file.read()
                self.hparams.progress_logger.save(
                    name=str(stage_folder / file_name.name), content=content
                )
        self.evaluation_metric.clear()

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

    def create_perfect_samples(self):
        """Creates the best samples that can be created using
        the vocoder provided, for comparison purposes"""
        if not self.hparams.progress_logger["perfect_samples_created"]:
            sample_loader = sb.dataio.dataloader.make_dataloader(
                self.sample_data, **self.hparams.sample_dataloader_opts
            )
            for batch in sample_loader:
                batch = batch.to(self.device)
                sample_tokens, length = batch.audio_pad
                sample_tokens = self.select_layers(sample_tokens)
                vocoder_out = self.vocoder(
                    sample_tokens, length, spk=batch.spk_emb.data.squeeze(1)
                )
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

    def vocoder(self, audio, length, spk):
        vocoder_kwargs = {}
        if self.hparams.vocoder_takes_spk_emb:
            vocoder_kwargs["spk"] = spk
        return self.modules.vocoder(audio, length, **vocoder_kwargs)


INPUT_FEATURE_MAP = {"text": "label_norm", "phonemes": "phn"}


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
    representation_mode = RepresentationMode(
        hparams.get("representation_mode", RepresentationMode.DISCRETE)
    )

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

    use_silence_padding = hparams.get("use_silence_padding", True)
    if "token_model_layers" in hparams:
        audio_tokens_per_step = len(hparams["token_model_layers"])
    else:
        audio_tokens_per_step = hparams["audio_tokens_per_step"]
    if use_silence_padding:
        silence_token, silence_emb = get_silence_token(
            hparams["token_model"],
            extract_emb=True,
            model_kwargs=hparams.get("token_model_kwargs"),
        )
    else:
        silence_token = (
            torch.ones(audio_tokens_per_step, dtype=torch.int64)
            * hparams["eos_index"]
        )

    silence_padding = (
        silence_token
        if representation_mode == RepresentationMode.DISCRETE
        else silence_emb
    )
    silence_padding = silence_padding.cpu()
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

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def sig_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    def spk_emb_random_match(uttid, dataset, spk_sample):
        # Sample a speaker-matched embedding
        selected_idx = spk_sample[uttid]

        # Retrieve the embedding value from the dataset
        with dataset.output_keys_as(["spk_emb"]):
            spk_emb = dataset[selected_idx]["spk_emb"]
        return spk_emb

    dynamic_items = [text_pipeline, tokens_pipeline, audio_pipeline]
    output_keys = ["uttid", "tokens", "audio_pad", "audio_bos", "spk_emb"]

    if hparams["spk_emb_shuffle"]:
        output_keys.append("spk_emb_random_match")

    init_sequence_encoder(hparams)

    resample_fn = {}
    for dataset in data_info:
        dataset_dynamic_items = list(dynamic_items)
        dataset_output_keys = list(output_keys)
        if dataset != "train":
            dataset_dynamic_items.append(sig_pipeline)
            dataset_output_keys += ["sig", "label_norm_eval"]

        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=dataset_dynamic_items,
            output_keys=dataset_output_keys,
        )

        add_prepared_features(
            dataset=dynamic_dataset,
            save_path=Path(hparams["prepare_save_folder"]) / "features",
            id_key="uttid",
            features=[audio_features, "spk_emb"],
        )

        datasets[dataset] = dynamic_dataset
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False
        if hparams["spk_emb_shuffle"]:
            spk_idx, spk_samplers = group_by_speaker(dynamic_dataset, hparams)
            spk_sample = {}
            spk_emb_random_match_pipeline = partial(
                spk_emb_random_match,
                spk_sample=spk_sample,
                dataset=dynamic_dataset.filtered_sorted(),
            )
            dynamic_dataset.add_dynamic_item(
                func=spk_emb_random_match_pipeline,
                takes=["uttid"],
                provides=["spk_emb_random_match"],
            )
            resample_fn[dataset] = partial(
                resample_spk,
                spk_idx=spk_idx,
                sample=spk_sample,
                dataset=dynamic_dataset,
                spk_samplers=spk_samplers,
            )
            resample_fn[dataset](epoch=0)

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

    # Exclude samples without phonemes
    if hparams["input"] == "phonemes":
        for key in datasets:
            datasets[key] = datasets[key].filtered_sorted(
                key_test={"phn": lambda value: value}
            )

    datasets["sample"] = select_sample(hparams, datasets)
    filter_valid(hparams, datasets)
    return datasets, silence_padding, resample_fn


def filter_valid(hparams, datasets):
    """Filters the validation set, if applicable

    Arguments
    ---------
    hparams : dict
        experiment hyperparameters
    datasets : dict
        a dictionary of datasets
    """
    valid_sample_count = hparams.get("valid_sample_count")
    if valid_sample_count:
        datasets["valid"] = datasets["valid"].filtered_sorted(
            select_n=valid_sample_count
        )


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


def group_by_speaker(dataset, hparams):
    """Groups utterance IDs in a dataset by speaker, for selection. The selection
    is stable based on the seed - calling this method multiple times will always
    result in the same order

    Arguments
    ---------
    dataset : torch.Tensor
        the dataset from which to select items
    hparams : dict
        hyperparameters

    Returns
    -------
    spk_idx : dict
        a str -> int dictionary with a list of utterance indexes
        for every speaker
    spk_samplers : dict
        a reproducible sampler for every speaker
    spk_samplers_it : dict
        an iterator for each sampler
    """
    spk_idx = {}
    spk_samplers = {}
    speakers = []
    generator = torch.Generator()
    generator.manual_seed(hparams["seed"])

    # Group by speaker
    with dataset.output_keys_as(["spk_id"]):
        for idx, item in enumerate(dataset):
            spk_id = item["spk_id"]
            if spk_id not in spk_idx:
                spk_idx[spk_id] = []
            spk_idx[spk_id].append(idx)
            speakers.append(spk_id)

    # Create a reproducible sampler
    for spk_id in speakers:
        sampler = hparams["spk_sampler"](data_source=spk_idx[spk_id])
        spk_samplers[spk_id] = sampler

    return spk_idx, spk_samplers


def resample_spk(sample, spk_idx, spk_samplers, dataset, epoch):
    """Selects new samples and updates the sample dictionary
    provided

    Arguments
    ---------
    sample : dict
        the sample dictionary
    spk_idx : dict
        Data item indexes grouped by speaker
    spk_samplers : dict
        A sampler for each speaker
    dataset : speechbrain.dataio.dataset.DynamicItemDataset
    epoch : int
        The epoch number
    """
    if epoch is None:
        epoch = 0
    spk_samplers_it = {}
    for spk_id, sampler in spk_samplers.items():
        sampler.set_epoch(epoch)
        spk_samplers_it[spk_id] = iter(sampler)
    with dataset.output_keys_as(["uttid", "spk_id"]):
        for item in dataset:
            spk_item_idx = next(spk_samplers_it[item["spk_id"]])
            dataset_item_idx = spk_idx[item["spk_id"]][spk_item_idx]
            sample[item["uttid"]] = dataset_item_idx


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


RE_PUNCTUATION = re.compile(
    "|".join(re.escape(char) for char in string.punctuation)
)


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        yaml = fin.read()

    eval_hparams_file = Path(hparams_file).parent / "eval.yaml"
    if eval_hparams_file.exists():
        logger.info(
            "Using evaluation hyperparameters from %s", eval_hparams_file
        )
        with open(eval_hparams_file) as eval_hparams:
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

    from libritts_prepare import prepare_libritts

    # Data preparation, to be run on only one process.
    representation_mode = RepresentationMode(
        hparams.get("representation_mode", RepresentationMode.DISCRETE)
    )
    audio_features = (
        "audio_tokens"
        if representation_mode == RepresentationMode.DISCRETE
        else "audio_ssl"
    )
    if not hparams["skip_prep"]:
        with hparams["freezer"]:
            extract_features = [audio_features, "spk_emb"]
            if hparams["input"] == "phonemes":
                extract_features.append("phn")
            run_on_main(
                prepare_libritts,
                kwargs={
                    "data_folder": hparams["data_folder"],
                    "save_folder": hparams["prepare_save_folder"],
                    "save_json_train": hparams["train_json"],
                    "save_json_valid": hparams["valid_json"],
                    "save_json_test": (
                        hparams["test_json"]
                        if "test" in hparams["splits"]
                        else None
                    ),
                    "sample_rate": hparams["sample_rate"],
                    "train_split": hparams["train_split"],
                    "valid_split": hparams["valid_split"],
                    "test_split": (
                        hparams["test_split"]
                        if "test" in hparams["splits"]
                        else None
                    ),
                    "extract_features": extract_features,
                    "seed": hparams["seed"],
                    "extract_features_opts": hparams["extract_features_opts"],
                    "model_name": hparams["model"].__class__.__name__,
                    "device": run_opts.get("device", "cpu"),
                },
            )

    # We can now directly create the datasets for training, valid, and test
    (datasets, silence_padding, resample_fn) = dataio_prepare(hparams)

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
    tts_brain.resample_fn = resample_fn

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
