#!/usr/bin/env/python3
"""Recipe for training a grapheme-to-phoneme system with one of the available datasets.

See README.md for more details

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Artem Ploujnikov 2021
"""
from speechbrain.dataio.dataset import (
    FilteredSortedDynamicItemDataset,
    DynamicItemDataset,
)
from speechbrain.dataio.sampler import BalancingDataSampler
from speechbrain.utils.data_utils import undo_padding
import datasets
import logging
import os
import random
import torch
import speechbrain as sb
import sys
from enum import Enum
from collections import namedtuple
from hyperpyyaml import load_hyperpyyaml
from functools import partial
from speechbrain.utils.distributed import run_on_main
from speechbrain.pretrained.training import save_for_pretrained
from speechbrain.lobes.models.g2p.dataio import (
    enable_eos_bos,
    grapheme_pipeline,
    phoneme_pipeline,
    tokenizer_encode_pipeline,
    add_bos_eos,
    get_sequence_key,
    phonemes_to_label,
)
from speechbrain.dataio.wer import print_alignments
from speechbrain.wordemb.util import expand_to_chars
from io import StringIO
from speechbrain.utils import hpopt as hp
import numpy as np

logger = logging.getLogger(__name__)

G2PPredictions = namedtuple(
    "G2PPredictions",
    "p_seq char_lens hyps ctc_logprobs attn",
    defaults=[None] * 4,
)


class TrainMode(Enum):
    """An enumeration that represents the trainining mode

    NORMAL: trains the sequence-to-sequence model
    HOMOGRAPH: fine-tunes a trained model on homographs"""

    NORMAL = "normal"
    HOMOGRAPH = "homograph"


# Define training procedure
class G2PBrain(sb.Brain):
    def __init__(self, train_step_name, *args, **kwargs):
        """Class constructor

        Arguments
        ---------
        train_step_name: the name of the training step, for curriculum learning
        """
        super().__init__(*args, **kwargs)
        self.train_step_name = train_step_name
        self.train_step = next(
            step
            for step in self.hparams.train_steps
            if step["name"] == train_step_name
        )
        self.epoch_counter = self.train_step["epoch_counter"]
        self.mode = TrainMode(train_step.get("mode", TrainMode.NORMAL))
        self.last_attn = None
        self.lr_annealing = getattr(
            self.train_step, "lr_annealing", self.hparams.lr_annealing
        )
        self.phn_key = get_sequence_key(
            key="phn_encoded",
            mode=getattr(self.hparams, "phoneme_sequence_mode", "bos"),
        )
        self.grapheme_key = get_sequence_key(
            key="grapheme_encoded",
            mode=getattr(self.hparams, "grapheme_sequence_mode", "bos"),
        )
        self.beam_searcher = (
            self.hparams.beam_searcher_lm
            if self.hparams.use_language_model
            else self.hparams.beam_searcher
        )
        self.beam_searcher_valid = getattr(
            self.hparams, "beam_searcher_valid", self.beam_searcher
        )
        self.start_epoch = None

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        self._recover_checkpoint()

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Gets called at the beginning of ``evaluate()``

        Default implementation loads the best-performing checkpoint for
        evaluation, based on stored metrics.

        Arguments
        ---------
        max_key : str
            Key to use for finding best checkpoint (higher is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        min_key : str
            Key to use for finding best checkpoint (lower is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        """
        self._recover_checkpoint(min_key, max_key)

    def _recover_checkpoint(self, min_key=None, max_key=None):
        """loads the best-performing checkpoint, based on stored metrics.

        Arguments
        ---------
        max_key : str
            Key to use for finding best checkpoint (higher is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        min_key : str
            Key to use for finding best checkpoint (lower is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        """
        if self.checkpointer is not None:
            step = self.train_step["name"]
            logger.info(f"Attempting to restore checkpoint for step {step}")
            result = self.checkpointer.recover_if_possible(
                device=torch.device(self.device),
                min_key=min_key,
                max_key=max_key,
                ckpt_predicate=(lambda ckpt: ckpt.meta.get("step") == step),
            )
            if result is None:
                logger.info(
                    "No checkpoint fount for step %s, "
                    "attempting to recover any checkpoint",
                    step,
                )
                result = self.checkpointer.recover_if_possible(
                    device=torch.device(self.device),
                    min_key=min_key,
                    max_key=max_key,
                )
                if result:
                    logger.info(
                        "Recovered checkpoint with metadata %s", result.meta
                    )
                else:
                    logger.info("No checkpoint found")

    def compute_forward(self, batch, stage):
        """Forward computations from the char batches to the output probabilities."""
        batch = batch.to(self.device)

        # Get graphemes or phonemes
        graphemes, grapheme_lens = getattr(batch, self.grapheme_key)
        phn_encoded = getattr(batch, self.phn_key)
        word_emb = None
        # Use word embeddings (if applicable)
        if self.hparams.use_word_emb:
            word_emb = self.modules.word_emb.batch_embeddings(batch.char)
            char_word_emb = expand_to_chars(
                emb=word_emb,
                seq=graphemes,
                seq_len=grapheme_lens,
                word_separator=self.grapheme_word_separator_idx,
            )
        else:
            word_emb, char_word_emb = None, None

        # Forward pass through the model
        p_seq, char_lens, encoder_out, attn = self.modules["model"](
            grapheme_encoded=(graphemes.detach(), grapheme_lens),
            phn_encoded=phn_encoded,
            word_emb=char_word_emb,
        )

        self.last_attn = attn

        hyps = None

        # Apply CTC, if applicable
        ctc_logprobs = None
        if stage == sb.Stage.TRAIN and self.is_ctc_active(stage):
            # Output layer for ctc log-probabilities
            ctc_logits = self.modules.ctc_lin(encoder_out)
            ctc_logprobs = self.hparams.log_softmax(ctc_logits)

        if stage != sb.Stage.TRAIN and self.hparams.enable_metrics:
            beam_searcher = (
                self.beam_searcher_valid
                if stage == sb.Stage.VALID
                else self.beam_searcher
            )
            hyps, scores = beam_searcher(encoder_out, char_lens)

        return G2PPredictions(p_seq, char_lens, hyps, ctc_logprobs, attn)

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets.


        Arguments
        ---------
        predictions: G2PPredictions
            the predictions (as computed by compute_forward)
        batch: PaddedBatch
            a raw G2P data batch
        stage: speechbrain.Stage
            the training stage
        """
        phns_eos, phn_lens_eos = batch.phn_encoded_eos
        phns, phn_lens = batch.phn_encoded
        loss_seq = self.hparams.seq_cost(
            predictions.p_seq, phns_eos, phn_lens_eos
        )
        if self.is_ctc_active(stage):
            seq_weight = 1 - self.hparams.ctc_weight
            loss_ctc = self.hparams.ctc_cost(
                predictions.ctc_logprobs,
                phns_eos,
                predictions.char_lens,
                phn_lens_eos,
            )
            loss = seq_weight * loss_seq + self.hparams.ctc_weight * loss_ctc
        else:
            loss = loss_seq
        if self.mode == TrainMode.HOMOGRAPH:
            # When tokenization is used, the length of the words
            # in the original phoneme space is not equal to the tokenized
            # words but the raw data only supplies non-tokenized information
            phns_base, phn_lens_base = (
                batch.phn_raw_encoded
                if self.hparams.phn_tokenize
                else (None, None)
            )
            homograph_loss = (
                self.hparams.homograph_loss_weight
                * self.hparams.homograph_cost(
                    phns=phns,
                    phn_lens=phn_lens,
                    p_seq=predictions.p_seq,
                    subsequence_phn_start=batch.homograph_phn_start,
                    subsequence_phn_end=batch.homograph_phn_end,
                    phns_base=phns_base,
                    phn_lens_base=phn_lens_base,
                )
            )
            loss += homograph_loss

        # Record losses for posterity
        if self.hparams.enable_metrics and stage != sb.Stage.TRAIN:
            self._add_sequence_metrics(predictions, batch)
            if self.mode == TrainMode.HOMOGRAPH:
                self._add_homograph_metrics(predictions, batch)

        return loss

    def _add_sequence_metrics(self, predictions, batch):
        """Extracts the homograph from the sequence, computes metrics for it
        and registers them

        Arguments
        ---------
        predictions: G2PPredictions
            the predictions (as computed by compute_forward)
        batch: PaddedBatch
            a raw G2P data batch
        """
        phns_eos, _ = batch.phn_encoded_eos
        phns, phn_lens = batch.phn_encoded
        self.seq_metrics.append(
            batch.sample_id, predictions.p_seq, phns_eos, phn_lens
        )
        self.per_metrics.append(
            batch.sample_id,
            predictions.hyps,
            phns,
            None,
            phn_lens,
            self.hparams.out_phoneme_decoder,
        )

    def _add_homograph_metrics(self, predictions, batch):
        """Extracts the homograph from the sequence, computes metrics for it
        and registers them

        Arguments
        ---------
        predictions: G2PPredictions
            the predictions (as computed by compute_forward)
        batch: PaddedBatch
            a raw G2P data batch
        """
        phns, phn_lens = batch.phn_encoded
        phns_base, phn_base_lens = (
            batch.phn_raw_encoded if self.hparams.phn_tokenize else (None, None)
        )
        (
            p_seq_homograph,
            phns_homograph,
            phn_lens_homograph,
        ) = self.hparams.homograph_extractor(
            phns,
            phn_lens,
            predictions.p_seq,
            subsequence_phn_start=batch.homograph_phn_start,
            subsequence_phn_end=batch.homograph_phn_end,
            phns_base=phns_base,
            phn_base_lens=phn_base_lens,
        )
        hyps_homograph = self.hparams.homograph_extractor.extract_hyps(
            phns_base if phns_base is not None else phns,
            predictions.hyps,
            batch.homograph_phn_start,
            use_base=True,
        )
        self.seq_metrics_homograph.append(
            batch.sample_id, p_seq_homograph, phns_homograph, phn_lens_homograph
        )
        self.per_metrics_homograph.append(
            batch.sample_id,
            hyps_homograph,
            phns_homograph,
            None,
            phn_lens_homograph,
            self.hparams.out_phoneme_decoder,
        )
        prediction_labels = phonemes_to_label(
            phns=hyps_homograph, decoder=self.hparams.out_phoneme_decoder
        )
        phns_homograph_list = undo_padding(phns_homograph, phn_lens_homograph)
        target_labels = phonemes_to_label(
            phns_homograph_list, decoder=self.hparams.out_phoneme_decoder
        )
        self.classification_metrics_homograph.append(
            batch.sample_id,
            predictions=prediction_labels,
            targets=target_labels,
            categories=batch.homograph_wordid,
        )

    def is_ctc_active(self, stage):
        """Determines whether or not the CTC loss should be enabled.
        It is enabled only if a ctc_lin module has been defined in
        the hyperparameter file, only during training and only for
        the number of epochs determined by the ctc_epochs hyperparameter
        of the corresponding training step.

        Arguments
        ---------
        stage: speechbrain.Stage
            the training stage
        """
        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.epoch_counter.current
        return current_epoch <= self.train_step["ctc_epochs"]

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()

        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.seq_metrics = self.hparams.seq_stats()

        if self.start_epoch is None:
            self.start_epoch = epoch

        if self.hparams.enable_metrics:
            if stage != sb.Stage.TRAIN:
                self.per_metrics = self.hparams.per_stats()

            if self.mode == TrainMode.HOMOGRAPH:
                self.seq_metrics_homograph = self.hparams.seq_stats_homograph()
                self.classification_metrics_homograph = (
                    self.hparams.classification_stats_homograph()
                )

                if stage != sb.Stage.TRAIN:
                    self.per_metrics_homograph = (
                        self.hparams.per_stats_homograph()
                    )

        if self.mode == TrainMode.HOMOGRAPH:
            self._set_word_separator()
        self.grapheme_word_separator_idx = self.hparams.grapheme_encoder.lab2ind[
            " "
        ]
        if self.hparams.use_word_emb:
            self.modules.word_emb = self.hparams.word_emb().to(self.device)

    def _set_word_separator(self):
        """Determines the word separators to be used"""
        if self.hparams.phn_tokenize:
            word_separator_idx = self.hparams.token_space_index
            word_separator_base_idx = self.phoneme_encoder.lab2ind[" "]
        else:
            word_separator_base_idx = (
                word_separator_idx
            ) = self.phoneme_encoder.lab2ind[" "]

        self.hparams.homograph_cost.word_separator = word_separator_idx
        self.hparams.homograph_cost.word_separator_base = (
            word_separator_base_idx
        )
        self.hparams.homograph_extractor.word_separator = word_separator_idx
        self.hparams.homograph_extractor.word_separator_base = (
            word_separator_base_idx
        )

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        elif self.hparams.enable_metrics:
            per = self.per_metrics.summarize("error_rate")
        ckpt_predicate, ckpt_meta, min_keys = None, {}, None
        if stage == sb.Stage.VALID:
            if (
                isinstance(
                    self.hparams.lr_annealing,
                    sb.nnet.schedulers.NewBobScheduler,
                )
                and self.hparams.enable_metrics
            ):
                old_lr, new_lr = self.hparams.lr_annealing(per)
            elif isinstance(
                self.hparams.lr_annealing, sb.nnet.schedulers.ReduceLROnPlateau,
            ):
                old_lr, new_lr = self.hparams.lr_annealing(
                    optim_list=[self.optimizer],
                    current_epoch=epoch,
                    current_loss=self.train_loss,
                )
            else:
                old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            stats = {
                "stats_meta": {"epoch": epoch, "lr": old_lr},
                "train_stats": {"loss": self.train_loss},
                "valid_stats": {"loss": stage_loss},
            }
            if self.hparams.enable_metrics:
                stats["valid_stats"].update(
                    {
                        "seq_loss": self.seq_metrics.summarize("average"),
                        "PER": per,
                    }
                )

                if self.mode == TrainMode.HOMOGRAPH:
                    per_homograph = self.per_metrics_homograph.summarize(
                        "error_rate"
                    )
                    stats["valid_stats"].update(
                        {
                            "seq_loss_homograph": self.seq_metrics_homograph.summarize(
                                "average"
                            ),
                            "PER_homograph": per_homograph,
                        }
                    )
                    ckpt_meta = {"PER_homograph": per_homograph, "PER": per}
                    min_keys = ["PER_homograph"]
                    ckpt_predicate = self._has_homograph_per
                else:
                    ckpt_meta = {"PER": per}
                    min_keys = ["PER"]
                hp.report_result(stats["valid_stats"])

            stats = self._add_stats_prefix(stats)
            self.hparams.train_logger.log_stats(**stats)
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(**stats)
                self.save_samples()
            if (
                self.hparams.ckpt_enable
                and epoch % self.hparams.ckpt_frequency == 0
            ):
                ckpt_meta["step"] = self.train_step["name"]
                self.checkpointer.save_and_keep_only(
                    meta=ckpt_meta,
                    min_keys=min_keys,
                    ckpt_predicate=ckpt_predicate,
                )
            if self.hparams.enable_interim_reports:
                if self.hparams.enable_metrics:
                    self._write_reports(epoch, final=False)

            if self.epoch_counter.should_stop(
                current=epoch, current_metric=per,
            ):
                self.epoch_counter.current = self.epoch_counter.limit

        if stage == sb.Stage.TEST:
            test_stats = {"loss": stage_loss}
            if self.hparams.enable_metrics:
                test_stats["PER"] = per
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.epoch_counter.current},
                test_stats=test_stats,
            )
            if self.hparams.enable_metrics:
                self._write_reports(epoch)

    def _has_homograph_per(self, ckpt):
        """Determines if the provided checkpoint has a homograph PER. Used
        when selecting the best epochs for the homograph loss.

        Arguments
        ---------
        ckpt: speechbrain.utils.checkpoints.Checkpoint
            a checkpoint

        Returns
        -------
        result: bool
            whether it contains a homograph PER"""
        return "PER_homograph" in ckpt.meta

    def _get_interim_report_path(self, epoch, file_path):
        """Determines the path to the interim, per-epoch report

        Arguments
        ---------
        epoch: int
            the epoch number
        file_path: str
            the raw report path
        """
        output_path = os.path.join(
            self.hparams.output_folder,
            "reports",
            self.train_step["name"],
            str(epoch),
        )

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return os.path.join(output_path, os.path.basename(file_path))

    def _get_report_path(self, epoch, key, final):
        """Determines the path in which to save a given report

        Arguments
        ---------
        epoch: int
            the epoch number
        key: str
            the key within the training step definition in the
            hyperparameter file (e.g. "wer_file")
        final: bool
            whether or not this si the final report. If
            final is false, an epoch number will be inserted into the path

        Arguments
        ---------
        file_name: str
            the report file name
        """
        file_name = self.train_step[key]
        if not final:
            file_name = self._get_interim_report_path(epoch, file_name)
        return file_name

    def _write_reports(self, epoch, final=True):
        """Outputs all reports for a given epoch

        Arguments
        ---------
        epoch: int
            the epoch number

        final: bool
            whether or not the reports are final (i.e.
            after the final epoch)

        Returns
        -------
        file_name: str
            the report file name
        """
        wer_file_name = self._get_report_path(epoch, "wer_file", final)
        self._write_wer_file(wer_file_name)
        if self.mode == TrainMode.HOMOGRAPH:
            homograph_stats_file_name = self._get_report_path(
                epoch, "homograph_stats_file", final
            )
            self._write_homograph_file(homograph_stats_file_name)

    def _write_wer_file(self, file_name):
        """Outputs the Word Error Rate (WER) file

        Arguments
        ---------
        file_name: str
            the report file name
        """
        with open(file_name, "w") as w:
            w.write("\nseq2seq loss stats:\n")
            self.seq_metrics.write_stats(w)
            w.write("\nPER stats:\n")
            self.per_metrics.write_stats(w)
            logger.info("seq2seq, and PER stats written to file: %s", file_name)

    def _write_homograph_file(self, file_name):
        """Outputs the detailed homograph report, detailing the accuracy
        percentage for each homograph, as well as the relative frequencies
        of particular output sequences output by the model

        Arguments
        ---------
        file_name: str
            the report file name
        """
        with open(file_name, "w") as w:
            self.classification_metrics_homograph.write_stats(w)

    def _add_stats_prefix(self, stats):
        """
        Adds a training step prefix to every key in the provided statistics
        dictionary

        Arguments
        ---------
        stats: dict
            a statistics dictionary

        Returns
        ---------
        stats: dict
            a prefixed statistics dictionary
        """
        prefix = self.train_step["name"]
        return {
            stage: {
                f"{prefix}_{key}": value for key, value in stage_stats.items()
            }
            for stage, stage_stats in stats.items()
        }

    @property
    def tb_writer(self):
        """Returns the raw TensorBoard logger writer"""
        return self.hparams.tensorboard_train_logger.writer

    @property
    def tb_global_step(self):
        """Returns the global step number in the Tensorboard writer"""
        global_step = self.hparams.tensorboard_train_logger.global_step
        prefix = self.train_step["name"]
        return global_step["valid"][f"{prefix}_loss"]

    def save_samples(self):
        """Saves attention alignment and text samples to the Tensorboard
        writer"""
        self._save_attention_alignment()
        self._save_text_alignments()

    def _save_text_alignments(self):
        """Saves text predictions aligned with lables (a sample, for progress
        tracking)"""
        if not self.hparams.enable_metrics:
            return
        last_batch_sample = self.per_metrics.scores[
            -self.hparams.eval_prediction_sample_size :
        ]
        metrics_by_wer = sorted(
            self.per_metrics.scores, key=lambda item: item["WER"], reverse=True
        )
        worst_sample = metrics_by_wer[
            : self.hparams.eval_prediction_sample_size
        ]
        sample_size = min(
            self.hparams.eval_prediction_sample_size,
            len(self.per_metrics.scores),
        )
        random_sample = np.random.choice(
            self.per_metrics.scores, sample_size, replace=False
        )
        text_alignment_samples = {
            "last_batch": last_batch_sample,
            "worst": worst_sample,
            "random": random_sample,
        }
        prefix = self.train_step["name"]
        for key, sample in text_alignment_samples.items():
            self._save_text_alignment(
                tag=f"valid/{prefix}_{key}", metrics_sample=sample
            )

    def _save_attention_alignment(self):
        """Saves attention alignments"""
        attention = self.last_attn[0]
        if attention.dim() > 2:
            attention = attention[0]
        alignments_max = (
            attention.max(dim=-1)
            .values.max(dim=-1)
            .values.unsqueeze(-1)
            .unsqueeze(-1)
        )
        alignments_output = (
            attention.T.flip(dims=(1,)) / alignments_max
        ).unsqueeze(0)
        prefix = self.train_step["name"]
        self.tb_writer.add_image(
            f"valid/{prefix}_attention_alignments",
            alignments_output,
            self.tb_global_step,
        )

    def _save_text_alignment(self, tag, metrics_sample):
        """Saves a single text sample

        Arguments
        ---------
        tag: str
            the tag - for Tensorboard
        metrics_sample:  list
            List of wer details by utterance,
            see ``speechbrain.utils.edit_distance.wer_details_by_utterance``
            for format. Has to have alignments included.
        """
        with StringIO() as text_alignments_io:
            print_alignments(
                metrics_sample,
                file=text_alignments_io,
                print_header=False,
                sample_separator="\n  ---  \n",
            )
            text_alignments_io.seek(0)
            alignments_sample = text_alignments_io.read()
            alignments_sample_md = f"```\n{alignments_sample}\n```"
        self.tb_writer.add_text(tag, alignments_sample_md, self.tb_global_step)


def sort_data(data, hparams, train_step):
    """Sorts the dataset according to hyperparameter values

    Arguments
    ---------
    data: speechbrain.dataio.dataset.DynamicItemDataset
        the dataset to be sorted
    hparams: dict
        raw hyperparameters
    train_step: dict
        the hyperparameters of the training step

    Returns
    -------
    data: speechbrain.dataio.dataset.DynamicItemDataset
        sorted data
    """
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        data = data.filtered_sorted(sort_key="duration")

    elif hparams["sorting"] == "descending":
        data = data.filtered_sorted(sort_key="duration", reverse=True)

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    sample = train_step.get("sample")
    if sample:
        sample_ids = list(data.data_ids)
        if train_step.get("sample_random"):
            random.shuffle(sample_ids)
        sample_ids = sample_ids[:sample]
        data = FilteredSortedDynamicItemDataset(data, sample_ids)

    return data


def filter_origins(data, hparams):
    """Filters a dataset using a specified list of origins,
    as indicated by the "origin" key in the hyperparameters
    provided

    Arguments
    ---------
    data: speechbrain.dataio.dataset.DynamicItemDataset
        the data to be filtered
    hparams: dict
        the hyperparameters data

    Results
    -------
    data: speechbrain.dataio.dataset.DynamicItemDataset
        the filtered data
    """
    origins = hparams.get("origins")
    if origins and origins != "*":
        origins = set(origins.split(","))
        data = data.filtered_sorted(
            key_test={"origin": lambda origin: origin in origins}
        )
    return data


def filter_homograph_positions(dataset):
    """Removes any defective homograph samples

    Arguments
    ---------
    data: speechbrain.dataio.dataset.DynamicItemDataset
        the data to be filtered

    Results
    -------
    data: speechbrain.dataio.dataset.DynamicItemDataset
        the filtered data
    """
    return dataset.filtered_sorted(
        key_test={
            "homograph_char_end": lambda value: value > 0,
            "homograph_phn_end": lambda value: value > 0,
        }
    )


def validate_hparams(hparams):
    result = True
    supports_homograph = not (
        (
            hparams.get("char_tokenize")
            and not hparams.get("char_token_wordwise")
        )
        or (
            hparams.get("phn_tokenize")
            and not hparams.get("phn_token_wordwise")
        )
    )
    if not supports_homograph and hparams.get("homograph_epochs", 0) > 0:
        logger.error(
            "ERROR: Non-wordwise tokenization is not supported with "
            "homograph disambiguation training"
        )
        result = False
    return result


DATASET_SPLITS = ["train", "valid", "test"]


def load_datasets(hparams, train_step):
    """Flexibly loads the specified dataset. If a custom loader is
    provided, it will be used. Otherwise, it will default to the
    arrow dataset loader"""
    data_folder = hparams["data_folder"]
    dataset = datasets.load_dataset(
        hparams["dataset"], cache_dir=hparams["data_folder"]
    )
    train_step_name = train_step.get("name", "sentence")
    results = [
        DynamicItemDataset.from_arrow_dataset(
            dataset[f"{train_step_name}_{key}"],
            replacements={"data_root": data_folder},
        )
        for key in DATASET_SPLITS
    ]
    return results


# TODO: Split this up into smaller functions
def dataio_prep(hparams, train_step=None):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    Arguments
    ---------
    hparams: dict
        the hyperparameters dictionary

    train_step: dict
        the hyperparameters for the training step being executed

    Returns
    -------
    train_data: speechbrain.dataio.dataset.DynamicItemDataset
        the training dataset

    valid_data: speechbrain.dataio.dataset.DynamicItemDataset
        the validation dataset

    test_data: speechbrain.dataio.dataset.DynamicItemDataset
        the test dataset

    phoneme_encoder: speechbrain.dataio.encoder.TextEncoder
        the phoneme encoder
    """

    if not train_step:
        train_step = hparams

    # 1. Load the datasets:
    train_data, valid_data, test_data = load_datasets(hparams, train_step)

    if hparams["sorting"] == "ascending":
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    is_homograph = (
        TrainMode(train_step.get("mode", TrainMode.NORMAL))
        == TrainMode.HOMOGRAPH
    )

    train_data = sort_data(train_data, hparams, train_step)
    valid_data = sort_data(valid_data, hparams, train_step)
    test_data = sort_data(test_data, hparams, train_step)

    datasets = [train_data, valid_data, test_data]

    phoneme_encoder = hparams["phoneme_encoder"]
    grapheme_encoder = hparams["grapheme_encoder"]

    # 2. Define grapheme and phoneme pipelines:
    enable_eos_bos(
        tokens=hparams["phonemes"],
        encoder=phoneme_encoder,
        bos_index=hparams["bos_index"],
        eos_index=hparams["eos_index"],
    )
    enable_eos_bos(
        tokens=hparams["graphemes"],
        encoder=grapheme_encoder,
        bos_index=hparams["bos_index"],
        eos_index=hparams["eos_index"],
    )
    if hparams.get("char_tokenize"):
        grapheme_pipeline_item = partial(
            tokenizer_encode_pipeline,
            tokenizer=hparams["grapheme_tokenizer"],
            tokens=hparams["graphemes"],
            wordwise=hparams["char_token_wordwise"],
            token_space_index=hparams["token_space_index"],
        )
    else:
        grapheme_pipeline_item = partial(
            grapheme_pipeline, grapheme_encoder=grapheme_encoder
        )

    if hparams.get("phn_tokenize"):
        phoneme_pipeline_item = partial(
            tokenizer_encode_pipeline,
            tokenizer=hparams["phoneme_tokenizer"],
            tokens=hparams["phonemes"],
            char_map=hparams["phn_char_map"],
            wordwise=hparams["phn_token_wordwise"],
            token_space_index=hparams["token_space_index"],
        )
        # Ensure the tokenizers are trained
        if "grapheme_tokenizer" in hparams:
            hparams["grapheme_tokenizer"]()
        if "phoneme_tokenizer" in hparams:
            hparams["phoneme_tokenizer"]()
        enable_eos_bos(
            tokens=hparams["phonemes"],
            encoder=phoneme_encoder,
            bos_index=hparams["bos_index"],
            eos_index=hparams["eos_index"],
        )
    else:
        phoneme_pipeline_item = partial(
            phoneme_pipeline, phoneme_encoder=phoneme_encoder,
        )

    phn_bos_eos_pipeline_item = partial(add_bos_eos, encoder=phoneme_encoder)
    grapheme_bos_eos_pipeline_item = partial(
        add_bos_eos,
        # TODO: Use the grapheme encoder here (this will break some models)
        encoder=phoneme_encoder,
    )

    dynamic_items = [
        {
            "func": grapheme_pipeline_item,
            "takes": ["char"],
            "provides": [
                "grapheme_list",
                "grpaheme_encoded_list",
                "grapheme_encoded",
            ],
        },
        {
            "func": phoneme_pipeline_item,
            "takes": ["phn"],
            "provides": ["phn_list", "phn_encoded_list", "phn_encoded"],
        },
        {
            "func": phn_bos_eos_pipeline_item,
            "takes": ["phn_encoded"],
            "provides": [
                "phn_encoded_bos",
                "phn_len_bos",
                "phn_encoded_eos",
                "phn_len_eos",
            ],
        },
        {
            "func": grapheme_bos_eos_pipeline_item,
            "takes": ["grapheme_encoded"],
            "provides": [
                "grapheme_encoded_bos",
                "grapheme_len_bos",
                "grapheme_encoded_eos",
                "grapheme_len_eos",
            ],
        },
    ]

    if hparams.get("phn_tokenize"):
        # A raw tokenizer is needed to determine the correct
        # word boundaries from data
        phoneme_raw_pipeline = partial(
            phoneme_pipeline, phoneme_encoder=phoneme_encoder,
        )
        dynamic_items.append(
            {
                "func": phoneme_raw_pipeline,
                "takes": ["phn"],
                "provides": [
                    "phn_raw_list",
                    "phn_raw_encoded_list",
                    "phn_raw_encoded",
                ],
            }
        )
    for dynamic_item in dynamic_items:
        sb.dataio.dataset.add_dynamic_item(datasets, **dynamic_item)

    # 3. Set output:
    output_keys = [
        "sample_id",
        "grapheme_encoded",
        "grapheme_encoded_bos",
        "grapheme_encoded_eos",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    ]
    if is_homograph:
        output_keys += [
            "homograph_wordid",
            "homograph_phn_start",
            "homograph_phn_end",
        ]

    if hparams.get("use_word_emb", False):
        output_keys.append("char")
    if (
        hparams.get("phn_tokenize", False)
        and "phn_raw_encoded" not in output_keys
    ):
        output_keys.append("phn_raw_encoded")

    sb.dataio.dataset.set_output_keys(
        datasets, output_keys,
    )
    if "origins" in hparams:
        datasets = [filter_origins(dataset, hparams) for dataset in datasets]

    if train_step.get("mode") == "homograph":
        datasets = [filter_homograph_positions(dataset) for dataset in datasets]

    train_data, valid_data, test_data = datasets
    return train_data, valid_data, test_data, phoneme_encoder


def load_dependencies(hparams, run_opts):
    """Loads any pre-trained dependencies (e.g. language models)

    Arguments
    ---------
    hparams: dict
        the hyperparamters dictionary
    run_opts: dict
        run options
    """
    deps_pretrainer = hparams.get("deps_pretrainer")
    if deps_pretrainer:
        run_on_main(deps_pretrainer.collect_files)
        deps_pretrainer.load_collected(device=run_opts["device"])


def check_tensorboard(hparams):
    """Checks whether Tensorboard is enabled and initializes the logger if it is

    Arguments
    ---------
    hparams: dict
        the hyperparameter dictionary
    """
    if hparams["use_tensorboard"]:
        try:
            from speechbrain.utils.train_logger import TensorboardLogger

            hparams["tensorboard_train_logger"] = TensorboardLogger(
                hparams["tensorboard_logs"]
            )
        except ImportError:
            logger.warning(
                "Could not enable TensorBoard logging - TensorBoard is not available"
            )
            hparams["use_tensorboard"] = False


if __name__ == "__main__":
    # CLI:
    with hp.hyperparameter_optimization(objective_key="PER") as hp_ctx:
        # Set a default PER
        hp.report_result({"PER": 0.0})

        hparams_file, run_opts, overrides = hp_ctx.parse_arguments(sys.argv[1:])

        # Load hyperparameters file with command-line overrides
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Validate hyperparameters
        if not validate_hparams(hparams):
            sys.exit(1)

        # Initialize ddp (useful only for multi-GPU DDP training)
        sb.utils.distributed.ddp_init_group(run_opts)

        if hparams.get("use_language_model"):
            load_dependencies(hparams, run_opts)
        check_tensorboard(hparams)

        from tokenizer_prepare import prepare_tokenizer  # noqa

        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )
        if hparams.get("char_tokenize") or hparams.get("phn_tokenize"):
            path_keys = [
                "grapheme_tokenizer_output_folder",
                "phoneme_tokenizer_output_folder",
            ]
            paths = [hparams[key] for key in path_keys]
            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(path)
            run_on_main(
                prepare_tokenizer,
                kwargs={
                    "dataset_name": hparams.get("dataset"),
                    "data_folder": hparams["data_folder"],
                    "save_folder": hparams["save_folder"],
                    "phonemes": hparams["phonemes"],
                },
            )
        for train_step in hparams["train_steps"]:
            epochs = train_step["epoch_counter"].limit
            if epochs < 1:
                logger.info("Skipping training step: %s", train_step["name"])
                continue
            logger.info("Running training step: %s", train_step["name"])
            # Dataset IO prep: creating Dataset objects and proper encodings for phones
            train_data, valid_data, test_data, phoneme_encoder = dataio_prep(
                hparams, train_step
            )

            # Trainer initialization
            g2p_brain = G2PBrain(
                train_step_name=train_step["name"],
                modules=hparams["modules"],
                opt_class=hparams["opt_class"],
                hparams=hparams,
                run_opts=run_opts,
                checkpointer=hparams["checkpointer"],
            )
            g2p_brain.phoneme_encoder = phoneme_encoder

            # NOTE: This gets modified after the first run and causes a double
            # agument issue
            dataloader_opts = train_step.get(
                "dataloader_opts", hparams.get("dataloader_opts", {})
            )
            if (
                "ckpt_prefix" in dataloader_opts
                and dataloader_opts["ckpt_prefix"] is None
            ):
                del dataloader_opts["ckpt_prefix"]

            train_dataloader_opts = dataloader_opts
            if train_step.get("balance"):
                sampler = BalancingDataSampler(
                    train_data, train_step["balance_on"]
                )
                train_dataloader_opts = dict(dataloader_opts, sampler=sampler)
            start_epoch = train_step["epoch_counter"].current

            # Training/validation loop
            g2p_brain.fit(
                train_step["epoch_counter"],
                train_data,
                valid_data,
                train_loader_kwargs=train_dataloader_opts,
                valid_loader_kwargs=dataloader_opts,
            )

            # Test
            skip_test = hparams.get("skip_test", False)
            if isinstance(skip_test, str):
                skip_test = train_step["name"] in skip_test.split(",")

            if not skip_test:
                g2p_brain.evaluate(
                    test_data,
                    min_key=train_step.get("performance_key"),
                    test_loader_kwargs=dataloader_opts,
                )

            if hparams.get("save_for_pretrained"):
                save_for_pretrained(
                    hparams, min_key=train_step.get("performance_key")
                )
