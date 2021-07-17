#!/usr/bin/env/python3
"""Recipe for training a grapheme-to-phoneme system with librispeech lexicon.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch.

To run this recipe, do the following:
> python train.py hparams/train.yaml

With the default hyperparameters, the system employs an LSTM encoder.
The decoder is based on a standard  GRU. The neural network is trained with
negative-log.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders,  and many other possible variations.


Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
"""
import sys
import speechbrain as sb
from collections import namedtuple
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.pretrained.training import PretrainedModelMixin
from speechbrain.lobes.models.g2p.attnrnn.dataio import (
    grapheme_pipeline,
    phoneme_pipeline,
)
from speechbrain.dataio.wer import print_alignments
from io import StringIO
import numpy as np


G2PPredictions = namedtuple(
    "G2PPredictions", "p_seq char_lens hyps ctc_logprobs attn", defaults=[None] * 4)

# Define training procedure
class G2PBrain(sb.Brain, PretrainedModelMixin):
    def __init__(self, train_step_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_step_name = train_step_name
        self.train_step = next(
            step for step in self.hparams.train_steps
            if step['name'] == train_step_name)
        self.epoch_counter = self.train_step["epoch_counter"]
        self.has_ctc = hasattr(self.hparams, 'ctc_lin')
        self.last_attn = None

    def compute_forward(self, batch, stage):
        """Forward computations from the char batches to the output probabilities."""
        batch = batch.to(self.device)

        graphemes, grapheme_lens = batch.grapheme_encoded
        p_seq, char_lens, encoder_out, attn = self.modules["model"](
            grapheme_encoded=(graphemes.detach(), grapheme_lens),
            phn_encoded=batch.phn_encoded_bos,
        )
        self.last_attn = attn

        hyps = None
        ctc_logprobs = None
        if stage == sb.Stage.TRAIN and self.is_ctc_active(stage):
                # Output layer for ctc log-probabilities
                ctc_logits = self.modules.ctc_lin(encoder_out)
                ctc_logprobs = self.hparams.log_softmax(ctc_logits)

        if stage != sb.Stage.TRAIN:
            hyps, scores = self.hparams.beam_searcher(encoder_out, char_lens)

        return G2PPredictions(p_seq, char_lens, hyps, ctc_logprobs, attn)

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        ids = batch.id
        phns_eos, phn_lens_eos = batch.phn_encoded_eos
        phns, phn_lens = batch.phn_encoded
        graphemes, grapheme_lens = batch.grapheme_encoded
        loss_seq = self.hparams.seq_cost(predictions.p_seq, phns_eos, phn_lens_eos)
        if self.is_ctc_active(stage):
            seq_weight = 1 - self.hparams.ctc_weight
            loss_ctc = self.hparams.ctc_cost(
                predictions.ctc_logprobs,
                phns_eos, predictions.char_lens, phn_lens_eos
            )
            loss = seq_weight + loss_seq + self.hparams.ctc_weight * loss_ctc
        else:
            loss = loss_seq

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.seq_metrics.append(ids, predictions.p_seq, phns_eos, phn_lens)
            self.per_metrics.append(
                ids,
                predictions.hyps,
                phns,
                None,
                phn_lens,
                self.phoneme_encoder.decode_ndim,
            )

        return loss

    def is_ctc_active(self, stage):
        if not self.has_ctc or stage != sb.Stage.TRAIN:
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

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            stats = {
                "stats_meta": {"epoch": epoch, "lr": old_lr},
                "train_stats": {"loss": self.train_loss},
                "valid_stats": {
                    "loss": stage_loss,
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "PER": per,
                },
            }
            stats = self._add_stats_prefix(stats)
            self.hparams.train_logger.log_stats(**stats)
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(**stats)
                self.save_samples()

            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            wer_file = self.train_step['wer_file']
            with open(wer_file, "w") as w:
                w.write("\nseq2seq loss stats:\n")
                self.seq_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "seq2seq, and PER stats written to file",
                    wer_file
                )

    def _add_stats_prefix(self, stats):
        prefix = self.train_step["name"]
        return {
            stage: {f"{prefix}_{key}": value
                    for key, value in stage_stats.items()}
            for stage, stage_stats in stats.items()
        }

    @property
    def tb_writer(self):
        return self.hparams.tensorboard_train_logger.writer

    @property
    def tb_global_step(self):
        global_step = self.hparams.tensorboard_train_logger.global_step
        prefix = self.train_step["name"]
        return global_step["valid"][f"{prefix}_loss"]

    def save_samples(self):
        self._save_attention_alignment()
        self._save_text_alignments()

    def _save_text_alignments(self):
        last_batch_sample = self.per_metrics.scores[
            -self.hparams.eval_prediction_sample_size:]
        metrics_by_wer = sorted(
            self.per_metrics.scores,
            key=lambda item: item["WER"],
            reverse=True)
        worst_sample = metrics_by_wer[:self.hparams.eval_prediction_sample_size]
        sample_size = min(
            self.hparams.eval_prediction_sample_size,
            len(self.per_metrics.scores))
        random_sample = np.random.choice(
            self.per_metrics.scores, sample_size,
            replace=False)
        text_alignment_samples = {
            "last_batch": last_batch_sample,
            "worst": worst_sample,
            "random": random_sample
        }
        prefix = self.train_step["name"]
        for key, sample in text_alignment_samples.items():
            self._save_text_alignment(
                tag=f"valid/{prefix}_{key}",
                metrics_sample=sample)

    def _save_attention_alignment(self):
        attention = self.last_attn[0]
        alignments_max = (
            attention
                .max(dim=-1).values
                .max(dim=-1).values
                .unsqueeze(-1)
                .unsqueeze(-1))
        alignments_output = (
            attention.T.flip(dims=(1,)) / alignments_max).unsqueeze(0)
        prefix = self.train_step["name"]
        self.tb_writer.add_image(
            f"valid/{prefix}_attention_alignments",
            alignments_output, self.tb_global_step)

    def _save_text_alignment(self, tag, metrics_sample):
        with StringIO() as text_alignments_io:
            print_alignments(
                metrics_sample,
                file=text_alignments_io,
                print_header=False,
                sample_separator='\n  ---  \n')
            text_alignments_io.seek(0)
            alignments_sample = text_alignments_io.read()
            alignments_sample_md = f"```\n{alignments_sample}\n```"
        self.tb_writer.add_text(
            tag, alignments_sample_md, self.tb_global_step)



def sort_data(data, hparams):
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        data = data.filtered_sorted(sort_key="duration")

    elif hparams["sorting"] == "descending":
        data = data.filtered_sorted(
            sort_key="duration", reverse=True
        )

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    return data


def dataio_prep(hparams, train_step=None):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    if not train_step:
        train_step = hparams
    data_folder = hparams["data_folder"]
    data_load = hparams["data_load"]
    # 1. Declarations:
    train_data = data_load(
        train_step["train_data"], replacements={"data_root": data_folder},
    )
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
    train_data = sort_data(train_data, hparams)

    valid_data = data_load(
        train_step["valid_data"], replacements={"data_root": data_folder},
    )
    valid_data = sort_data(valid_data, hparams)

    test_data = data_load(
        train_step["test_data"], replacements={"data_root": data_folder},
    )
    test_data = sort_data(test_data, hparams)


    datasets = [train_data, valid_data, test_data]

    phoneme_encoder = sb.dataio.encoder.TextEncoder()

    # 2. Define grapheme pipeline:
    sb.dataio.dataset.add_dynamic_item(
        datasets,
        grapheme_pipeline(
            graphemes=hparams["graphemes"],
            space_separated=hparams['graphemes_space_separated']),
    )
    # 3. Define phoneme pipeline:
    sb.dataio.dataset.add_dynamic_item(
        datasets,
        phoneme_pipeline(
            phonemes=hparams["phonemes"],
            phoneme_encoder=phoneme_encoder,
            bos_index=hparams["bos_index"],
            eos_index=hparams["eos_index"],
            space_separated=hparams["phonemes_space_separated"]
        ),
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "grapheme_encoded",
            "phn_encoded",
            "phn_encoded_eos",
            "phn_encoded_bos",
        ],
    )

    return train_data, valid_data, test_data, phoneme_encoder


def check_language_model(hparams):
    """Checks whether or not the language
       model is being used and makes the necessary
       adjustments"""

    if hparams.get("use_language_model"):
        hparams["beam_searcher"] = hparams["beam_searcher_lm"]
        load_dependencies(hparams)
    else:
        if "beam_searcher_lm" in hparams:
            del hparams["beam_searcher_lm"]

def load_dependencies(hparams):
    deps_pretrainer = hparams.get("deps_pretrainer")
    if deps_pretrainer:
        run_on_main(deps_pretrainer.collect_files)
        deps_pretrainer.load_collected(device=run_opts["device"])


def check_tensorboard(hparams):
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    check_language_model(hparams)
    check_tensorboard(hparams)

    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    if hparams['build_lexicon']:
        # multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_librispeech,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["save_folder"],
                "create_lexicon": True,
                "skip_prep": hparams["skip_prep"],
                "select_n_sentences": hparams.get("select_n_sentences"),
            },
        )
    for train_step in hparams['train_steps']:
        epochs = train_step['epoch_counter'].limit
        if epochs < 1:
            print(f"Skipping training step: {train_step['name']}")
            continue
        print(f"Running training step: {train_step['name']}")
        # Dataset IO prep: creating Dataset objects and proper encodings for phones
        train_data, valid_data, test_data, phoneme_encoder = dataio_prep(hparams, train_step)

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
            "dataloader_opts",
            hparams.get("dataloader_opts", {}))
        if 'ckpt_prefix' in dataloader_opts and dataloader_opts['ckpt_prefix'] is None:
            del dataloader_opts['ckpt_prefix']
        # Training/validation loop
        g2p_brain.fit(
            train_step['epoch_counter'],
            train_data,
            valid_data,
            train_loader_kwargs=dataloader_opts,
            valid_loader_kwargs=dataloader_opts,
        )

        # Test
        g2p_brain.evaluate(
            test_data, min_key="PER", test_loader_kwargs=dataloader_opts,
        )

        if hparams.get("save_for_pretrained"):
            g2p_brain.save_for_pretrained()
