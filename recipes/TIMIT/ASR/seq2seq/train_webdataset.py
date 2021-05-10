#!/usr/bin/env python3
"""Recipe for training a phoneme recognizer on TIMIT.
The system relies on an encoder, a decoder, and attention mechanisms between them.
Training is done with NLL. CTC loss is also added on the top of the encoder.
Greedy search is using for validation, while beamsearch is used at test time to
improve the system performance.

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder /path/to/TIMIT

This version of the recipe uses webdataset, which needs to be installed
separately.
> pip install webdataset

Authors
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Abdel Heba 2020
 * Aku Rouhe 2021
"""

import os
import sys
import glob
import math
import pathlib
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import webdataset as wds

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                phns_bos = torch.cat([phns_bos, phns_bos])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        e_in = self.modules.emb(phns_bos)
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        if stage == sb.Stage.VALID:
            hyps, scores = self.hparams.greedy_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        elif stage == sb.Stage.TEST:
            hyps, scores = self.hparams.beam_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        return p_ctc, p_seq, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        if stage == sb.Stage.TRAIN:
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, hyps = predictions

        ids = batch.id
        phns_eos, phn_lens_eos = batch.phn_encoded_eos
        phns, phn_lens = batch.phn_encoded

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)
            phns_eos = torch.cat([phns_eos, phns_eos], dim=0)
            phn_lens_eos = torch.cat([phn_lens_eos, phn_lens_eos], dim=0)

        loss_ctc = self.hparams.ctc_cost(p_ctc, phns, wav_lens, phn_lens)
        loss_seq = self.hparams.seq_cost(p_seq, phns_eos, phn_lens_eos)
        loss = self.hparams.ctc_weight * loss_ctc
        loss += (1 - self.hparams.ctc_weight) * loss_seq

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.ctc_metrics.append(ids, p_ctc, phns, wav_lens, phn_lens)
            self.seq_metrics.append(ids, p_seq, phns_eos, phn_lens_eos)
            self.per_metrics.append(
                ids, hyps, phns, None, phn_lens, self.label_encoder.decode_ndim,
            )

        return loss

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
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.seq_metrics = self.hparams.seq_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            logger.info(
                "Average batchsize: "
                + str(
                    self.train_loader.total_samples
                    / self.train_loader.total_steps
                )
            )
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "PER": per,
                },
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nseq2seq loss stats:\n")
                self.seq_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC, seq2seq, and PER stats written to file",
                    self.hparams.wer_file,
                )


def make_shards(hparams):
    """Create TAR shards based on earlier data prep

    The full SpeechBrain dataprep for TIMIT is a bit involved, see the
    timit_prepare.py file. The end result is simple though: basically a list of
    examples, each containing, among other things, a path to a wav file, and
    the phone transcript (TIMIT uses phone sequences instead of word sequences
    like normal ASR datasets).
    """
    if hparams["skip_prep"]:
        return
    for split_name, manifest_path in hparams["manifests"].items():
        examples = sb.dataio.dataio.load_data_json(
            manifest_path, replacements={"data_root": hparams["data_folder"]}
        )
        split_dir = pathlib.Path(hparams["shards_root"]) / split_name
        split_dir.mkdir(exist_ok=True, parents=True)
        shardpattern = f"{split_dir}/split-%06d.tar"
        with wds.ShardWriter(
            shardpattern, maxcount=hparams["examples_per_shard"],
        ) as fo:
            for uttid, data in examples.items():
                sig = sb.dataio.dataio.read_audio(data["wav"])
                # Webdataset samples are represented as dicts
                # __key__ is a special entry, a unique id for each sample
                sample = {
                    "__key__": uttid,
                    "wav.pyd": sig,
                    "phn.txt": data["phn"],
                }
                fo.write(sample)


def wds_dataset_prep(hparams):
    """Create loading pipelines for the data subsets"""
    # 1. List the actual shard files for the data subsets
    train_shards = glob.glob(f"{hparams['shards_root']}/train/split-*.tar")
    valid_shards = glob.glob(f"{hparams['shards_root']}/valid/split-*.tar")
    test_shards = glob.glob(f"{hparams['shards_root']}/test/split-*.tar")

    # 2. Label encoding:
    # Prep a basic loader:
    train_data = (
        wds.WebDataset(train_shards)
        .decode()
        .rename(id="__key__", sig="wav.pyd", phn="phn.txt")
    )
    # Then create the label encoder:
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_iterables=[
            (sample["phn"].strip().split() for sample in train_data)
        ],
        special_labels=special_labels,
        sequence_input=True,
    )

    # Finally make a function which extends each sample:
    def tokenize(sample):
        text = sample["phn"].strip().split()
        phn_encoded = label_encoder.encode_sequence_torch(text, allow_unk=False)
        sample["phn_encoded"] = phn_encoded
        sample["phn_encoded_bos"] = label_encoder.prepend_bos_index(phn_encoded)
        sample["phn_encoded_eos"] = label_encoder.append_eos_index(phn_encoded)
        return sample

    # 3. Now prep the full datasets:
    # The train data can continue from the start above.
    # For train data, we repeat indefinitely, and batch dynamically
    train_data = (
        train_data.map(tokenize)
        .then(
            sb.dataio.iterators.dynamic_bucketed_batch,
            **hparams["dynamic_batch_kwargs"],
        )
        .repeat()
    )
    # For the valid and test data, we need to specify lengths:
    valid_data = (
        wds.WebDataset(
            valid_shards, length=math.ceil(400 / hparams["eval_batchsize"])
        )
        .decode()
        .rename(id="__key__", sig="wav.pyd", phn="phn.txt")
        .map(tokenize)
        .then(
            wds.iterators.batched,
            batchsize=hparams["eval_batchsize"],
            collation_fn=sb.dataio.batch.PaddedBatch,
            partial=True,
        )
    )
    test_data = (
        wds.WebDataset(
            test_shards, length=math.ceil(192 / hparams["eval_batchsize"])
        )
        .decode()
        .rename(id="__key__", sig="wav.pyd", phn="phn.txt")
        .map(tokenize)
        .then(
            wds.iterators.batched,
            batchsize=hparams["eval_batchsize"],
            collation_fn=sb.dataio.batch.PaddedBatch,
            partial=True,
        )
    )

    return train_data, valid_data, test_data, label_encoder


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Find and parse the data (here using existing prep script, which creates
    # JSON files)
    from timit_prepare import prepare_timit  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_timit,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["manifests"]["train"],
            "save_json_valid": hparams["manifests"]["valid"],
            "save_json_test": hparams["manifests"]["test"],
            "skip_prep": hparams["skip_prep"],
        },
    )
    # Then create the shards:
    run_on_main(make_shards, args=[hparams])

    # Finally, make the datasets:
    train_data, valid_data, test_data, label_encoder = wds_dataset_prep(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder
    train_loader = asr_brain.make_dataloader(
        train_data, sb.Stage.TRAIN, **hparams["train_dataloader_opts"]
    )
    asr_brain.train_loader = train_loader

    # Training/validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_loader,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
    )

    # Test
    asr_brain.evaluate(
        test_data, min_key="PER",
    )
