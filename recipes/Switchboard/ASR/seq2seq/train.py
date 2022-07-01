#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence ASR system with Switchboard.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch.

To run this recipe, do the following:
> python train.py hparams/train_BPE1000.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard  GRU.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split, and many other possible variations.

This recipe assumes that the tokenizer is already trained.

Authors
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
 * Andreas Nautsch 2021
 * Dominik Wagner 2022
"""
import csv
import os
import re
import sys
from collections import defaultdict

import torch
import logging

import torchaudio

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def __init__(
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
        profiler=None,
    ):

        self.glm_alternatives = self._read_glm_csv(hparams["output_folder"])

        super().__init__(
            modules=modules,
            opt_class=opt_class,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
            profiler=profiler,
        )

    def _read_glm_csv(self, save_folder):
        """Load the ARPA Hub4-E and Hub5-E alternate spellings and contractions map"""
        alternatives_dict = defaultdict(list)
        with open(os.path.join(save_folder, "glm.csv")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                alts = row[1].split("|")
                alternatives_dict[row[0]] += alts
        return alternatives_dict

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                logits = self.modules.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(logits)
                return p_ctc, p_seq, wav_lens
            else:
                return p_seq, wav_lens
        else:
            if stage == sb.Stage.VALID:
                p_tokens, scores = self.hparams.valid_search(x, wav_lens)
            else:
                p_tokens, scores = self.hparams.test_search(x, wav_lens)
            return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.TRAIN:
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                p_ctc, p_seq, wav_lens = predictions
            else:
                p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # Add ctc loss if necessary
        if (
            stage == sb.Stage.TRAIN
            and current_epoch <= self.hparams.number_of_ctc_epochs
        ):
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens
            )
            loss = self.hparams.ctc_weight * loss_ctc
            loss += (1 - self.hparams.ctc_weight) * loss_seq
        else:
            loss = loss_seq

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split()
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split() for wrd in batch.words]

            # Check for possible word alternatives and exclusions
            if stage == sb.Stage.TEST:
                target_words, predicted_words = self.normalize_words(
                    target_words, predicted_words
                )

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def expand_contractions(self, text) -> list:
        """
        Some regular expressions for expanding common contractions and for splitting linked words.

        Parameters
        ----------
        text : str
            Text to process

        Returns
        -------
        A list of tokens
        """
        # Specific contractions
        text = re.sub(r"won\'t", "WILL NOT", text, flags=re.IGNORECASE)
        text = re.sub(r"can\'t", "CAN NOT", text, flags=re.IGNORECASE)
        text = re.sub(r"let\'s", "LET US", text, flags=re.IGNORECASE)
        text = re.sub(r"ain\'t", "AM NOT", text, flags=re.IGNORECASE)
        text = re.sub(r"y\'all", "YOU ALL", text, flags=re.IGNORECASE)
        text = re.sub(r"can\'t", "CANNOT", text, flags=re.IGNORECASE)
        text = re.sub(r"can not", "CANNOT", text, flags=re.IGNORECASE)
        text = re.sub(r"\'cause", "BECAUSE", text, flags=re.IGNORECASE)

        # More general contractions
        text = re.sub(r"n\'t", " NOT", text, flags=re.IGNORECASE)
        text = re.sub(r"\'re", " ARE", text, flags=re.IGNORECASE)
        text = re.sub(r"\'s", " IS", text, flags=re.IGNORECASE)
        text = re.sub(r"\'d", " WOULD", text, flags=re.IGNORECASE)
        text = re.sub(r"\'ll", " WILL", text, flags=re.IGNORECASE)
        text = re.sub(r"\'t", " NOT", text, flags=re.IGNORECASE)
        text = re.sub(r"\'ve", " HAVE", text, flags=re.IGNORECASE)
        text = re.sub(r"\'m", " AM", text, flags=re.IGNORECASE)

        # Split linked words
        text = re.sub(r"-", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s\s+", " ", text)
        text = text.split()
        return text

    def expand_contractions_batch(self, text_batch):
        """
        Wrapper that handles a batch of predicted or
        target words for contraction expansion
        """
        parsed_batch = []
        for batch in text_batch:
            # Remove incomplete words
            batch = [t for t in batch if not t.startswith("-")]
            text = " ".join(batch)
            parsed = self.expand_contractions(text)
            parsed_batch.append(parsed)
        return parsed_batch

    def normalize_words(self, target_words_batch, predicted_words_batch):
        """
        Remove some references and hypotheses we don't want to score.
        We remove incomplete words (i.e. words that start with "-"),
        expand common contractions (e.g. I'v -> I have),
        and split linked words (e.g. pseudo-rebel -> pseudo rebel).
        Then we check if some of the predicted words have mapping rules according
        to the glm (alternatives) file.
        Finally, we check if a predicted word is on the exclusion list.
        The exclusion list contains stuff like "MM", "HM", "AH", "HUH", which would get mapped,
        into hesitations by the glm file anyway.
        The goal is to remove all the things that appear in the reference as optional/deletable
        (i.e. inside parentheses).
        If we delete these tokens, there is no loss,
        and if we recognize them correctly, there is no gain.

        The procedure is adapted from Kaldi's local/score.sh script.

        Parameters
        ----------
        target_words_batch : list
            List of length <batch_size> containing lists of target words for each utterance
        predicted_words_batch : list of list
            List of length <batch_size> containing lists of predicted words for each utterance

        Returns
        -------

        A new list containing the filtered predicted words.

        """
        excluded_words = [
            "[NOISE]",
            "[LAUGHTER]",
            "[VOCALIZED-NOISE]",
            "[VOCALIZED",
            "NOISE]",
            "<UNK>",
            "UH",
            "UM",
            "EH",
            "MM",
            "HM",
            "AH",
            "HUH",
            "HA",
            "ER",
            "OOF",
            "HEE",
            "ACH",
            "EEE",
            "EW",
        ]

        target_words_batch = self.expand_contractions_batch(target_words_batch)
        predicted_words_batch = self.expand_contractions_batch(
            predicted_words_batch
        )

        # Find all possible alternatives for each word in the target utterance
        alternative2tgt_word_batch = []
        for tgt_utterance in target_words_batch:
            alternative2tgt_word = defaultdict(str)
            for tgt_wrd in tgt_utterance:
                # print("tgt_wrd", tgt_wrd)
                alts = self.glm_alternatives[tgt_wrd]
                for alt in alts:
                    if alt != tgt_wrd and len(alt) > 0:
                        alternative2tgt_word[alt] = tgt_wrd
            alternative2tgt_word_batch.append(alternative2tgt_word)

        # See if a predicted word is on the exclusion list
        # and if it matches one of the valid alternatives
        checked_predicted_words_batch = []
        for i, pred_utterance in enumerate(predicted_words_batch):
            alternative2tgt_word = alternative2tgt_word_batch[i]
            checked_predicted_words = []
            for pred_wrd in pred_utterance:
                if pred_wrd in excluded_words:
                    continue
                tgt_wrd = alternative2tgt_word[pred_wrd]
                if len(tgt_wrd) > 0:
                    pred_wrd = tgt_wrd
                checked_predicted_words.append(pred_wrd)
            checked_predicted_words_batch.append(checked_predicted_words)
        return target_words_batch, checked_predicted_words_batch

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
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for _, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "channel", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, channel, start, stop):
        # Select a speech segment from the sph file
        # start and end times are already frames.
        # This is done in data preparation stage.
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        info = torchaudio.info(wav)

        resampled = sig
        # Maybe resample to 16kHz
        if int(info.sample_rate) != int(hparams["sample_rate"]):
            resampled = torchaudio.transforms.Resample(
                info.sample_rate, hparams["sample_rate"],
            )(sig)

        resampled = resampled.transpose(0, 1).squeeze(1)
        # Select the proper audio channel of the segment
        if channel == "A":
            resampled = resampled[:, 0]
        else:
            resampled = resampled[:, 1]
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides(
        "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(words):
        yield words
        tokens_list = tokenizer.encode_as_ids(words)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "words", "tokens_bos", "tokens_eos", "tokens"],
    )
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa
        from speechbrain.dataio.dataloader import SaveableDataLoader  # noqa
        from speechbrain.dataio.batch import PaddedBatch  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        hop_size = dynamic_hparams["feats_hop_size"]

        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: int(float(x["duration"]) * (1 / hop_size)),
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: int(float(x["duration"]) * (1 / hop_size)),
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Switchboard)
    from switchboard_prepare import prepare_switchboard  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_switchboard,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "splits": hparams["splits"],
            "split_ratio": hparams["split_ratio"],
            "skip_prep": hparams["skip_prep"],
            "add_fisher_corpus": hparams["add_fisher_corpus"],
            "max_utt": hparams["max_utt"],
        },
    )

    # create the dataset objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    # Depending on the path given in the hparams YAML file,
    # we download the pretrained LM and Tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {"batch_sampler": train_bsampler}
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_swbd and test_callhome
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
