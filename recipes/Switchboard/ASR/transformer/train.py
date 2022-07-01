#!/usr/bin/env python3
"""Recipe for training a Transformer ASR system with Switchboard.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/transformer.yaml

With the default hyperparameters, the system employs a convolutional frontend and a transformer.
The decoder is based on a Transformer decoder. Beamsearch coupled with a Transformer
language model is used  on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
Switchboard dataset (~300 h).

The best model is the average of the checkpoints from last 5 epochs.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE), and many
other possible variations.


Authors
 * Jianyuan Zhong 2020
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020, 2021, 2022
 * Titouan Parcollet 2021, 2022
 * Dominik Wagner 2022
"""
import csv
import os
import re
import sys
from collections import defaultdict

import torch
import logging
from pathlib import Path

import torchaudio

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):

    def __init__(self, modules=None, opt_class=None, hparams=None,
                 run_opts=None, checkpointer=None, profiler=None):

        self.glm_alternatives = self._read_glm_csv(hparams["output_folder"])

        super().__init__(modules=modules,
                         opt_class=opt_class,
                         hparams=hparams,
                         run_opts=run_opts,
                         checkpointer=checkpointer,
                         profiler=profiler,
                         )

    def _read_glm_csv(self, save_folder):
        alternatives_dict = defaultdict(list)
        # additional GEEZ --> JEEZ
        # HE IS --> HE'S
        # THAT IS --> THAT's
        with open(os.path.join(save_folder, "glm.csv")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                alternatives = row[1].split("|")
                alternatives_dict[row[0]] += alternatives
        return alternatives_dict

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.modules.CNN(feats)

        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficiency, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, hyps,) = predictions

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
        ).sum()

        # now as training progresses we use real prediction from the prev step instead of teacher forcing

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum()

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.words]

                # Check for possible word alternatives and exclusions
                if stage == sb.Stage.TEST:
                    target_words, predicted_words = self.normalize_words(target_words, predicted_words)

                self.wer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def expand_contractions(self, text) -> list:
        # specific
        text = re.sub(r"won\'t", "WILL NOT", text, flags=re.IGNORECASE)
        text = re.sub(r"can\'t", "CAN NOT", text, flags=re.IGNORECASE)
        text = re.sub(r"let\'s", "LET US", text, flags=re.IGNORECASE)
        text = re.sub(r"ain\'t", "AM NOT", text, flags=re.IGNORECASE)
        text = re.sub(r"y\'all", "YOU ALL", text, flags=re.IGNORECASE)
        text = re.sub(r"can\'t", "CANNOT", text, flags=re.IGNORECASE)
        text = re.sub(r"can not", "CANNOT", text, flags=re.IGNORECASE)
        text = re.sub(r"\'cause", "BECAUSE", text, flags=re.IGNORECASE)

        # general
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
        parsed_batch = []
        for batch in text_batch:
            # Remove incomplete words
            # batch = [re.sub(r"^-", "", t, flags=re.IGNORECASE) for t in batch]
            batch = [t for t in batch if not t.startswith("-")]
            text = " ".join(batch)
            parsed = self.expand_contractions(text)
            parsed_batch.append(parsed)
        return parsed_batch

    def normalize_words(self, target_words_batch, predicted_words_batch):
        """
        Remove some references and hypotheses we don't want to score.
        We remove incomplete words (i.e. words that start with "-"), expand common contractions (e.g. I'v -> I have),
        and split linked words (e.g. pseudo-rebel -> pseudo rebel).
        Then we check if some of the predicted words have mapping rules according to the glm (alternatives) file.
        Finally, we check if a predicted word is on the exclusion list.
        The exclusion list contains stuff like "MM", "HM", "AH", "HUH", which gets mapped by the glm file,
        into hesitations. The goal is to remove all the things that appear in the reference as optionally
        deletable (inside parentheses), as if we delete these there is no loss, while
        if we get them correct there is no gain.

        In Kaldi, the filtering procedure looks like this (seeh local/score.sh):

        cp ${ctm} ${score_dir}/tmpf;
                cat ${score_dir}/tmpf | grep -i -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
                grep -i -v -E '<UNK>' | \
                grep -i -v -E ' (UH|UM|EH|MM|HM|AH|HUH|HA|ER|OOF|HEE|ACH|EEE|EW)$' | \
                grep -v -- '-$' > ${ctm};


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
        excluded_words = ["[NOISE]", "[LAUGHTER]",
                          "[VOCALIZED-NOISE]", "[VOCALIZED", "NOISE]", "<UNK>", "UH", "UM", "EH",
                          "MM", "HM", "AH", "HUH", "HA", "ER", "OOF",
                          "HEE", "ACH", "EEE", "EW"]

        target_words_batch = self.expand_contractions_batch(target_words_batch)
        predicted_words_batch = self.expand_contractions_batch(predicted_words_batch)

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

        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            self.scaler.scale(loss / self.grad_accumulation_factor).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer_step += 1

                # anneal lr every update
                self.hparams.noam_annealing(self.optimizer)
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.optimizer_step += 1

                # anneal lr every update
                self.hparams.noam_annealing(self.optimizer)

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint average if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


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
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
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

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

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

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav", "channel", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav, channel, start, stop):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).

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

        if hparams["speed_perturb"]:
            # sig = sb.dataio.dataio.read_audio(wav)
            # factor = np.random.uniform(0.95, 1.05)
            # sig = resample(sig.numpy(), 16000, int(16000*factor))
            speed = sb.processing.speech_augmentation.SpeedPerturb(
                16000, [x for x in range(95, 105)]
            )
            resampled = speed(resampled.unsqueeze(0)).squeeze(
                0
            )  # torch.from_numpy(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

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

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from switchboard_prepare import prepare_switchboard  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

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

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, tokenizer = dataio_prepare(hparams)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

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
            test_datasets[k],
            max_key="ACC",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )