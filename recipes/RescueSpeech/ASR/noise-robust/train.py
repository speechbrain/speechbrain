#!/usr/bin/env python3
"""Recipe for noise robust speech recognition. It provides a simple
combination of a unfrozen speech enhancement model (SepFormer already
fine-tuned on noisy RescueSpeech) and a speech recognition model
(fine-tuned on clean RescueSpeech). The ASR employs Whisper encoder
-decoder to fine-tune on the NLL.

The training is performed jointly allowing both enhancement and
ASR model to update its weight.

This is an adaption from LibriSpeech Whisper recipe.

To run this recipe, do the following:
> python train.py hparams/robust_asr_16k.yaml

Authors
 * Sangeet Sagar 2023
 * Adel Moumen 2022
 * Titouan Parcollet 2022
"""

import os
import sys
import csv
import logging
import numpy as np
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

import torch
import torchaudio
import torch.nn.functional as F
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import undo_padding

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.clean_sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos

        predictions, clean = self.compute_forward_enhance(batch, stage)

        # Enhanced signal is to be fed into ASR
        wavs = predictions[0]

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                if self.hparams.do_augmentation:
                    wavs = self.hparams.augmentation(wavs, wav_lens)

        # We compute the padding mask and replace the values with the pad_token_id
        # that the Whisper decoder expect to see.
        abs_tokens_lens = (bos_tokens_lens * bos_tokens.shape[1]).long()
        pad_mask = (
            torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
            < abs_tokens_lens[:, None]
        )
        bos_tokens[~pad_mask] = self.tokenizer.pad_token_id

        # Forward encoder + decoder
        enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)

        log_probs = self.hparams.log_softmax(logits)

        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _ = self.hparams.valid_greedy_searcher(enc_out, wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_beam_searcher(enc_out, wav_lens)

        return predictions, clean, [log_probs, hyps, wav_lens]

    def compute_forward_enhance(self, batch, stage):
        """Forward computations from the noisy to the separated signals.
        """
        noisy = batch.noisy_sig
        clean = batch.clean_sig
        noise = batch.noise_sig[0]

        # Unpack lists and put tensors in the right device
        noisy, noisy_lens = noisy
        noisy, noisy_lens = noisy.to(self.device), noisy_lens.to(self.device)
        # Convert clean to tensor
        clean = clean[0].unsqueeze(-1).to(self.device)

        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    noisy, clean = self.add_speed_perturb(clean, noisy_lens)

                    # Reverb already added, not adding any reverb
                    clean_rev = clean
                    noisy = clean.sum(-1)
                    # if we reverberate, we set the clean to be reverberant
                    if not self.hparams.dereverberate:
                        clean = clean_rev

                    noise = noise.to(self.device)
                    len_noise = noise.shape[1]
                    len_noisy = noisy.shape[1]
                    min_len = min(len_noise, len_noisy)

                    # add the noise
                    noisy = noisy[:, :min_len] + noise[:, :min_len]

                    # fix the length of clean also
                    clean = clean[:, :min_len, :]

                if self.hparams.use_wavedrop:
                    noisy = self.hparams.wavedrop(noisy, noisy_lens)

                if self.hparams.limit_training_signal_len:
                    noisy, clean = self.cut_signals(noisy, clean)

        # Separation
        if self.use_freq_domain:
            noisy_w = self.compute_feats(noisy)
            est_mask = self.modules.masknet(noisy_w)

            sep_h = noisy_w * est_mask
            est_source = self.hparams.resynth(torch.expm1(sep_h), noisy)
        else:
            noisy_w = self.hparams.enhance_model["Encoder"](noisy)
            est_mask = self.modules.masknet(noisy_w)

            sep_h = noisy_w * est_mask
            est_source = self.hparams.enhance_model["Decoder"](sep_h[0])

        # T changed after conv1d in encoder, fix it here
        T_origin = noisy.size(1)
        T_est = est_source.size(1)
        est_source = est_source.squeeze(-1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin]

        return [est_source, sep_h], clean.squeeze(-1)

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        log_probs, hyps, wav_lens, = predictions
        batch = batch.to(self.device)
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        loss = self.hparams.nll_loss(
            log_probs, tokens_eos, length=tokens_eos_lens,
        )

        if stage != sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens

            # Decode token terms to words
            predicted_words = self.tokenizer.batch_decode(
                hyps, skip_special_tokens=True
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(
                target_words, skip_special_tokens=True
            )

            if hasattr(self.hparams, "normalized_transcripts"):
                predicted_words = [
                    self.tokenizer._normalize(text).split(" ")
                    for text in predicted_words
                ]

                target_words = [
                    self.tokenizer._normalize(text).split(" ")
                    for text in target_words
                ]
            else:
                predicted_words = [text.split(" ") for text in predicted_words]

                target_words = [text.split(" ") for text in target_words]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def compute_objectives_enhance(self, predictions, clean):
        """Computes the si-snr loss"""
        predicted_wavs, predicted_specs = predictions

        if self.use_freq_domain:
            target_specs = self.compute_feats(clean)
            loss = self.hparams.enhance_loss(target_specs, predicted_specs)
        else:
            loss = self.hparams.enhance_loss(
                clean.unsqueeze(-1), predicted_wavs.unsqueeze(-1)
            )
        return loss.mean()

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""

        predictions, clean, outputs = self.compute_forward(
            batch, sb.Stage.TRAIN
        )
        enhance_loss = (
            self.compute_objectives_enhance(predictions, clean)
            * self.hparams.sepformer_weight
        )
        loss = (
            self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            * self.hparams.asr_weight
        )
        loss = torch.add(enhance_loss, loss)

        if loss.requires_grad:
            loss.backward()

        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions, clean, outputs = self.compute_forward(batch, stage=stage)

        with torch.no_grad():
            enhance_loss = (
                self.compute_objectives_enhance(predictions, clean)
                * self.hparams.sepformer_weight
            )
            loss = (
                self.compute_objectives(outputs, batch, stage=stage)
                * self.hparams.asr_weight
            )
            loss = torch.add(enhance_loss, loss)

        if stage != sb.Stage.TRAIN:
            self.pesq_metric.append(
                ids=batch.id, predict=predictions[0].cpu(), target=clean.cpu()
            )
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            # Define function taking (prediction, target) for parallel eval
            def pesq_eval(pred_wav, target_wav):
                """Computes the PESQ evaluation metric"""
                psq_mode = (
                    "wb" if self.hparams.enhance_sample_rate == 16000 else "nb"
                )
                try:
                    return pesq(
                        fs=self.hparams.enhance_sample_rate,
                        ref=target_wav.numpy(),
                        deg=pred_wav.numpy(),
                        mode=psq_mode,
                    )
                except Exception:
                    print("pesq encountered an error for this data item")
                    return 0

            self.pesq_metric = MetricStats(
                metric=pesq_eval, n_jobs=1, batch_eval=False
            )
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

        # Freeze models before training
        else:
            for model in self.hparams.frozen_models:
                if (
                    hasattr(self.hparams, "unfreeze_epoch")
                    and epoch >= self.hparams.unfreeze_epoch
                    and (
                        not hasattr(self.hparams, "unfrozen_models")
                        or model in self.hparams.unfrozen_models
                    )
                ):
                    self.modules[model].train()
                    for p in self.modules[model].parameters():
                        p.requires_grad = True  # Model's weight will be updated
                else:
                    self.modules[model].eval()
                    for p in self.modules[model].parameters():
                        p.requires_grad = False  # Model is frozen

    def on_evaluate_start(self, max_key=None, min_key=None):
        self.checkpointer.recover_if_possible(max_key=max_key, min_key=min_key)
        checkpoints = self.checkpointer.find_checkpoints(
            min_key=min_key,
            max_key=max_key,
            max_num_checkpoints=self.hparams.checkpoint_avg,
        )
        for model in self.modules:
            if (
                model not in self.hparams.frozen_models
                or hasattr(self.hparams, "unfrozen_models")
                and model in self.hparams.unfrozen_models
            ):
                model_state_dict = sb.utils.checkpoints.average_checkpoints(
                    checkpoints, model
                )
                self.modules[model].load_state_dict(model_state_dict)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""

        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["pesq"] = self.pesq_metric.summarize("average")
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_whisper, new_lr_whisper = self.hparams.lr_annealing_whisper(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer, new_lr_whisper
            )

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr_whisper": old_lr_whisper},
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
            with open(self.hparams.test_wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def add_speed_perturb(self, clean, targ_lens):
        """
        Adds speed perturbation and random_shift to the input signals
        (Only for enhance Model)
        """

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_clean = []
            recombine = True

            for i in range(clean.shape[-1]):
                new_target = self.hparams.speedperturb(
                    clean[:, :, i], targ_lens
                )
                new_clean.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(clean.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_clean[i] = new_clean[i].to(self.device)
                    new_clean[i] = torch.roll(
                        new_clean[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    clean = torch.zeros(
                        clean.shape[0],
                        min_len,
                        clean.shape[-1],
                        device=clean.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_clean):
                    clean[:, :, i] = new_clean[i][:, 0:min_len]

        noisy = clean.sum(-1)
        return noisy, clean

    def cut_signals(self, noisy, clean):
        """
        This function selects a random segment of a given length within the noisy.
        The corresponding clean are selected accordingly
        (Only for enhance Model)
        """
        randstart = torch.randint(
            0,
            1 + max(0, noisy.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        clean = clean[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        noisy = noisy[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return noisy, clean

    def fix_sample_rate(self, wavs):
        """
        Fix sample rate of all samples in a batch
        """
        resampled_wavs = []
        for wav in wavs:
            wav = wav.cpu()
            resampled_wavs.append(
                torchaudio.transforms.Resample(
                    self.hparams.enhance_sample_rate,
                    self.hparams.asr_sample_rate,
                )(wav).to(self.device)
            )
        return torch.stack(resampled_wavs, dim=-2)

    def save_audio(self, snt_id, noisy, clean, predictions, batch):
        """
        saves the test audio (noisy, clean, and estimated sources) on disk
        (Only for enhance Model)
        """
        # Create outout folder
        f_name = batch.noisy_wav[0].split("/")[-1].replace(".wav", "")
        save_path = os.path.join(
            self.hparams.output_folder, "enhanced_wavs", f_name
        )
        os.makedirs(save_path, exist_ok=True)

        # Estimated source
        signal = predictions[0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "enhanced.wav")
        torchaudio.save(
            save_file,
            signal.unsqueeze(0).cpu(),
            self.hparams.enhance_sample_rate,
        )

        # Original source
        signal = clean[0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "clean.wav")
        torchaudio.save(
            save_file,
            signal.unsqueeze(0).cpu(),
            self.hparams.enhance_sample_rate,
        )

        # noisy
        signal = noisy[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "noisy.wav")
        torchaudio.save(
            save_file,
            signal.unsqueeze(0).cpu(),
            self.hparams.enhance_sample_rate,
        )

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""
        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        count = 0
        # Variable init
        all_sisnrs = []
        all_sisnrs_i = []
        all_pesqs = []
        all_stois = []
        all_sdrs = []
        all_sdrs_i = []
        csv_columns = [
            "snt_id",
            "snt",
            "sdr",
            "sdr_i",
            "si-snr",
            "si-snr_i",
            "pesq",
            "stoi",
            "csig",
            "cbak",
            "covl",
        ]

        test_loader = sb.dataio.dataloader.make_dataloader(test_data)

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    # Apply Enhancement
                    noisy, noisy_len = batch.noisy_sig
                    snt_id = batch.id
                    clean = batch.clean_sig
                    noisy_wav = (
                        batch.noisy_wav[0].split("/")[-1].replace(".wav", "")
                    )

                    with torch.no_grad():
                        predictions, clean = self.compute_forward_enhance(
                            batch, sb.Stage.TEST
                        )

                    # Write enhanced wavs for sanity check
                    if self.hparams.save_audio:
                        self.save_audio(
                            snt_id[0],
                            batch.noisy_sig,
                            clean,
                            predictions[0],
                            batch,
                        )

                    psq_mode = (
                        "wb"
                        if self.hparams.enhance_sample_rate == 16000
                        else "nb"
                    )

                    try:
                        # Compute SI-SNR
                        sisnr = self.compute_objectives_enhance(
                            predictions, clean
                        )

                        # Compute SI-SNR improvement
                        noisy_signal = noisy

                        noisy_signal = noisy_signal.to(clean.device)
                        sisnr_baseline = self.compute_objectives_enhance(
                            [noisy_signal.squeeze(-1), None], clean
                        )
                        sisnr_i = sisnr - sisnr_baseline

                        # Compute SDR
                        sdr, _, _, _ = bss_eval_sources(
                            clean[0].t().cpu().numpy(),
                            predictions[0][0].t().detach().cpu().numpy(),
                        )

                        sdr_baseline, _, _, _ = bss_eval_sources(
                            clean[0].t().cpu().numpy(),
                            noisy_signal[0].t().detach().cpu().numpy(),
                        )

                        sdr_i = sdr.mean() - sdr_baseline.mean()

                        # Compute PESQ
                        psq = pesq(
                            self.hparams.enhance_sample_rate,
                            clean.squeeze().cpu().numpy(),
                            predictions[0].squeeze().cpu().numpy(),
                            mode=psq_mode,
                        )
                        # Compute STOI
                        stoi_score = stoi(
                            clean.squeeze().cpu().numpy(),
                            predictions[0].squeeze().cpu().numpy(),
                            fs_sig=self.hparams.enhance_sample_rate,
                            extended=False,
                        )

                    except Exception:
                        # this handles all sorts of error that may
                        # occur when evaluating a enhanced file.
                        count += 1
                        continue

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "snt": noisy_wav,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                        "pesq": psq,
                        "stoi": stoi_score,
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())
                    all_pesqs.append(psq)
                    all_stois.append(stoi_score)
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())

                row = {
                    "snt_id": "avg",
                    "snt": "avg",
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                    "pesq": np.array(all_pesqs).mean(),
                    "stoi": np.array(all_stois).mean(),
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean PESQ {}".format(np.array(all_pesqs).mean()))
        logger.info("Mean STOI {}".format(np.array(all_stois).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))
        logger.info("Total discarded files {}".format(count))


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("clean_wav")
    @sb.utils.data_pipeline.provides("clean_sig")
    def audio_pipeline_clean(wav):
        info = torchaudio.info(wav)
        clean_sig = sb.dataio.dataio.read_audio(wav)
        clean_sig = torchaudio.transforms.Resample(
            info.sample_rate, hparams["enhance_sample_rate"],
        )(clean_sig)
        return clean_sig

    @sb.utils.data_pipeline.takes("noise_wav")
    @sb.utils.data_pipeline.provides("noise_sig")
    def audio_pipeline_noise(wav):
        info = torchaudio.info(wav)
        noise_sig = sb.dataio.dataio.read_audio(wav)
        noise_sig = torchaudio.transforms.Resample(
            info.sample_rate, hparams["enhance_sample_rate"],
        )(noise_sig)
        return noise_sig

    @sb.utils.data_pipeline.takes("noisy_wav")
    @sb.utils.data_pipeline.provides("noisy_wav", "noisy_sig")
    def audio_pipeline_noisy(wav):
        info = torchaudio.info(wav)
        noisy_sig = sb.dataio.dataio.read_audio(wav)
        noisy_sig = torchaudio.transforms.Resample(
            info.sample_rate, hparams["enhance_sample_rate"],
        )(noisy_sig)
        return wav, noisy_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_clean)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_noise)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_noisy)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode(wrd)
        # avoid bos and eos tokens.
        tokens_list = tokens_list[1:-1]
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "clean_sig",
            "noise_sig",
            "noisy_sig",
            "tokens_list",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "noisy_wav",
        ],
    )

    return train_data, valid_data, test_data


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

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

    # 1.  # Dataset prep (parsing RescueSpeech dataset)
    from rescuespeech_prepare import prepare_RescueSpeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_RescueSpeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "accented_letters": hparams["accented_letters"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer
    tokenizer.set_prefix_tokens(hparams["language"], "transcribe", False)

    # we need to prepare the tokens for searchers
    hparams["valid_greedy_searcher"].set_decoder_input_tokens(
        tokenizer.prefix_tokens
    )
    hparams["valid_greedy_searcher"].set_language_token(
        tokenizer.prefix_tokens[1]
    )

    hparams["test_beam_searcher"].set_decoder_input_tokens(
        tokenizer.prefix_tokens
    )
    hparams["test_beam_searcher"].set_language_token(tokenizer.prefix_tokens[1])

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Load pre-trained models
    pretrained = "asr_pretrained"
    if pretrained in hparams:
        run_on_main(hparams[pretrained].collect_files)
        hparams[pretrained].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )

    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer

    # determine if frequency domain enhancement or not
    use_freq_domain = hparams.get("use_freq_domain", False)
    asr_brain.use_freq_domain = use_freq_domain

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )

    # Testing
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_loader_kwargs"],
    )

    # Eval
    asr_brain.save_results(test_data)
