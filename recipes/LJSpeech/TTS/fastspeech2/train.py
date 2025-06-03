"""
 Recipe for training the FastSpeech2 Text-To-Speech model, an end-to-end
 neural text-to-speech (TTS) system introduced in 'FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
synthesis' paper
 (https://arxiv.org/abs/2006.04558)
 To run this recipe, do the following:
 # python train.py hparams/train.yaml
 Authors
 * Sathvik Udupa 2022
 * Yingzhi Wang 2022
 * Pradnya Kandarkar 2023
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.inference.text import GraphemeToPhoneme
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.utils.data_utils import scalarize
from speechbrain.utils.logger import get_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)


class FastSpeech2Brain(sb.Brain):
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics
        """
        self.hparams.progress_sample_logger.reset()
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        self.g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p")
        self.spn_token_encoded = (
            self.input_encoder.encode_sequence_torch(["spn"]).int().item()
        )
        return super().on_fit_start()

    def compute_forward(self, batch, stage):
        """Computes the forward pass
        Arguments
        ---------
        batch: str
            a single batch
        stage: speechbrain.Stage
            the training stage
        Returns
        -------
        the model output
        """
        inputs, _ = self.batch_to_device(batch)

        tokens, durations, pitch, energy, no_spn_seqs, last_phonemes = inputs

        # Forward pass for the silent token predictor module
        if (
            self.hparams.epoch_counter.current
            > self.hparams.train_spn_predictor_epochs
        ):
            self.hparams.modules["spn_predictor"].eval()
            with torch.no_grad():
                spn_preds = self.hparams.modules["spn_predictor"](
                    no_spn_seqs, last_phonemes
                )
        else:
            spn_preds = self.hparams.modules["spn_predictor"](
                no_spn_seqs, last_phonemes
            )

        # Forward pass for the FastSpeech2 module
        (
            predict_mel_post,
            predict_postnet_output,
            predict_durations,
            predict_pitch,
            predict_avg_pitch,
            predict_energy,
            predict_avg_energy,
            predict_mel_lens,
        ) = self.hparams.model(tokens, durations, pitch, energy)

        return (
            predict_mel_post,
            predict_postnet_output,
            predict_durations,
            predict_pitch,
            predict_avg_pitch,
            predict_energy,
            predict_avg_energy,
            predict_mel_lens,
            spn_preds,
        )

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The model generated spectrograms and other metrics from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        x, y, metadata = self.batch_to_device(batch, return_metadata=True)
        self.last_batch = [x[0], y[-2], y[-3], predictions[0], *metadata]
        self._remember_sample([x[0], *y, *metadata], predictions)
        loss = self.hparams.criterion(
            predictions, y, self.hparams.epoch_counter.current
        )
        self.last_loss_stats[stage] = scalarize(loss)
        return loss["total_loss"]

    def _remember_sample(self, batch, predictions):
        """Remembers samples of spectrograms and the batch for logging purposes
        Arguments
        ---------
        batch: tuple
            a training batch
        predictions: tuple
            predictions (raw output of the FastSpeech2
             model)
        """
        (
            tokens,
            spectogram,
            durations,
            pitch,
            energy,
            mel_lengths,
            input_lengths,
            spn_labels,
            labels,
            wavs,
        ) = batch
        (
            mel_post,
            postnet_mel_out,
            predict_durations,
            predict_pitch,
            predict_avg_pitch,
            predict_energy,
            predict_avg_energy,
            predict_mel_lens,
            spn_preds,
        ) = predictions
        self.hparams.progress_sample_logger.remember(
            target=self.process_mel(spectogram, mel_lengths),
            output=self.process_mel(postnet_mel_out, mel_lengths),
            raw_batch=self.hparams.progress_sample_logger.get_batch_sample(
                {
                    "tokens": tokens,
                    "input_lengths": input_lengths,
                    "mel_target": spectogram,
                    "mel_out": postnet_mel_out,
                    "mel_lengths": predict_mel_lens,
                    "durations": durations,
                    "predict_durations": predict_durations,
                    "labels": labels,
                    "wavs": wavs,
                }
            ),
        )

    def process_mel(self, mel, len, index=0):
        """Converts a mel spectrogram to one that can be saved as an image
        sample  = sqrt(exp(mel))
        Arguments
        ---------
        mel: torch.Tensor
            the mel spectrogram (as used in the model)
        len: int
            length of the mel spectrogram
        index: int
            batch index
        Returns
        -------
        mel: torch.Tensor
            the spectrogram, for image saving purposes
        """
        assert mel.dim() == 3
        return torch.sqrt(torch.exp(mel[index][: len[index]]))

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
        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            self.last_epoch = epoch
            lr = self.hparams.noam_annealing.current_lr

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr": lr},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )
            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0
                and epoch >= self.hparams.progress_samples_min_run
            )

            if output_progress_sample:
                logger.info("Saving predicted samples")
                (
                    inference_mel,
                    mel_lens,
                    inf_mel_spn_pred,
                    mel_lens_spn_pred,
                ) = self.run_inference()
                self.hparams.progress_sample_logger.save(epoch)
                self.run_vocoder(
                    inference_mel, mel_lens, sample_type="with_spn"
                )
                self.run_vocoder(
                    inf_mel_spn_pred, mel_lens_spn_pred, sample_type="no_spn"
                )
            # Save the current checkpoint and delete previous checkpoints.
            # UNCOMMENT THIS
            self.checkpointer.save_and_keep_only(
                meta=self.last_loss_stats[stage],
                min_keys=["total_loss"],
            )
        # We also write statistics about test data spectogram to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )

    def run_inference(self):
        """Produces a sample in inference mode with predicted durations."""
        if self.last_batch is None:
            return
        tokens, *_, labels, _ = self.last_batch

        # Generates inference samples without using the silent phoneme predictor
        (
            _,
            postnet_mel_out,
            _,
            _,
            _,
            _,
            _,
            predict_mel_lens,
        ) = self.hparams.model(tokens)

        self.hparams.progress_sample_logger.remember(
            infer_output=self.process_mel(
                postnet_mel_out, [len(postnet_mel_out[0])]
            )
        )

        # Generates inference samples using the silent phoneme predictor

        # Preprocessing required at the inference time for the input text
        # "label" below contains input text
        # "phoneme_labels" contain the phoneme sequences corresponding to input text labels
        # "last_phonemes_combined" is used to indicate whether the index position is for a last phoneme of a word
        phoneme_labels = list()
        last_phonemes_combined = list()

        for label in labels:
            phoneme_label = list()
            last_phonemes = list()

            words = label.split()
            words = [word.strip() for word in words]
            words_phonemes = self.g2p(words)

            for words_phonemes_seq in words_phonemes:
                for phoneme in words_phonemes_seq:
                    if not phoneme.isspace():
                        phoneme_label.append(phoneme)
                        last_phonemes.append(0)
                last_phonemes[-1] = 1

            phoneme_labels.append(phoneme_label)
            last_phonemes_combined.append(last_phonemes)

        # Inserts silent phonemes in the input phoneme sequence
        all_tokens_with_spn = list()
        max_seq_len = -1
        for i in range(len(phoneme_labels)):
            phoneme_label = phoneme_labels[i]
            token_seq = (
                self.input_encoder.encode_sequence_torch(phoneme_label)
                .int()
                .to(self.device)
            )
            last_phonemes = torch.LongTensor(last_phonemes_combined[i]).to(
                self.device
            )

            # Runs the silent phoneme predictor
            spn_preds = (
                self.hparams.modules["spn_predictor"]
                .infer(token_seq.unsqueeze(0), last_phonemes.unsqueeze(0))
                .int()
            )

            spn_to_add = torch.nonzero(spn_preds).reshape(-1).tolist()

            tokens_with_spn = list()

            for token_idx in range(token_seq.shape[0]):
                tokens_with_spn.append(token_seq[token_idx].item())
                if token_idx in spn_to_add:
                    tokens_with_spn.append(self.spn_token_encoded)

            tokens_with_spn = torch.LongTensor(tokens_with_spn).to(self.device)
            all_tokens_with_spn.append(tokens_with_spn)
            if max_seq_len < tokens_with_spn.shape[-1]:
                max_seq_len = tokens_with_spn.shape[-1]

        # "tokens_with_spn_tensor" holds the input phoneme sequence with silent phonemes
        tokens_with_spn_tensor = torch.LongTensor(
            tokens.shape[0], max_seq_len
        ).to(self.device)
        tokens_with_spn_tensor.zero_()

        for seq_idx, seq in enumerate(all_tokens_with_spn):
            tokens_with_spn_tensor[seq_idx, : len(seq)] = seq

        (
            _,
            postnet_mel_out_spn_pred,
            _,
            _,
            _,
            _,
            _,
            predict_mel_lens_spn_pred,
        ) = self.hparams.model(tokens_with_spn_tensor)

        return (
            postnet_mel_out,
            predict_mel_lens,
            postnet_mel_out_spn_pred,
            predict_mel_lens_spn_pred,
        )

    def run_vocoder(self, inference_mel, mel_lens, sample_type=""):
        """Uses a pretrained vocoder to generate audio from predicted mel
        spectogram. By default, uses speechbrain hifigan.

        Arguments
        ---------
        inference_mel: torch.Tensor
            predicted mel from fastspeech2 inference
        mel_lens: torch.Tensor
            predicted mel lengths from fastspeech2 inference
            used to mask the noise from padding
        sample_type: str
            used for logging the type of the inference sample being generated

        Returns
        -------
        None
        """
        if self.last_batch is None:
            return
        *_, wavs = self.last_batch

        inference_mel = inference_mel[: self.hparams.progress_batch_sample_size]
        mel_lens = mel_lens[0 : self.hparams.progress_batch_sample_size]
        assert (
            self.hparams.vocoder == "hifi-gan"
            and self.hparams.pretrained_vocoder is True
        ), "Specified vocoder not supported yet"
        logger.info(
            f"Generating audio with pretrained {self.hparams.vocoder_source} vocoder"
        )
        hifi_gan = HIFIGAN.from_hparams(
            source=self.hparams.vocoder_source,
            savedir=self.hparams.vocoder_download_path,
        )
        waveforms = hifi_gan.decode_batch(
            inference_mel.transpose(2, 1), mel_lens, self.hparams.hop_length
        )
        for idx, wav in enumerate(waveforms):
            path = os.path.join(
                self.hparams.progress_sample_path,
                str(self.last_epoch),
                f"pred_{sample_type}_{Path(wavs[idx]).stem}.wav",
            )
            torchaudio.save(path, wav, self.hparams.sample_rate)

    def batch_to_device(self, batch, return_metadata=False):
        """Transfers the batch to the target device
        Arguments
        ---------
        batch: tuple
            the batch to use
        return_metadata: bool
            indicates whether the metadata should be returned
        Returns
        -------
        batch: tuple
            the batch on the correct device
        """

        (
            text_padded,
            durations,
            input_lengths,
            mel_padded,
            pitch_padded,
            energy_padded,
            output_lengths,
            len_x,
            labels,
            wavs,
            no_spn_seq_padded,
            spn_labels_padded,
            last_phonemes_padded,
        ) = batch

        durations = durations.to(self.device, non_blocking=True).long()
        phonemes = text_padded.to(self.device, non_blocking=True).long()
        input_lengths = input_lengths.to(self.device, non_blocking=True).long()
        spectogram = mel_padded.to(self.device, non_blocking=True).float()
        pitch = pitch_padded.to(self.device, non_blocking=True).float()
        energy = energy_padded.to(self.device, non_blocking=True).float()
        mel_lengths = output_lengths.to(self.device, non_blocking=True).long()
        no_spn_seqs = no_spn_seq_padded.to(
            self.device, non_blocking=True
        ).long()
        spn_labels = spn_labels_padded.to(self.device, non_blocking=True).long()
        last_phonemes = last_phonemes_padded.to(
            self.device, non_blocking=True
        ).long()
        x = (phonemes, durations, pitch, energy, no_spn_seqs, last_phonemes)
        y = (
            spectogram,
            durations,
            pitch,
            energy,
            mel_lengths,
            input_lengths,
            spn_labels,
        )
        metadata = (labels, wavs)
        if return_metadata:
            return x, y, metadata
        return x, y


def dataio_prepare(hparams):
    # Load lexicon
    lexicon = hparams["lexicon"]
    input_encoder = hparams.get("input_encoder")

    # add a dummy symbol for idx 0 - used for padding.
    lexicon = ["@@"] + lexicon
    input_encoder.update_from_iterable(lexicon, sequence_input=False)
    input_encoder.add_unk()

    # load audio, text and durations on the fly; encode audio and text.
    @sb.utils.data_pipeline.takes(
        "wav",
        "label_phoneme",
        "durations",
        "pitch",
        "start",
        "end",
        "spn_labels",
        "last_phoneme_flags",
    )
    @sb.utils.data_pipeline.provides("mel_text_pair")
    def audio_pipeline(
        wav,
        label_phoneme,
        dur,
        pitch,
        start,
        end,
        spn_labels,
        last_phoneme_flags,
    ):
        durs = np.load(dur)
        durs_seq = torch.from_numpy(durs).int()
        label_phoneme = label_phoneme.strip()
        label_phoneme = label_phoneme.split()
        text_seq = input_encoder.encode_sequence_torch(label_phoneme).int()

        assert len(text_seq) == len(
            durs
        ), f"{len(text_seq)}, {len(durs), len(label_phoneme)}, ({label_phoneme})"  # ensure every token has a duration

        no_spn_label, last_phonemes = list(), list()
        for i in range(len(label_phoneme)):
            if label_phoneme[i] != "spn":
                no_spn_label.append(label_phoneme[i])
                last_phonemes.append(last_phoneme_flags[i])

        no_spn_seq = input_encoder.encode_sequence_torch(no_spn_label).int()

        spn_labels = [
            spn_labels[i]
            for i in range(len(label_phoneme))
            if label_phoneme[i] != "spn"
        ]

        audio, fs = torchaudio.load(wav)

        audio = audio.squeeze()
        audio = audio[int(fs * start) : int(fs * end)]

        mel, energy = hparams["mel_spectogram"](audio=audio)
        mel = mel[:, : sum(durs)]
        energy = energy[: sum(durs)]
        pitch = np.load(pitch)
        pitch = torch.from_numpy(pitch)
        pitch = pitch[: mel.shape[-1]]
        return (
            text_seq,
            durs_seq,
            mel,
            pitch,
            energy,
            len(text_seq),
            last_phonemes,
            no_spn_seq,
            spn_labels,
        )

    # define splits and load it as sb dataset
    datasets = {}

    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_json"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["mel_text_pair", "wav", "label", "durations", "pitch"],
        )
    return datasets, input_encoder


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    sb.utils.distributed.ddp_init_group(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from ljspeech_prepare import prepare_ljspeech

    sb.utils.distributed.run_on_main(
        prepare_ljspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "split_ratio": hparams["split_ratio"],
            "model_name": hparams["model"].__class__.__name__,
            "seed": hparams["seed"],
            "pitch_n_fft": hparams["n_fft"],
            "pitch_hop_length": hparams["hop_length"],
            "pitch_min_f0": hparams["min_f0"],
            "pitch_max_f0": hparams["max_f0"],
            "skip_prep": hparams["skip_prep"],
            "use_custom_cleaner": True,
        },
    )

    datasets, input_encoder = dataio_prepare(hparams)

    # Brain class initialization
    fastspeech2_brain = FastSpeech2Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    fastspeech2_brain.input_encoder = input_encoder
    # Training
    fastspeech2_brain.fit(
        fastspeech2_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )


if __name__ == "__main__":
    main()
