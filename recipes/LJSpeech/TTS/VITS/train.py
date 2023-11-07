"""
 Recipe for training the VITS Text-To-Speech model, an end-to-end
 neural text-to-speech (TTS) system introduced in 'Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech' paper
 (https://arxiv.org/abs/2106.06103)
 To run this recipe, do the following:
 # python train.py hparams/train.yaml
 Authors
 * Sathvik Udupa 2023

"""

import os
import sys
import torch
import logging 
import torchaudio
import numpy as np
import speechbrain as sb
from speechbrain.pretrained import HIFIGAN
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import scalarize

sys.path.append("/home/wtc7/Sathvik/speechbrain_imp//speechbrain/lobes/models/VITS.py")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


class VITSBrain(sb.Brain):
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics"""
        self.hparams.progress_sample_logger.reset()
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
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
        inputs1, inputs2, _ = self.batch_to_device(batch)



        # Forward pass for the VITS module
        pred = self.modules.vits_mel_predict(inputs1)
        z_slices, y_slices = self.hparams.vits_random_slicer(
            z=pred[0], 
            z_lengths=inputs1[-1],
            y=inputs2[0],
            hop_length=self.hparams.hop_length,
        )
        y_g_hat = self.modules.generator(z_slices)
        scores_fake, feats_fake = self.modules.discriminator(y_g_hat.detach())
        scores_real, feats_real = self.modules.discriminator(y_slices)

        hifigan_outputs = (y_g_hat, y_slices, scores_fake, feats_fake, scores_real, feats_real)
        
        return (hifigan_outputs, pred)

    def fit_batch(self, batch):
        """Fits a single batch
        Arguments
        ---------
        batch: tuple
            a training batch
        Returns
        -------
        loss: torch.Tensor
            detached loss
        """
        outputs = self.compute_forward(batch, sb.core.Stage.TRAIN)
        losses = self.compute_objectives(outputs, batch, sb.core.Stage.TRAIN)
        
        return outputs

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """
        if self.opt_class is not None:
            (
                opt_g_class,
                opt_d_class,
                sch_g_class,
                sch_d_class,
            ) = self.opt_class
            print(opt_g_class)
            self.optimizer_g = opt_g_class(self.modules.generator.parameters())
            self.optimizer_d = opt_d_class(
                self.modules.discriminator.parameters()
            )
            self.scheduler_g = sch_g_class(self.optimizer_g)
            self.scheduler_d = sch_d_class(self.optimizer_d)

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "optimizer_g", self.optimizer_g
                )
                self.checkpointer.add_recoverable(
                    "optimizer_d", self.optimizer_d
                )
                self.checkpointer.add_recoverable(
                    "scheduler_g", self.scheduler_d
                )
                self.checkpointer.add_recoverable(
                    "scheduler_d", self.scheduler_d
                )

    def zero_grad(self, set_to_none=False):
        if self.opt_class is not None:
            self.optimizer_g.zero_grad(set_to_none)
            self.optimizer_d.zero_grad(set_to_none)

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
        x1, x2, y, metadata = self.batch_to_device(batch, return_metadata=True)
        # self.last_batch = [x[0], y[-2], y[-3], predictions[0], *metadata]
        # self._remember_sample([x[0], *y, *metadata], predictions)
        
        loss = self.hparams.criterion(
            predictions, self.hparams
        )
        self.last_loss_stats[stage] = scalarize(loss)
        return loss


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
                meta=self.last_loss_stats[stage], min_keys=["total_loss"],
            )
        # We also write statistics about test data spectogramto stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )

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
            input_lengths,
            mel_padded,
            mel_lengths,
            wavs_padded,
            wav_lengths,
            labels,
            wavs
        ) = batch

        text_padded = text_padded.to(self.device, non_blocking=True).long()
        input_lengths = input_lengths.to(self.device, non_blocking=True).long()
        mel = mel_padded.to(self.device, non_blocking=True).float()
        mel_lengths = mel_lengths.to(self.device, non_blocking=True).long()
        # wavs_padded = wavs_padded.to(self.device, non_blocking=True).float()
        wavs_padded = wavs_padded.float()
        # wav_lengths = wav_lengths.to(self.device, non_blocking=True).long()
        
        x1 = (
            text_padded, 
            input_lengths, 
            mel,
            mel_lengths, 
        )
        x2 = (
            wavs_padded,
        )
        y = (
            mel,
            mel_lengths,
            input_lengths,
            wavs_padded,
            wav_lengths
        )
        metadata = (labels, wavs)
        if return_metadata:
            return x1, x2, y, metadata
        return x1, x2, y


def dataio_prepare(hparams):
    # Load lexicon
    with open(hparams["lexicon"], 'r') as f:
        lexicon = list(f.read())
    input_encoder = hparams.get("input_encoder")

    # add a dummy symbol for idx 0 - used for padding.
    lexicon = ["@@"] + lexicon
    input_encoder.update_from_iterable(lexicon, sequence_input=False)
    input_encoder.add_unk()

    # load audio, text on the fly; encode audio and text.
    @sb.utils.data_pipeline.takes(
        "wav",
        "label",
    )
    @sb.utils.data_pipeline.provides("mel_text_pair")
    def audio_pipeline(
        wav,
        label,
    ):  
        label = list(label.strip())
        text_seq = input_encoder.encode_sequence_torch(label).int()
        audio, fs = torchaudio.load(wav)
        audio = audio.squeeze()

        mel, energy = hparams["mel_spectogram"](audio=audio)
        return (
            text_seq,
            audio,
            mel,
            label,
            wav
        )

    # define splits and load it as sb dataset
    datasets = {}

    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_json"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["mel_text_pair", "wav", "label",],
        )
    return datasets, input_encoder


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    sb.utils.distributed.ddp_init_group(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    sys.path.append("../")
    from ljspeech_prepare import prepare_ljspeech

    sb.utils.distributed.run_on_main(
        prepare_ljspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "split_ratio": hparams["split_ratio"],
            "model_name": hparams["vits_mel_predict"].__class__.__name__,
            "seed": hparams["seed"],
            "skip_prep": hparams["skip_prep"],
            "use_custom_cleaner": True,
        },
    )

    datasets, input_encoder = dataio_prepare(hparams)

    # Brain class initialization
    vits_brain = VITSBrain(
        modules=hparams["modules"],
        opt_class=[
            hparams["opt_class_generator"],
            hparams["opt_class_discriminator"],
            hparams["sch_class_generator"],
            hparams["sch_class_discriminator"],
        ],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    vits_brain.input_encoder = input_encoder
    # Training
    vits_brain.fit(
        vits_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )


if __name__ == "__main__":
    main()
