#!/usr/bin/python3
"""Recipe for training sound class embeddings (e.g, xvectors) using the UrbanSound8k.
We employ an encoder followed by a sound classifier.

To run this recipe, use the following command:
> python train_class_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hparams/train_x_vectors.yaml (for standard xvectors)
    hparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Authors
    * David Whipps 2021
    * Ala Eddine Limame 2021

Based on VoxCeleb By:
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from esc50_prepare import prepare_esc50
from sklearn.metrics import confusion_matrix
import numpy as np
from confusion_matrix_fig import create_cm_fig
import matplotlib.pyplot as plt
from os import makedirs

import torch.nn.functional as F

import librosa
from librosa.core import stft, istft
import scipy.io.wavfile as wavf
import soundfile as sf
from speechbrain.processing.NMF import spectral_phase


eps = 1e-10


class InterpreterESC50Brain(sb.core.Brain):
    """Class for sound class embedding training" """

    @torch.no_grad()
    def interpret_batch(self, batch):
        """ Interprets first element of `batch`.
        TODO: add overlap test on samples from batch """
        batch = batch.to(self.device)
        wavs, _ = batch.sig
        wavs = wavs[0].unsqueeze(0)

        # compute stft and logmel, and phase
        X_stft = self.modules.compute_stft(wavs)
        X_stft_phase = spectral_phase(X_stft)
        X_stft_power = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        X_logmel = self.modules.compute_fbank(X_stft_power)

        embeddings, f_I = self.hparams.embedding_model(X_logmel)

        psi_out = self.modules.psi(f_I)  # generate nmf activations

        # cut the length of psi
        psi_out = psi_out[:, :, : X_stft_power.shape[1]]

        # cem: do we need this here?
        reconstructed = self.hparams.nmf(
            psi_out
        )  #  generate log-mag spectrogram

        # get the classifier output
        predictions = self.hparams.classifier(embeddings).squeeze(1)
        pred_cl = torch.argmax(predictions, dim=1)[0].item()
        # print(pred_cl)

        spec_shape = reconstructed.shape
        nmf_dictionary = self.hparams.nmf.return_W(dtype="torch")

        # computes time activations per component
        # FROM NOW ON WE FOLLOW THE PAPER'S NOTATION
        psi_out = psi_out.squeeze()
        z = self.modules.theta.hard_att(psi_out).squeeze()
        theta_c_w = self.modules.theta.classifier[0].weight[pred_cl]

        # some might be negative, relevance of component
        r_c_x = theta_c_w * z / torch.abs(theta_c_w * z).max()
        # define selected components
        L = torch.arange(r_c_x.shape[0])[r_c_x > 0.2].tolist()

        # get the log power spectra, this is needed as NMF is trained on log-power spectra
        X_stft_power_log = (
            torch.log(X_stft_power + 1).transpose(1, 2).squeeze(0)
        )

        # get the contribution of each component
        # X_ks = torch.zeros(len(L), spec_shape[1], spec_shape[2]).to(self.device)
        # sum_X_k = torch.zeros(spec_shape[1], spec_shape[2]).to(self.device)
        # for (i, k) in enumerate(L):
        #     X_k = nmf_dictionary[:, k].unsqueeze(1) @ psi_out[k, :].unsqueeze(0)
        #     sum_X_k += X_k
        #     X_ks[i] = X_k
        # cem : for the denominator we need to sum over all K, not just the selected ones.
        X_withselected = nmf_dictionary[:, L] @ psi_out[L, :]
        Xhat = nmf_dictionary @ psi_out

        # need the eps for the denominator
        eps = 1e-10
        # X_int = (X_ks / (sum_X_k.unsqueeze(0)+eps)).sum(0) * X_stft_power_log
        X_int = (X_withselected / (Xhat + eps)) * X_stft_power_log

        # get back to the standard stft
        X_int = torch.exp(X_int) - 1

        # add the phase of the original audio
        X_int_wphase = (
            (X_int.permute(1, 0).unsqueeze(0) * torch.exp(1j * X_stft_phase))
            .cpu()
            .numpy()
            .squeeze()
        )

        # invert back to time domain (cem: need to check if this is the exact same SB istft)
        # I am being lazy by using numpy here for istft as it supports complex numbers directly
        x_int = istft(X_int_wphase.transpose(), win_length=1024, hop_length=512)
        # x_int = self.modules.compute_istft(X_int_wphase)

        # save reconstructed and original spectrograms
        makedirs(
            os.path.join(
                self.hparams.output_folder, f"audios_from_interpretation",
            ),
            exist_ok=True,
        )

        epoch = self.hparams.epoch_counter.current
        sf.write(
            os.path.join(
                self.hparams.output_folder,
                f"audios_from_interpretation",
                f"original_{epoch}.wav",
            ),
            wavs[0].cpu().numpy(),
            self.hparams.sample_rate,
        )

        sf.write(
            os.path.join(
                self.hparams.output_folder,
                f"audios_from_interpretation",
                f"interpretation_{epoch}.wav",
            ),
            x_int,
            self.hparams.sample_rate,
        )

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + sound classifier.
        Data augmentation and environmental corruption are applied to the
        input sound.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # no data augmentation here
        if stage == sb.Stage.TRAIN and False:

            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                wavs_aug = augment(wavs, lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)

        elif (
            stage == sb.Stage.VALID
            and (
                self.hparams.epoch_counter.current
                % self.hparams.interpret_period
            )
            == 0
        ):
            # save some samples
            self.interpret_batch(batch)

        X_stft = self.modules.compute_stft(wavs)
        X_stft_power = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        X_logmel = self.modules.compute_fbank(X_stft_power)

        # Embeddings + sound classifier
        embeddings, f_I = self.hparams.embedding_model(X_logmel)

        psi_out = self.modules.psi(f_I)  # generate nmf activations

        # cut the length of psi
        psi_out = psi_out[:, :, : X_stft_power.shape[1]]

        reconstructed = self.hparams.nmf(
            psi_out
        )  #  generate log-mag spectrogram

        predictions = self.hparams.classifier(embeddings).squeeze(1)

        theta_out = self.modules.theta(
            psi_out
        )  # generate classifications from time activations

        return (reconstructed, psi_out), (predictions, theta_out)

    def compute_objectives(self, reconstructions, batch, stage):
        """Computes the loss using class-id as label."""
        (
            (reconstructions, time_activations),
            (classification_out, theta_out,),
        ) = reconstructions

        uttid = batch.id
        classid, _ = batch.class_string_encoded

        batch = batch.to(self.device)
        wavs, _ = batch.sig

        X_stft = self.modules.compute_stft(wavs).to(self.device)
        X_stft_power = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        X_stft_logpower = torch.log(X_stft_power + 1).transpose(1, 2)

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and False:
            classid = torch.cat([classid] * self.n_augment, dim=0)

        loss_nmf = ((reconstructions - X_stft_logpower) ** 2).mean()
        # loss_nmf = loss_nmf / reconstructions.shape[0]  # avg on batches
        loss_nmf = self.hparams.alpha * loss_nmf
        # loss_nmf += self.hparams.beta * torch.linalg.norm(time_activations)

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        self.last_batch = batch
        self.batch_to_plot = (reconstructions.clone(), X_stft_logpower.clone())

        theta_out = -torch.log(theta_out)
        loss_fdi = (F.softmax(classification_out, dim=0) * theta_out).mean()

        return loss_nmf + loss_fdi

    @staticmethod
    def select_component(idx, inp_lg_spec, H, W):
        # Selects the contribution of component j (j=idx) from the given input log magnitude spectrogram
        # Assume integer/numpy arrays for all input arguments, W of shape N_FREQ x N_COMP and H of N_COMP x N_TIME
        # Do W.abs() to force positive values
        W_mat = torch.abs(W)
        # ratio = np.outer(W_mat[:, idx], H[idx]) / (0.000001 + np.dot(W_mat, H))
        ratio = (W_mat[:, idx].unsqueeze(1) * H[idx].unsqueeze(0)) / (
            eps + torch.matmul(W_mat, H)
        )

        # comp = np.exp(inp_lg_spec * ratio) - 1
        comp = inp_lg_spec * ratio
        # comp = torch.Tensor(comp)
        # ratio = torch.Tensor(ratio)

        return comp, ratio

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Plots in subplots the values of `self.batch_to_plot` and saves the
        plot to the experiment folder. `self.hparams.output_folder`"""

        pred, target = self.batch_to_plot
        pred = pred.detach().cpu().numpy()[:2, ...]
        target = target.detach().cpu().numpy()[:2, ...]

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].set_ylabel("Predicted")
        ax[0, 0].imshow(pred[0, ...])
        ax[0, 1].imshow(pred[1, ...])
        ax[1, 0].set_ylabel("Target")
        ax[1, 0].imshow(target[0, ...])
        ax[1, 1].imshow(target[1, ...])
        # ax.
        makedirs(
            os.path.join(self.hparams.output_folder, f"reconstructions"),
            exist_ok=True,
        )
        plt.savefig(
            os.path.join(
                self.hparams.output_folder,
                f"reconstructions",
                f"{str(stage)}_{epoch}.png",
            )
        )

        return super().on_stage_end(stage, stage_loss, epoch)


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_audio_folder = hparams["audio_data_folder"]
    config_sample_rate = hparams["sample_rate"]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        wave_file = data_audio_folder + "/{:}".format(wav)

        sig, read_sr = torchaudio.load(wave_file)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        # the librosa version
        fs, inp_audio = wavf.read(wave_file)
        inp_audio = inp_audio.astype(np.float32)
        inp_audio = inp_audio / inp_audio.max()
        # if self.noise:
        #     energy_signal = (inp_audio ** 2).mean()
        #     noise = np.random.normal(0, 0.05, inp_audio.shape[0])
        #     energy_noise = (noise ** 2).mean()
        #     const = np.sqrt(energy_signal / energy_noise)
        #     noise = const * noise
        #     inp_audio = inp_audio + noise

        return torch.from_numpy(inp_audio)

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_string")
    @sb.utils.data_pipeline.provides("class_string", "class_string_encoded")
    def label_pipeline(class_string):
        yield class_string
        class_string_encoded = label_encoder.encode_label_torch(class_string)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "class_string_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="class_string",
    )

    return datasets, label_encoder


if __name__ == "__main__":

    # # This flag enables the inbuilt cudnn auto-tuner
    # torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # classifier is fixed here
    hparams["embedding_model"].eval()
    hparams["classifier"].eval()

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

    run_on_main(
        prepare_esc50,
        kwargs={
            "data_folder": hparams["data_folder"],
            "audio_data_folder": hparams["audio_data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "train_fold_nums": hparams["train_fold_nums"],
            "valid_fold_nums": hparams["valid_fold_nums"],
            "test_fold_nums": hparams["test_fold_nums"],
            "skip_manifest_creation": hparams["skip_manifest_creation"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    Interpreter_brain = InterpreterESC50Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if "pretrained_esc50" in hparams:
        run_on_main(hparams["pretrained_esc50"].collect_files)
        hparams["pretrained_esc50"].load_collected()

    hparams["embedding_model"].to(hparams["device"])
    hparams["classifier"].to(hparams["device"])
    hparams["embedding_model"].eval()
    hparams["nmf"].to(hparams["device"])

    if not hparams["test_only"]:
        Interpreter_brain.fit(
            epoch_counter=Interpreter_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )

    # # Load the best checkpoint for evaluation
    # test_stats = ESC50_brain.evaluate(
    #     test_set=datasets["test"],
    #     min_key="error",
    #     progressbar=True,
    #     test_loader_kwargs=hparams["dataloader_options"],
    # )
