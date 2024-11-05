#!/usr/bin/python3

"""Recipe to interpret an audio classifier by-design via activation maps thresholding (AMT).

To run this recipe, use the following command:
> python intepret_amt.py hparams/<amt-config>.yaml --data_folder /yourpath/ESC-50-master

Authors
    * Cem Subakan 2022, 2023
    * Francesco Paissan 2022, 2023
    * Luca Della Libera 2024
"""

import os
import sys

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchvision
from hyperpyyaml import load_hyperpyyaml
from torch.nn import functional as F

import speechbrain as sb
from speechbrain.processing.NMF import spectral_phase
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.metric_stats import MetricStats


class InterpreterESC50Brain(sb.core.Brain):
    """Class for interpreter training."""

    def invert_stft_with_phase(self, X_int, X_stft_phase):
        """Inverts STFT spectra given phase."""
        X_stft_phase_sb = torch.cat(
            (
                torch.cos(X_stft_phase).unsqueeze(-1),
                torch.sin(X_stft_phase).unsqueeze(-1),
            ),
            dim=-1,
        )

        X_stft_phase_sb = X_stft_phase_sb[:, : X_int.shape[1], :, :]
        if X_int.ndim == 3:
            X_int = X_int.unsqueeze(-1)
        X_wpsb = X_int * X_stft_phase_sb
        x_int_sb = self.modules.compute_istft(X_wpsb)

        return x_int_sb

    def preprocess(self, wavs):
        """Pre-process wavs."""
        X_stft = self.modules.compute_stft(wavs)
        X_stft_power = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        X_stft_logpower = torch.log1p(X_stft_power)

        return X_stft_logpower, X_stft, X_stft_power

    @torch.no_grad()
    def classifier_forward(self, X_stft_logpower):
        """The forward pass for the classifier."""
        config = self.hparams.embedding_model.config
        # Resize to match expected resolution
        net_input = torchvision.transforms.functional.resize(
            X_stft_logpower, (config.image_size, config.image_size)
        )
        # Expand to have 3 channels
        net_input = net_input[:, None, ...].expand(-1, 3, -1, -1)
        if config.model_type == "focalnet":
            hcat = self.hparams.embedding_model(net_input).feature_maps[-1]
            embeddings = hcat.mean(dim=(-1, -2))
            modulators = [
                encoder_stage.layers[-1].modulation.modulator
                for encoder_stage in self.hparams.embedding_model.focalnet.encoder.stages
            ]
            modulators = [x.norm(dim=-3, p=2, keepdim=True) for x in modulators]
            # Upsample spatial dimensions
            modulators = [
                torchvision.transforms.functional.resize(
                    x, X_stft_logpower.shape[-2:]
                )
                for x in modulators
            ]
            xhat = modulators[-1]
        elif config.model_type == "vit":
            model_output = self.hparams.embedding_model(
                net_input, output_attentions=True
            )

            hcat = model_output.last_hidden_state.movedim(-1, -2)
            embeddings = hcat.mean(dim=-1)

            # Take the representations from CLS token
            num_heads = self.hparams.embedding_model.config.num_attention_heads
            attentions = [x[:, :, 0, 1:] for x in model_output.attentions]

            # Reshape the attention scores to resemble mini patches
            num_patches = (
                self.hparams.embedding_model.config.image_size
                // self.hparams.embedding_model.config.patch_size
            )

            attentions = [
                x.reshape(-1, num_heads, num_patches, num_patches)
                for x in attentions
            ]
            attentions = [x.mean(dim=-3, keepdim=True) for x in attentions]
            # Upsample spatial dimensions
            attentions = [
                torchvision.transforms.functional.resize(
                    x, X_stft_logpower.shape[-2:]
                )
                for x in attentions
            ]
            xhat = attentions[-1]
        else:
            raise NotImplementedError

        predictions = self.hparams.classifier(embeddings).squeeze(1)
        class_pred = predictions.argmax(1)

        threshold = xhat.reshape(len(xhat), -1).quantile(
            self.hparams.quantile, dim=-1
        )[:, None, None, None]
        xhat[xhat < threshold] = -float("inf")
        xhat[xhat >= threshold] = float("inf")

        return xhat, predictions, class_pred

    def interpret_computation_steps(self, wavs, print_probability=False):
        """Computation steps to get the interpretation spectrogram."""
        X_stft_logpower, X_stft, X_stft_power = self.preprocess(wavs)
        X_stft_phase = spectral_phase(X_stft)

        xhat, predictions, class_pred = self.classifier_forward(X_stft_logpower)
        if print_probability:
            predictions = F.softmax(predictions, dim=1)
            class_prob = predictions[0, class_pred].item()
            print(f"classifier_prob: {class_prob}")

        xhat = xhat.squeeze(1)

        Tmax = xhat.shape[1]
        if self.hparams.use_mask_output:
            xhat = F.sigmoid(xhat)
            X_int = xhat * X_stft_logpower[:, :Tmax, :]
        else:
            xhat = F.softplus(xhat)
            th = xhat.max() * self.hparams.mask_th
            X_int = (xhat > th) * X_stft_logpower[:, :Tmax, :]

        return X_int, X_stft_phase, class_pred, X_stft_logpower, xhat

    def interpret_sample(self, wavs, batch=None):
        """Get the interpratation for a given wav file."""

        # Get the interpretation spectrogram, phase, and the predicted class
        X_int, X_stft_phase, pred_cl, _, _ = self.interpret_computation_steps(
            wavs
        )
        X_stft_phase = X_stft_phase[:, : X_int.shape[1], :]
        if not (batch is None):
            x_int_sb = self.invert_stft_with_phase(X_int, X_stft_phase)

            # Save reconstructed and original spectrograms
            os.makedirs(
                os.path.join(
                    self.hparams.output_folder,
                    "audios_from_interpretation",
                ),
                exist_ok=True,
            )

            current_class_ind = batch.class_string_encoded.data[0].item()
            current_class_name = self.hparams.label_encoder.ind2lab[
                current_class_ind
            ]
            predicted_class_name = self.hparams.label_encoder.ind2lab[
                pred_cl.item()
            ]
            torchaudio.save(
                os.path.join(
                    self.hparams.output_folder,
                    "audios_from_interpretation",
                    f"original_tc_{current_class_name}_pc_{predicted_class_name}.wav",
                ),
                wavs[0].unsqueeze(0).cpu(),
                self.hparams.sample_rate,
            )

            torchaudio.save(
                os.path.join(
                    self.hparams.output_folder,
                    "audios_from_interpretation",
                    f"interpretation_tc_{current_class_name}_pc_{predicted_class_name}.wav",
                ),
                x_int_sb.cpu(),
                self.hparams.sample_rate,
            )

        return X_int

    def overlap_test(self, batch):
        """Interpration test with overlapped audio."""
        wavs, _ = batch.sig
        wavs = wavs.to(self.device)

        if wavs.shape[0] <= 1:
            return

        s1 = wavs[0]
        s1 = s1 / s1.max()
        s2 = wavs[1]
        s2 = s2 / s2.max()

        # Create the mixture with s2 being the noise (lower gain)
        if (
            hasattr(self.hparams, "concat_sources")
            and self.hparams.concat_sources
        ):
            length = min(len(s1), len(s2))
            mid = length // 2
            s1[mid:] = 0.0
            s2[:mid] = 0.0
            mix = (s1 + s2).unsqueeze(0)
        else:
            mix = (s1 * 0.8 + (s2 * 0.2)).unsqueeze(0)
        mix = mix / mix.max()

        # Get the interpretation spectrogram, phase, and the predicted class
        (
            X_int,
            X_stft_phase,
            pred_cl,
            X_mix,
            mask,
        ) = self.interpret_computation_steps(mix)
        X_int = X_int[0, ...]
        X_stft_phase = X_stft_phase[0, : X_int.shape[0], ...].unsqueeze(0)
        pred_cl = pred_cl[0, ...]
        mask = mask[0, ...]

        temp = torch.expm1(X_int).unsqueeze(0).unsqueeze(-1)
        x_int_sb = self.invert_stft_with_phase(temp, X_stft_phase)

        # Save reconstructed and original spectrograms
        current_class_ind = batch.class_string_encoded.data[0].item()
        current_class_name = self.hparams.label_encoder.ind2lab[
            current_class_ind
        ]
        predicted_class_name = self.hparams.label_encoder.ind2lab[
            pred_cl.item()
        ]

        noise_class_ind = batch.class_string_encoded.data[1].item()
        noise_class_name = self.hparams.label_encoder.ind2lab[noise_class_ind]

        out_folder = os.path.join(
            self.hparams.output_folder,
            "overlap_test",
            f"tc_{current_class_name}_nc_{noise_class_name}_pc_{predicted_class_name}",
        )
        os.makedirs(
            out_folder,
            exist_ok=True,
        )

        torchaudio.save(
            os.path.join(out_folder, "mixture.wav"),
            mix.data.cpu(),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "source.wav"),
            s1.unsqueeze(0).data.cpu(),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "noise.wav"),
            s2.unsqueeze(0).data.cpu(),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "interpretation.wav"),
            x_int_sb.data.cpu(),
            self.hparams.sample_rate,
        )

        plt.figure(figsize=(15, 5), dpi=100)

        plt.subplot(161)
        (
            _,
            _,
            _,
            X_s1,
            _,
        ) = self.interpret_computation_steps(s1.unsqueeze(0))
        X_target = X_s1[0].permute(1, 0)[:, : X_int.shape[1]].cpu()
        plt.imshow(X_target, origin="lower")
        current_class_ind = batch.class_string_encoded.data[0].item()
        current_class_name = self.hparams.label_encoder.ind2lab[
            current_class_ind
        ]
        plt.title(current_class_name)
        plt.colorbar(fraction=0.05)

        plt.subplot(162)
        (
            _,
            _,
            _,
            X_s2,
            _,
        ) = self.interpret_computation_steps(s2.unsqueeze(0))
        X_target = X_s2[0].permute(1, 0)[:, : X_int.shape[1]].cpu()
        plt.imshow(X_target, origin="lower")
        current_class_ind = batch.class_string_encoded.data[1].item()
        current_class_name = self.hparams.label_encoder.ind2lab[
            current_class_ind
        ]
        plt.title(current_class_name)
        plt.colorbar(fraction=0.05)

        plt.subplot(163)
        X_target = X_mix[0].permute(1, 0)[:, : X_int.shape[1]].cpu()
        plt.imshow(X_target, origin="lower")
        predicted_class_name = self.hparams.label_encoder.ind2lab[
            pred_cl.item()
        ]
        plt.title(predicted_class_name)
        plt.colorbar(fraction=0.05)

        plt.subplot(164)
        plt.imshow(mask.data.cpu().permute(1, 0), origin="lower")
        plt.title("estimated mask")
        plt.colorbar(fraction=0.05)

        plt.subplot(165)
        plt.imshow(X_int.data.cpu().permute(1, 0).data.cpu(), origin="lower")
        plt.title("interpretation")
        plt.colorbar(fraction=0.05)

        plt.subplots_adjust()
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "specs.png"), bbox_inches="tight")
        plt.close()

    def debug_files(self, X_stft, xhat, X_stft_logpower, batch, wavs):
        """The helper function to create debugging images."""
        X_stft_phase = spectral_phase(X_stft)
        temp = xhat[0].transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        Xspec_est = torch.expm1(temp.permute(0, 2, 1, 3))
        xhat_tm = self.invert_stft_with_phase(Xspec_est, X_stft_phase)

        Tmax = Xspec_est.shape[1]
        if self.hparams.use_mask_output:
            X_masked = xhat[0] * X_stft_logpower[0, :Tmax, :]
        else:
            th = xhat[0].max() * 0.15
            X_masked = (xhat[0] > th) * X_stft_logpower[0, :Tmax, :]

        X_est_masked = torch.expm1(X_masked).unsqueeze(0).unsqueeze(-1)
        xhat_tm_masked = self.invert_stft_with_phase(X_est_masked, X_stft_phase)

        plt.figure(figsize=(10, 5), dpi=100)

        plt.subplot(141)
        X_target = X_stft_logpower[0].permute(1, 0)[:, : xhat.shape[1]].cpu()
        plt.imshow(X_target, origin="lower")
        plt.title("input")
        plt.colorbar(fraction=0.05)

        plt.subplot(142)
        input_masked = X_target > (
            X_target.max(keepdim=True, dim=-1)[0].max(keepdim=True, dim=-2)[0]
            * self.hparams.mask_th
        )
        plt.imshow(input_masked, origin="lower")
        plt.title("input masked")
        plt.colorbar(fraction=0.05)

        plt.subplot(143)
        if self.hparams.use_mask_output:
            mask = xhat[0]
        else:
            mask = xhat[0] > th
        X_masked = mask * X_stft_logpower[0, :Tmax, :]
        plt.imshow(X_masked.permute(1, 0).data.cpu(), origin="lower")
        plt.colorbar(fraction=0.05)
        plt.title("interpretation")

        plt.subplot(144)
        plt.imshow(mask.permute(1, 0).data.cpu(), origin="lower")
        plt.colorbar(fraction=0.05)
        plt.title("estimated mask")

        out_folder = os.path.join(
            self.hparams.output_folder,
            "reconstructions",
            f"{batch.id[0]}",
        )
        os.makedirs(
            out_folder,
            exist_ok=True,
        )

        plt.subplots_adjust()
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_folder, "reconstructions.png"),
            bbox_inches="tight",
        )
        plt.close()

        torchaudio.save(
            os.path.join(out_folder, "reconstruction.wav"),
            xhat_tm.data.cpu(),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "reconstruction_masked.wav"),
            xhat_tm_masked.data.cpu(),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "true.wav"),
            wavs[0:1].data.cpu(),
            self.hparams.sample_rate,
        )

    def compute_forward(self, batch, stage):
        """Computation pipeline based on an encoder + sound classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        X_stft_logpower, X_stft, X_stft_power = self.preprocess(wavs)

        # Embeddings + sound classifier
        xhat, predictions, class_pred = self.classifier_forward(X_stft_logpower)

        xhat = xhat.squeeze(1)

        if self.hparams.use_mask_output:
            xhat = F.sigmoid(xhat)
        else:
            xhat = F.softplus(xhat)

        # Save some samples
        if self.hparams.save_interpretations:
            wavs = wavs[0].unsqueeze(0)
            self.interpret_sample(wavs, batch)
            self.overlap_test(batch)
            self.debug_files(X_stft, xhat, X_stft_logpower, batch, wavs)

        return predictions, xhat

    def compute_objectives(self, pred, batch, stage):
        """Helper function to compute the objectives."""
        predictions, xhat = pred

        batch = batch.to(self.device)
        wavs, lens = batch.sig

        uttid = batch.id
        classid, _ = batch.class_string_encoded

        X_stft_logpower, X_stft, X_stft_power = self.preprocess(wavs)

        Tmax = xhat.shape[1]

        _, theta_out, _ = self.classifier_forward(
            xhat * X_stft_logpower[:, :Tmax, :]
        )

        self.acc_metric.append(
            uttid, predict=predictions, target=classid, length=lens
        )

        self.top_3_fidelity.append(
            [batch.id] * theta_out.shape[0], theta_out, predictions
        )

        self.faithfulness.append(batch.id, wavs, predictions)

        return torch.as_tensor([0.0], device=self.device)

    def on_stage_start(self, stage, epoch=None):
        """Steps taken before stage start."""

        @torch.no_grad()
        def accuracy_value(predict, target, length):
            """Computes accuracy."""
            nbr_correct, nbr_total = sb.utils.Accuracy.Accuracy(
                predict.unsqueeze(1), target, length
            )
            acc = torch.tensor([nbr_correct / nbr_total])
            return acc

        @torch.no_grad()
        def compute_fidelity(theta_out, predictions):
            """Computes top-k fidelity of interpreter."""
            predictions = F.softmax(predictions, dim=1)
            theta_out = F.softmax(theta_out, dim=1)

            pred_cl = torch.argmax(predictions, dim=1)
            k_top = torch.argmax(theta_out, dim=1)

            # 1 element for each sample in batch, is 0 if pred_cl is in top k
            temp = (k_top == pred_cl).float()

            return temp

        @torch.no_grad()
        def compute_faithfulness(wavs, predictions):
            """Computes the faithfulness metric."""
            X2 = self.interpret_computation_steps(wavs)[0]

            _, predictions_masked, _ = self.classifier_forward(X2)

            predictions = F.softmax(predictions, dim=1)
            predictions_masked = F.softmax(predictions_masked, dim=1)

            # Get the prediction indices
            pred_cl = predictions.argmax(dim=1, keepdim=True)

            # Get the corresponding output probabilities
            predictions_selected = torch.gather(
                predictions, dim=1, index=pred_cl
            )
            predictions_masked_selected = torch.gather(
                predictions_masked, dim=1, index=pred_cl
            )

            faithfulness = (
                predictions_selected - predictions_masked_selected
            ).squeeze(1)

            return faithfulness

        self.top_3_fidelity = MetricStats(metric=compute_fidelity)
        self.faithfulness = MetricStats(metric=compute_faithfulness)
        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )
        return super().on_stage_start(stage, epoch)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Plots in subplots the values of `self.batch_to_plot` and saves the
        plot to the experiment folder `self.hparams.output_folder`."""
        current_fid = self.top_3_fidelity.summarize("average")
        test_stats = {
            "acc": self.acc_metric.summarize("average"),
            "input_fidelity": current_fid,
            "faithfulness_median": torch.Tensor(
                self.faithfulness.scores
            ).median(),
            "faithfulness_mean": torch.Tensor(self.faithfulness.scores).mean(),
        }

        # The train_logger writes a summary to stdout and to the log file
        self.hparams.train_logger.log_stats(
            stats_meta={"epoch": epoch}, test_stats=test_stats
        )


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""
    data_audio_folder = hparams["audio_data_folder"]
    config_sample_rate = hparams["sample_rate"]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
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

        sig = sig.float()
        sig = sig / sig.max()
        return sig

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_string")
    @sb.utils.data_pipeline.provides("class_string", "class_string_encoded")
    def label_pipeline(class_string):
        """The label pipeline."""
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
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="class_string",
    )

    return datasets, label_encoder


if __name__ == "__main__":
    # This flag enables the built-in cuDNN auto-tuner
    # torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
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

    from esc50_prepare import prepare_esc50

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

    if "pretrained_esc50" in hparams and hparams["use_pretrained"]:
        print("Loading model...")
        run_on_main(hparams["pretrained_esc50"].collect_files)
        hparams["pretrained_esc50"].load_collected()

    hparams["embedding_model"].to(run_opts["device"])
    hparams["classifier"].to(run_opts["device"])

    test_stats = Interpreter_brain.evaluate(
        test_set=datasets["test"],
        min_key="loss",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
    )
