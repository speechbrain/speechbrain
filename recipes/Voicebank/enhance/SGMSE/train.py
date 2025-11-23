import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from pesq import pesq
from sgmse.util.other import pad_spec
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as si_sdr,
    short_time_objective_intelligibility as stoi_tm,
)

import speechbrain as sb
from speechbrain.dataio.dataio import write_audio
from speechbrain.utils.metric_stats import MetricStats


class SGMSEBrain(sb.Brain):
    """
    A Brain class to train an SGMSE-based diffusion model.
    """

    def on_fit_start(self):
        """
        Called once in the beginning of training.
        """
        super().on_fit_start()

        self.writer = SummaryWriter(log_dir=self.hparams.save_dir)

        ema = self.modules["score_model"].ema
        self.checkpointer.add_recoverable(
            name="ema",
            obj=ema,
            custom_save_hook=lambda obj, path: torch.save(
                obj.state_dict(), path
            ),
            custom_load_hook=lambda obj, path, end: obj.load_state_dict(
                torch.load(path)
            ),
        )

        # STFT
        n_fft = self.hparams.n_fft
        hop_length = self.hparams.hop_length
        window_type = self.hparams.window_type
        self.window = self.get_window(window_type, n_fft).to(self.device)
        self.stft_kwargs = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "center": True,
            "return_complex": True,
        }

    def setup_inference(self):
        """
        Called once from inference script.
        Loads the checkpoint, restores the EMA shadow weights,
        swaps them into the DNN, and prepares STFT objects.
        """
        ema_obj = self.modules["score_model"].ema
        if "ema" not in self.checkpointer.recoverables:
            self.checkpointer.add_recoverable(
                name="ema",
                obj=ema_obj,
                custom_save_hook=lambda o, p: torch.save(o.state_dict(), p),
                custom_load_hook=lambda o, p, end: o.load_state_dict(
                    torch.load(p)
                ),
            )

        # Load checkpoint
        self.checkpointer.recover_if_possible()

        # Store EMA
        self.modules["score_model"].store_ema()

        # STFT
        n_fft = self.hparams.n_fft
        hop = self.hparams.hop_length
        win_t = self.hparams.window_type
        self.window = self.get_window(win_t, n_fft).to(self.device)
        self.stft_kwargs = dict(
            n_fft=n_fft, hop_length=hop, center=True, return_complex=True
        )

    def _step(self, x, y, model):
        """
        Perform a single diffusion step for the score-based model.

        This function samples a random time-step for each input in the batch, computes the corresponding
        marginal probability of the clean signal using the SDE, adds noise to generate a noisy version
        of the input (x_t), and obtains the model's prediction from this noisy input.

        Arguments
        ---------
        x: torch.Tensor
            Clean input signal spectrogram, of shape (B, 1, F, T).
        y: torch.Tensor
            Conditioning or auxiliary input spectrogram, of shape (B, 1, F, T).
        model: nn.Module
            Score-based generative model that contains the SDE and the forward method.

        Returns
        -------
        forward_out: torch.Tensor
            Model's prediction computed from the noisy input x_t, of shape (B, 1, F, T).
        x_t: torch.Tensor
            Noisy version of the clean input generated using the SDE, of shape (B, 1, F, T).
        z: torch.Tensor
            Noise tensor sampled from a standard normal distribution, of shape (B, 1, F, T).
        t: torch.Tensor
            Randomly sampled time-step for each sample in the batch, of shape (B,).
        mean: torch.Tensor
            Mean of the marginal probability distribution for the clean input, of shape (B, 1, F, T).
        x: torch.Tensor
            Clean input signal spectrogram, of shape (B, 1, F, T).
        """
        t = (
            torch.rand(x.shape[0], device=x.device)
            * (model.sde.T - model.t_eps)
            + model.t_eps
        )  # (B,)
        mean, std = model.sde.marginal_prob(
            x, y, t
        )  # (B,1,F,T) and (B,) respectively
        z = torch.randn_like(
            x
        )  # (B,1,F,T), i.i.d. normal distributed with var=0.5
        sigma = std[:, None, None, None]  # (B,1,1,1)
        x_t = mean + sigma * z  # (B,1,F,T)
        forward_out = model(x_t, y, t)

        return {
            "forward_out": forward_out,
            "x_t": x_t,
            "z": z,
            "t": t,
            "mean": mean,
            "x": x,
        }

    def compute_forward(self, batch, stage):
        """
        Compute forward pass for a given batch.

        This method obtains waveforms from the batch, applies STFT and any spectral
        transformations, and calls `_step` to perform a single diffusion step.
        During validation or test stages, it may also perform a "full enhancement"
        process on a subset of files.

        Arguments
        ---------
        batch: speechbrain.dataio.batch.PaddedBatch
            A batch of data containing
            clean and noisy signals, among other possible fields.
        stage: sb.Stage
            The current stage (TRAIN, VALID, TEST).

        Returns
        -------
        outs: dict
            A dictionary containing the forward pass outputs (including
            the model prediction and any enhanced waveforms if generated).
        """
        # Model and batch preparation
        model = self.modules["score_model"]
        batch = batch.to(self.device)

        # Extract waveforms
        x_wav = batch.clean_sig.data  # (B,S)
        y_wav = batch.noisy_sig.data  # (B,S)

        # STFT, Spec transformations, adding channel dim
        x = self.spec_fwd(self.stft(x_wav)).unsqueeze(1)  # (B,1,F,T)
        y = self.spec_fwd(self.stft(y_wav)).unsqueeze(1)  # (B,1,F,T)

        outs = self._step(x, y, model)

        # TRAIN: never run enhancement
        if stage == sb.Stage.TRAIN:
            return outs

        # VALID: only enhance up to eval_files_left
        if stage == sb.Stage.VALID:
            if self.eval_files_left <= 0:
                # nothing left to do in VALID
                return outs

            # How many files from current batch shall we process?
            B = y_wav.size(0)
            take = min(B, self.eval_files_left)
            self.eval_files_left -= take

            # Slice to that number
            x_wav = x_wav[:take]  # (num_eval_files,S)
            y_wav = y_wav[:take]  # (num_eval_files,S)
            uttids = batch.id[:take]

        # TEST: enhance everything
        if stage == sb.Stage.TEST:
            uttids = batch.id

        # Save original length in time dimension
        T_orig_wav = y_wav.size(1)

        # Enhancement
        x_hat = model.enhance(
            y,
            sampler_type=self.hparams.sampling["sampler_type"],
            predictor=self.hparams.sampling["predictor"],
            corrector=self.hparams.sampling["corrector"],
            N=self.hparams.sampling["N"],
            corrector_steps=self.hparams.sampling["corrector_steps"],
            snr=self.hparams.sampling["snr"],
        )  # (num_files, 1, F, T)

        # Unsqueeze channel dim
        x_hat = x_hat.squeeze(1)
        x = x.squeeze(1)

        # Reverse spech transformations
        x_hat = self.spec_back(x_hat)  # (num_files, F, T)
        x = self.spec_back(x)  # (num_files, F, T)

        # iSTFT
        x_hat_wav = self.istft(x_hat, T_orig_wav)  # (num_files, S)
        x_wav = self.istft(x, T_orig_wav)  # (num_files, S)

        outs.update(
            {
                "x_hat_wav": x_hat_wav,  # enhanced
                "x_wav": x_wav,  # clean
                "y_wav": y_wav,  # noisy
                "uttids": uttids,  # so compute_objectives can see them
            }
        )

        return outs

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the diffusion loss and optionally processes enhanced waveforms.

        This method takes the outputs of `compute_forward` (which include
        the model's prediction and possibly enhanced waveforms), computes
        the loss for training or collects metrics for validation/testing.

        Arguments
        ---------
        predictions: dict
            Dictionary containing forward pass outputs,
          e.g. from `_step`.
        batch: speechbrain.dataio.batch.PaddedBatch
            The current batch, which
          can be used for retrieving IDs or additional data if needed.
        stage: sb.Stage
            The current stage (TRAIN, VALID, TEST).

        Returns
        -------
        loss: torch.Tensor
            The computed diffusion loss for this batch.
        """
        model = self.modules["score_model"]

        # Extract items from predictions
        forward_out = predictions["forward_out"]  # (B,1,F,T)
        x_t = predictions["x_t"]  # (B,1,F,T)
        z = predictions["z"]  # (B,1,F,T)
        t = predictions["t"]  # (B,)
        mean = predictions["mean"]  # (B,1,F,T)
        x = predictions["x"]  # (B,1,F,T)

        # Pass the necessary inputs to the model loss
        loss = model.compute_loss(
            forward_out, x_t, z, t, mean, x, to_audio_func=self.to_audio
        )
        self.loss_metric.append(batch.id, forward_out, x_t, z, t, mean, x)

        # Only process enhanced wavs in VALID and TEST
        if stage != sb.Stage.TRAIN:
            x_wav = predictions.get("x_wav", None)
            x_hat_wav = predictions.get("x_hat_wav", None)
            y_wav = predictions.get("y_wav", None)
            uttids = predictions.get("uttids", None)

            if x_wav is not None:
                # STOI
                self.stoi_metric.append(batch.id, x_hat_wav, x_wav)

                # SISDR
                self.sisdr_metric.append(batch.id, x_hat_wav, x_wav)

                # PESQ
                x_wav_cpu = x_wav.cpu()
                x_hat_wav_cpu = x_hat_wav.cpu()
                y_wav_cpu = y_wav.cpu()
                self.pesq_metric.append(
                    batch.id, predict=x_hat_wav_cpu, target=x_wav_cpu
                )

                sr = self.hparams.sample_rate
                save_dir = self.hparams.enhanced_dir
                os.makedirs(save_dir, exist_ok=True)

                epoch_tag = (
                    f"ep{self.hparams.epoch_counter.current}"
                    if stage == sb.Stage.VALID
                    else "test"
                )
                for i, uid in enumerate(uttids):
                    clean_path = os.path.join(
                        save_dir, f"{epoch_tag}_{uid}_clean.wav"
                    )
                    enh_path = os.path.join(
                        save_dir, f"{epoch_tag}_{uid}_enhanced.wav"
                    )
                    noisy_path = os.path.join(
                        save_dir, f"{epoch_tag}_{uid}_noisy.wav"
                    )

                    write_audio(clean_path, x_wav_cpu[i], sr)
                    write_audio(enh_path, x_hat_wav_cpu[i], sr)
                    write_audio(noisy_path, y_wav_cpu[i], sr)
        return loss

    def fit_batch(self, batch):
        """
        Overridden method to train on a single batch.

        This performs the typical Brain forward-backward-update
        steps, and can include updates for EMA.

        Arguments
        ---------
        batch: speechbrain.dataio.batch.PaddedBatch
            The batch of data used for training.

        Returns
        -------
        loss: torch.Tensor
            The computed training loss for the batch.
        """
        # Standard "forward" + "objectives"
        loss = super().fit_batch(batch)

        # Update EMA for the diffusion model
        self.modules["score_model"].update_ema()

        return loss

    def enhance(self, y):
        """
        Run enhancement on a noisy signal.

        Arguments
        ---------
        y: torch.Tensor
            Noisy input signal, of shape (1, T).

        Returns
        -------
        x_hat_wav: torch.Tensor
            Enhanced signal, of shape (1, T).
        """
        model = self.modules["score_model"]

        norm = y.abs().max()
        y = y / norm
        T_orig = y.size(1)  # keep for iSTFT
        y = self.spec_fwd(self.stft(y)).unsqueeze(1)  # (B,1,F,T)
        F_orig, T_spec_orig = y.shape[-2:]

        y = pad_spec(
            y, mode="reflection"
        )  # pad for U-Net down-/up-sampling constraints

        # SGMSE
        x_hat = model.enhance(
            y,
            sampler_type=self.hparams.sampling["sampler_type"],
            predictor=self.hparams.sampling["predictor"],
            corrector=self.hparams.sampling["corrector"],
            N=self.hparams.sampling["N"],
            corrector_steps=self.hparams.sampling["corrector_steps"],
            snr=self.hparams.sampling["snr"],
        )  # (B,1,F,T)

        # revert to waveform
        x_hat = x_hat[:, :, :F_orig, :T_spec_orig].squeeze(1)  # drop ch-dim
        x_hat = self.spec_back(x_hat)
        x_hat_wav = self.istft(x_hat, length=T_orig)  # trim padding
        x_hat_wav = x_hat_wav * norm  # restore scale
        return x_hat_wav

    def on_stage_start(self, stage, epoch=None):
        """
        Called at the beginning of each stage (TRAIN, VALID, TEST).

        This method initializes or resets metrics for that stage.
        It can also be used to set flags or other stage-specific fields.

        Arguments
        ---------
        stage: sb.Stage
            The current stage (TRAIN, VALID, TEST).
        epoch: int, optional
            The current epoch number, if applicable.

        Returns
        -------
        None
        """
        self.loss_metric = MetricStats(
            metric=lambda forward_out, x_t, z, t, mean, x: self.modules[
                "score_model"
            ].compute_loss(forward_out, x_t, z, t, mean, x, reduction="none")
        )

        if stage == sb.Stage.TRAIN:
            return  # Nothing else to prepare for TRAIN

        if stage == sb.Stage.VALID:
            self.modules[
                "score_model"
            ].store_ema()  # Only for VALID, because TEST is wrapped in on_evaluate_start()
            self.eval_files_left = self.hparams.modules[
                "score_model"
            ].num_eval_files
            self.save_counter = 0

        # Build MetricStats objects
        self.stoi_metric = MetricStats(
            metric=lambda pred, tgt: stoi_tm(
                pred, tgt, fs=self.hparams.sample_rate, extended=False
            )
        )

        self.sisdr_metric = MetricStats(
            metric=lambda pred, tgt: si_sdr(pred, tgt)
        )

        self.pesq_metric = MetricStats(
            metric=lambda pred_wav, target_wav: pesq(
                fs=self.hparams.sample_rate,
                ref=target_wav.numpy().squeeze(),
                deg=pred_wav.numpy().squeeze(),
                mode="wb",
            ),
            batch_eval=False,
            n_jobs=1,
        )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        Called at the end of each stage (TRAIN, VALID, TEST).

        Summarizes and prints the loss metrics, and for non-training stages,
        prints additional evaluation metrics (e.g., PESQ, STOI).

        Arguments
        ---------
        stage: sb.Stage
            The current stage (TRAIN, VALID, TEST).
        stage_loss: torch.Tensor
            The aggregated loss over the stage.
        epoch: int, optional
            The current epoch number, if applicable.

        Returns
        -------
        None
        """
        # Get a human-readable name for the stage:
        stage_name = stage.name if hasattr(stage, "name") else str(stage)

        # Summarize the loss metric (average loss over the stage)
        avg_loss = self.loss_metric.summarize("average")

        # Print to console
        if epoch is not None:
            print(f"Epoch {epoch} | Avg {stage_name} Loss: {avg_loss:.4f}")
        else:
            print(f"Avg {stage_name} Loss: {avg_loss:.4f}")

        # Log training loss
        self.writer.add_scalar(f"Loss_{stage_name}", avg_loss, epoch)

        if stage == sb.Stage.TRAIN:
            self.writer.flush()  # Manually write to disk to ensure real time updates
            return  # Nothing else to wrap up for TRAIN

        # Summarize metrics
        avg_pesq = self.pesq_metric.summarize("average")
        avg_stoi = self.stoi_metric.summarize("average")
        avg_sisdr = self.sisdr_metric.summarize("average")

        # Print summaries
        print(f"Avg PESQ: {avg_pesq:.4f}")
        print(f"Avg STOI: {avg_stoi:.4f}")
        print(f"Avg SI-SDR: {avg_sisdr:.4f}")

        # Write summaries to log
        self.writer.add_scalar(f"PESQ_{stage_name}", avg_pesq, epoch)
        self.writer.add_scalar(f"STOI_{stage_name}", avg_stoi, epoch)
        self.writer.add_scalar(f"SI-SDR_{stage_name}", avg_sisdr, epoch)

        if stage == sb.Stage.VALID:
            self.modules[
                "score_model"
            ].restore_ema()  # Only for VALID, because TEST is wrapped in on_evaluate_end()
            self.checkpointer.save_and_keep_only(
                meta={f"{stage_name}_loss": avg_loss},
                min_keys=[f"{stage_name}_loss"],
                num_to_keep=self.hparams.num_to_keep,
            )

        # Manually write to disk to ensure real time updates
        self.writer.flush()

    def on_evaluate_start(self, max_key=None, min_key=None):
        """
        Prepares evaluation.

        Arguments
        ---------
        max_key: str, optional
            Key used to track maximum metric value.
        min_key: str, optional
            Key used to track minimum metric value.
        """
        # Swap in the EMA weights for evaluation
        self.modules["score_model"].store_ema()
        super().on_evaluate_start(max_key=max_key, min_key=min_key)

    def on_evaluate_end(self):
        """
        Restore original weights after evaluation.
        """
        # Restore original weights
        self.modules["score_model"].restore_ema()
        super().on_evaluate_end()

    def to_audio(brain, spec, length=None):
        """
        Convert a complex spectrogram into time-domain audio.

        This method applies `spec_back` to invert the spectral transform
        (log or exponent, if used), followed by iSTFT to return time-domain
        waveforms.

        Arguments
        ---------
        spec: torch.Tensor
            Complex spectrogram of shape (B, F, T)
        length: int, optional
            The target number of samples in the output signal.

        Returns
        -------
        audio: torch.Tensor
            Time-domain waveform, shape (B, S)
        """
        return brain.istft(brain.spec_back(spec), length=length)

    def stft(self, sig):
        """
        Compute the short-time Fourier transform (STFT) of the given signal.

        Arguments
        ---------
        sig: torch.Tensor
            Time-domain signal of shape (B, S).

        Returns
        -------
        spec: torch.Tensor
            Complex STFT, shape (B, F, T).
        """
        return torch.stft(
            sig,
            **{**self.stft_kwargs, "window": self.window},
        )

    def istft(self, spec, length=None):
        """
        Compute the inverse short-time Fourier transform (iSTFT).

        This method reverts the STFT computed by `stft`, using the same
        parameters but without the `return_complex` key.

        Arguments
        ---------
        spec: torch.Tensor
            Complex STFT of shape (B, F, T).
        length: int, optional
            The desired number of samples in the output.

        Returns
        -------
        waveform: torch.Tensor
            Time-domain signal of shape (B, S).
        """
        stft_args = dict(self.stft_kwargs)
        stft_args.pop("return_complex", None)
        stft_args["window"] = self.window
        stft_args["length"] = length

        return torch.istft(spec, **stft_args)

    def spec_fwd(self, spec_cplx):
        """
        Forward spectral transform (e.g., log or exponent) on the complex spectrogram.

        Depending on `transform_type`, applies scaling or a log-based transform to
        the magnitude, preserving phase. Also multiplies by a factor if specified.

        Arguments
        ---------
        spec_cplx: torch.Tensor
            Complex spectrogram of shape (B, F, T).

        Returns
        -------
        spec_trans: torch.Tensor
            Transformed complex spectrogram of the same shape.
        """
        transform_type = self.hparams.transform_type
        factor = self.hparams.spec_factor
        e = getattr(self.hparams, "spec_abs_exponent", 1.0)

        if transform_type == "exponent":
            if e != 1.0:
                mag = spec_cplx.abs() ** e
                phase = spec_cplx.angle()
                spec_cplx = mag * torch.exp(1j * phase)
            spec_cplx *= factor

        elif transform_type == "log":
            mag = torch.log1p(spec_cplx.abs())
            phase = spec_cplx.angle()
            spec_cplx = mag * torch.exp(1j * phase)
            spec_cplx *= factor

        elif transform_type == "none":
            pass

        return spec_cplx

    def spec_back(self, spec_cplx):
        """
        Inverse spectral transform to revert log or exponent scaling.

        This method divides by the scale factor and reverts the transform
        (log or exponent) on the magnitude. The phase remains unchanged.

        Arguments
        ---------
        spec_cplx: torch.Tensor
            Complex spectrogram of shape (B, F, T).

        Returns
        -------
        spec_orig: torch.Tensor
            Original-like complex spectrogram of the same shape.
        """
        transform_type = self.hparams.transform_type
        factor = self.hparams.spec_factor
        e = getattr(self.hparams, "spec_abs_exponent", 1.0)

        if transform_type == "exponent":
            spec_cplx = spec_cplx / factor
            if e != 1.0:
                mag = spec_cplx.abs() ** (1.0 / e)
                phase = spec_cplx.angle()
                spec_cplx = mag * torch.exp(1j * phase)

        elif transform_type == "log":
            spec_cplx = spec_cplx / factor
            mag = torch.expm1(spec_cplx.abs())
            phase = spec_cplx.angle()
            spec_cplx = mag * torch.exp(1j * phase)

        elif transform_type == "none":
            pass

        return spec_cplx

    def get_window(self, window_type, window_length):
        """
        Build a window tensor for STFT based on the specified window type.

        Arguments
        ---------
        window_type: str
            Type of window function to use (e.g., 'hann', 'sqrthann').
        window_length: int
            The length of the window (e.g., n_fft).

        Returns
        -------
        window: torch.Tensor
            The generated window tensor of shape (window_length,).
        """
        if window_type == "sqrthann":
            return torch.sqrt(torch.hann_window(window_length, periodic=True))
        elif window_type == "hann":
            return torch.hann_window(window_length, periodic=True)
        else:
            raise NotImplementedError(
                f"Window type {window_type} not implemented!"
            )


def dataio_prep(hparams):
    """
    Prepare the datasets, launch training and evaluate the trained model.
    """
    seg_frames = hparams["segment_frames"]
    hop_length = hparams["hop_length"]
    target_len = (seg_frames - 1) * hop_length
    normalize = hparams.get("normalize", "noisy")
    data_dir = hparams["data_dir"]

    random_crop_train = hparams.get("random_crop_train", True)
    random_crop_valid = hparams.get("random_crop_valid", False)
    random_crop_test = hparams.get("random_crop_test", False)

    def build_pipeline(random_crop):
        @sb.utils.data_pipeline.takes("noisy_wav", "clean_wav")
        @sb.utils.data_pipeline.provides("noisy_sig", "clean_sig")
        def wav_pairs(noisy_wav, clean_wav):
            # Load waveforms
            sig_noisy = sb.dataio.dataio.read_audio(noisy_wav)
            sig_clean = sb.dataio.dataio.read_audio(clean_wav)

            orig_len = sig_clean.shape[-1]
            # Pad if too short
            if orig_len < target_len:
                needed = target_len - orig_len
                left_pad = needed // 2
                right_pad = needed - left_pad
                sig_noisy = F.pad(
                    sig_noisy, (left_pad, right_pad), mode="constant"
                )
                sig_clean = F.pad(
                    sig_clean, (left_pad, right_pad), mode="constant"
                )
            # Crop if too long
            elif orig_len > target_len:
                if random_crop:
                    start = np.random.randint(0, orig_len - target_len)
                else:
                    start = (orig_len - target_len) // 2
                sig_noisy = sig_noisy[..., start : start + target_len]
                sig_clean = sig_clean[..., start : start + target_len]

            # 5) normalize
            if normalize == "noisy":
                fac = sig_noisy.abs().max()
            elif normalize == "clean":
                fac = sig_clean.abs().max()
            else:
                fac = 1.0

            return sig_noisy / fac, sig_clean / fac

        return [wav_pairs]

    # create datasets
    datasets = {}
    for split, rc in zip(
        ["train", "valid", "test"],
        [random_crop_train, random_crop_valid, random_crop_test],
    ):
        pipelines = build_pipeline(rc)
        json_path = hparams[f"{split}_annotation"]
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_dir},
            dynamic_items=pipelines,
            output_keys=["id", "noisy_sig", "clean_sig"],
        )

    # optional length sorting
    if hparams["sorting"] in ("ascending", "descending"):
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="length", reverse=hparams["sorting"] == "descending"
        )
        hparams["train_dataloader_opts"]["shuffle"] = False

    return datasets


if __name__ == "__main__":
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to an existing run directory to resume.",
    )
    resume_args, remaining = cli.parse_known_args()

    hparams_file, run_opts, overrides = sb.parse_arguments(remaining)

    if resume_args.resume:  # Resume
        run_dir = Path(resume_args.resume).resolve()
        hparams_file = run_dir / "hyperparams.yaml"
        overrides = overrides or ""
    else:  # New
        run_name = f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        overrides = (overrides or "") + f"\nrun_name: '{run_name}'"

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    from voicebank_prepare import prepare_voicebank

    sb.utils.distributed.run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_dir"],
            "save_folder": hparams["data_dir"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create datasets
    datasets = dataio_prep(hparams)

    sb.create_experiment_directory(
        experiment_directory=os.path.join(
            hparams["output_folder"], hparams["run_name"]
        ),
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    sgmse_brain = SGMSEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Train
    sgmse_brain.fit(
        epoch_counter=sgmse_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Evaluate
    sgmse_brain.evaluate(
        test_set=datasets["test"],
        max_key="valid_loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
