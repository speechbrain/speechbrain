import datetime
from pathlib import Path
import torch
import speechbrain as sb
import torch.nn.functional as F
import soundfile as sf
import os
import numpy as np

from speechbrain.nnet.loss.stoi_loss import stoi_loss
from pesq import pesq
from torch.utils.tensorboard import SummaryWriter 

from speechbrain.utils.metric_stats import MetricStats
from speechbrain.lobes.models.sgmse.util.other import pad_spec # input needs to be padded to certain length to go through backbone

class SGMSEBrain(sb.Brain):
    """
    A Brain class to train an SGMSE-based diffusion model.
    """
    def on_fit_start(self):
        """
        Called once in the beginning of training. 

        Parameters:
        - None
        
        Returns:
        - None
        """
        super().on_fit_start()

        # Build a unique run name: e.g. run_2023-04-10_13-40-55
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"run_{timestamp}"

        # Logging
        log_dir = os.path.join(self.hparams.tensorboard_logs, run_name)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # Checkpointing
        checkpoints_dir = Path(self.hparams.save_folder) / run_name
        os.makedirs(checkpoints_dir, exist_ok=True)
        self.checkpointer.checkpoints_dir = checkpoints_dir # Override the path where Checkpointer saves files

        ema = self.modules["score_model"].ema
        self.checkpointer.add_recoverable(
            name="ema",
            obj=ema,
            custom_save_hook=lambda obj, path: torch.save(obj.state_dict(), path),
            custom_load_hook=lambda obj, path, end: obj.load_state_dict(torch.load(path)),
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

    def _step(self, x, y, model):
        """
        Perform a single diffusion step for the score-based model.

        This function samples a random time-step for each input in the batch, computes the corresponding
        marginal probability of the clean signal using the SDE, adds noise to generate a noisy version
        of the input (x_t), and obtains the model's prediction from this noisy input.

        Parameters:
        - x (torch.Tensor): Clean input signal spectrogram, of shape (B, 1, F, T).
        - y (torch.Tensor): Conditioning or auxiliary input spectrogram, of shape (B, 1, F, T).
        - model (nn.Module): Score-based generative model that contains the SDE and the forward method.

        Returns:
        - forward_out (torch.Tensor): Model's prediction computed from the noisy input x_t, of shape (B, 1, F, T).
        - x_t (torch.Tensor): Noisy version of the clean input generated using the SDE, of shape (B, 1, F, T).
        - z (torch.Tensor): Noise tensor sampled from a standard normal distribution, of shape (B, 1, F, T).
        - t (torch.Tensor): Randomly sampled time-step for each sample in the batch, of shape (B,).
        - mean (torch.Tensor): Mean of the marginal probability distribution for the clean input, of shape (B, 1, F, T).
        - x (torch.Tensor): Clean input signal spectrogram, of shape (B, 1, F, T).
        """
        t = torch.rand(x.shape[0], device=x.device) * (model.sde.T - model.t_eps) + model.t_eps # (B,)
        mean, std = model.sde.marginal_prob(x, y, t) # (B,1,F,T) and (B,) respectively
        z = torch.randn_like(x)  # (B,1,F,T), i.i.d. normal distributed with var=0.5
        sigma = std[:, None, None, None] # (B,1,1,1)
        x_t = mean + sigma * z # (B,1,F,T)
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

        Parameters:
        - batch (speechbrain.dataio.batch.BrainBatch): A batch of data containing 
          clean and noisy signals, among other possible fields.
        - stage (sb.Stage): The current stage (TRAIN, VALID, TEST).

        Returns:
        - outs (dict): A dictionary containing the forward pass outputs (including 
          the model prediction and any enhanced waveforms if generated).
        """
        # Model and batch preparation
        model = self.modules["score_model"]
        batch = batch.to(self.device)

        # Extract waveforms
        x_wav = batch.clean_sig.data # (B,S)
        y_wav = batch.noisy_sig.data # (B,S)

        # STFT
        x = self.stft(x_wav) # (B,F,T) 
        y = self.stft(y_wav) # (B,F,T) 

        # Spec transformations
        x = self.spec_fwd(x) # (B,F,T) 
        y = self.spec_fwd(y) # (B,F,T) 

        # Adding channel dimension needed because backbone expects 4 dimensions
        x = x.unsqueeze(1) # (B,1,F,T) 
        y = y.unsqueeze(1) # (B,1,F,T)

        # pad_mode = "reflection"  
        # x = pad_spec(x, mode=pad_mode) # TODO: necessary here?
        # y = pad_spec(y, mode=pad_mode) 

        outs = self._step(x, y, model) 

        if stage == sb.Stage.TEST or stage == sb.Stage.VALID and self.eval_files_left > 0:
            x_wav = batch.clean_sig.data  # (B,S) 
            y_wav = batch.noisy_sig.data  # (B,S)

            # how many files from current batch shall we process?
            B = y_wav.size(0)
            take = min(B, self.eval_files_left)
            self.eval_files_left -= take

            # slice to that number
            x_wav = x_wav[:take] # (num_eval_files,S)
            y_wav = y_wav[:take] # (num_eval_files,S)
            uttids = batch.id[:take]

            # Save original length in time dimension
            T_orig_wav = y_wav.size(1)
            
            # STFT
            x = self.stft(x_wav) # (num_files,F,T) 
            y = self.stft(y_wav) # (num_files,F,T) 
            F_orig_spec, T_orig_spec = y.shape[-2:] # save for removing padding

            # Spec transformations
            x = self.spec_fwd(x) # (num_files,F,T) 
            y = self.spec_fwd(y) # (num_files,F,T) 

            # Adding channel dimension needed because backbone expects 4 dimensions
            x = x.unsqueeze(1) # (num_files,1,F,T) 
            y = y.unsqueeze(1) # (num_files,1,F,T) 

            # pad_mode = "reflection"
            # x = pad_spec(x, mode=pad_mode)
            # y = pad_spec(y, mode=pad_mode)

            # Enhancement 
            x_hat = model.enhance( #TODO: dont hardcode enhancement params
                y,
                sampler_type="pc", 
                predictor="reverse_diffusion",
                corrector="ald",
                N=model.sde.N, corrector_steps=1, snr=0.5
            ) # (num_files, 1, F, T)

            # # Remove channel dim *and* padding
            # x_hat = x_hat[:, :, :F_orig_spec, :T_orig_spec].squeeze(1)   # (num_files, F_orig_spec, T_orig_spec)
            # x     = x[:, :, :F_orig_spec, :T_orig_spec].squeeze(1)       # crop the reference too

            x_hat = x_hat.squeeze(1)
            x = x.squeeze(1)

            # Reverse spech transformations
            x_hat = self.spec_back(x_hat) # (num_files, F, T)
            x = self.spec_back(x) # (num_files, F, T)

            # iSTFT
            x_hat_wav = self.istft(x_hat, T_orig_wav) # (num_files, S)
            x_wav = self.istft(x, T_orig_wav) # (num_files, S)

            outs.update({                                             
                "x_hat_wav": x_hat_wav, 
                "x_wav":     x_wav,
                "uttids":    uttids      # so compute_objectives can see them
            })

        return outs

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the diffusion loss and optionally processes enhanced waveforms.

        This method takes the outputs of `compute_forward` (which include 
        the model's prediction and possibly enhanced waveforms), computes 
        the loss for training or collects metrics for validation/testing.

        Parameters:
        - predictions (dict): Dictionary containing forward pass outputs, 
          e.g. from `_step`.
        - batch (speechbrain.dataio.batch.BrainBatch): The current batch, which 
          can be used for retrieving IDs or additional data if needed.
        - stage (sb.Stage): The current stage (TRAIN, VALID, TEST).

        Returns:
        - loss (torch.Tensor): The computed diffusion loss for this batch.
        """  
        model = self.modules["score_model"]

        # Extract items from predictions
        forward_out = predictions["forward_out"] # (B,1,F,T)
        x_t         = predictions["x_t"] # (B,1,F,T)
        z           = predictions["z"] # (B,1,F,T)
        t           = predictions["t"] # (B,)
        mean        = predictions["mean"] # (B,1,F,T)
        x           = predictions["x"] # (B,1,F,T)

        # Pass the necessary inputs to the model loss
        loss = model.compute_loss(forward_out, x_t, z, t, mean, x, to_audio_func=self.to_audio) #TODO: how to pass the to_audio func for data prediction loss?
        self.loss_metric.append(batch.id, forward_out, x_t, z, t, mean, x)
        
        # Only process enhanced wavs if they exist
        if stage != sb.Stage.TRAIN:
            x_wav = predictions.get("x_wav", None)
            x_hat_wav = predictions.get("x_hat_wav", None)
            uttids     = predictions.get("uttids", None) 
            if x_wav is not None:
                # STOI
                lens = torch.full((x_wav.shape[0],), x_wav.shape[1], device=x_wav.device) # TODO: data pipeline provide the true lengths
                self.stoi_metric.append(batch.id, x_wav, x_hat_wav, lens, reduction="batch")

                # PESQ
                x_wav_cpu = x_wav.cpu()
                x_hat_wav_cpu = x_hat_wav.cpu()
                self.pesq_metric.append(batch.id, target=x_wav_cpu, predict=x_hat_wav_cpu)

                sr = self.hparams.sample_rate
                save_folder = self.hparams.enhanced_folder
                os.makedirs(save_folder, exist_ok=True)
      
                epoch_tag = f"Ep{self.hparams.epoch_counter.current}" if stage == sb.Stage.VALID else ""                         
                for i, uid in enumerate(uttids):
                    clean_path = os.path.join(save_folder, f"{epoch_tag}_{uid}_clean.wav")
                    enh_path   = os.path.join(save_folder, f"{epoch_tag}_{uid}_enhanced.wav")
                    sf.write(clean_path, x_wav_cpu[i].numpy(), sr)
                    sf.write(enh_path,  x_hat_wav_cpu[i].numpy(), sr)
                    
        return loss

    def fit_batch(self, batch):
        """
        Overridden method to train on a single batch.

        This performs the typical Brain forward-backward-update 
        steps, and can include updates for EMA.

        Parameters:
        - batch (speechbrain.dataio.batch.BrainBatch): The batch of data used 
          for training.

        Returns:
        - loss (torch.Tensor): The computed training loss for the batch.
        """
        # Standard "forward" + "objectives"
        loss = super().fit_batch(batch)

        # Update EMA for the diffusion model
        self.modules["score_model"].update_ema()

        return loss

    def on_stage_start(self, stage, epoch=None):
        """
        Called at the beginning of each stage (TRAIN, VALID, TEST).

        This method initializes or resets metrics for that stage. 
        It can also be used to set flags or other stage-specific fields.

        Parameters:
        - stage (sb.Stage): The current stage (TRAIN, VALID, TEST).
        - epoch (int, optional): The current epoch number, if applicable.

        Returns:
        - None
        """
        self.loss_metric = MetricStats(
            metric=lambda forward_out, x_t, z, t, mean, x:
                self.modules["score_model"].compute_loss(forward_out, x_t, z, t, mean, x, reduction="none")
        )
        
        self.stoi_metric = MetricStats(metric=stoi_loss)

        # Define function taking (prediction, target) for parallel eval
        def pesq_eval(pred_wav, target_wav):
            """Computes the PESQ evaluation metric"""
            return pesq(
                fs=self.hparams.sample_rate,
                ref=target_wav.numpy().squeeze(),
                deg=pred_wav.numpy().squeeze(),
                mode="wb",
            )
        
        if stage != sb.Stage.TRAIN:
            self.modules["score_model"].store_ema()
            self.pesq_metric = MetricStats(
                metric=pesq_eval, batch_eval=False, n_jobs=1
            )
    
        if stage == sb.Stage.VALID:
                self.eval_files_left = self.hparams.modules['score_model'].num_eval_files
                self.save_counter = 0  
                self.modules["score_model"].store_ema() # Swap in the EMA weights for evaluation

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        Called at the end of each stage (TRAIN, VALID, TEST).

        Summarizes and prints the loss metrics, and for non-training stages,
        prints additional evaluation metrics (e.g., PESQ, STOI).

        Parameters:
        - stage (sb.Stage): The current stage (TRAIN, VALID, TEST).
        - stage_loss (torch.Tensor): The aggregated loss over the stage.
        - epoch (int, optional): The current epoch number, if applicable.

        Returns:
        - None
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

        # Tensorboard logging and additional metrices
        if stage == sb.Stage.TRAIN:
            # Log training loss
            self.writer.add_scalar("Loss/train", avg_loss, epoch) # TODO: if fails, reintroduce the below
            # if epoch is not None:
            #     self.writer.add_scalar("Loss/train", avg_loss, epoch)
            # else:
            #     self.writer.add_scalar("Loss/train", avg_loss)

        elif stage == sb.Stage.VALID:
            self.modules["score_model"].restore_ema()

            # Summarize PESQ
            avg_pesq = self.pesq_metric.summarize("average")
            print(f"Avg PESQ: {avg_pesq:.4f}")

            # Summarize STOI
            avg_stoi = self.stoi_metric.summarize("average")
            print(f"Avg STOI: {avg_stoi:.4f}")
            
            # TODO: if this fails, reuse the below
            self.writer.add_scalar("Loss/valid", avg_loss, epoch)
            self.writer.add_scalar("Metric/PESQ_valid", avg_pesq, epoch)
            self.writer.add_scalar("Metric/STOI_valid", avg_stoi, epoch)
            
            # if epoch is not None:
            #     self.writer.add_scalar("Loss/valid", avg_loss, epoch)
            #     self.writer.add_scalar("Metric/PESQ_valid", avg_pesq, epoch)
            #     self.writer.add_scalar("Metric/STOI_valid", avg_stoi, epoch)
            # else:
            #     self.writer.add_scalar("Loss/valid", avg_loss)
            #     self.writer.add_scalar("Metric/PESQ_valid", avg_pesq)
            #     self.writer.add_scalar("Metric/STOI_valid", avg_stoi)

            self.checkpointer.save_and_keep_only(
                meta={"valid_loss": avg_loss}, 
                min_keys=["valid_loss"], 
                num_to_keep=2 # TODO: read from hparams
            )

        elif stage == sb.Stage.TEST:
            self.modules["score_model"].restore_ema()
            # Summarize the PESQ
            avg_pesq = self.pesq_metric.summarize("average")
            print(f"Avg PESQ: {avg_pesq:.4f}")

            # Summarize STOI
            avg_stoi = self.stoi_metric.summarize("average")
            print(f"Avg STOI: {avg_stoi:.4f}")

            # there's no epoch concept in test stage so i take 0:
            self.writer.add_scalar("Loss/test", avg_loss, 0)
            self.writer.add_scalar("Metric/PESQ_test", avg_pesq, 0)
            self.writer.add_scalar("Metric/STOI_test", avg_stoi, 0)

        # Manually write to disk to ensure real time updates
        self.writer.flush()

    def on_evaluate_start(self, max_key=None, min_key=None):
        """
        Prepares evaluation.

        Parameters:
        - max_key (str, optional): Key used to track maximum metric value.
        - min_key (str, optional): Key used to track minimum metric value.

        Returns:
        - None
        """
        # Swap in the EMA weights for evaluation
        self.modules["score_model"].store_ema()
        super().on_evaluate_start(max_key=max_key, min_key=min_key)

    def on_evaluate_end(self):
        """
        Restore original weights after evaluation.

        Parameters:
        - None

        Returns:
        - None
        """
        # Restore original weights
        self.modules["score_model"].restore_ema()
        super().on_evaluate_end()

    # ---------------------------
    # STFT and Spec transforms
    # ---------------------------
    def to_audio(brain, spec, length=None):
        """
        Convert a complex spectrogram into time-domain audio.

        This method applies `spec_back` to invert the spectral transform
        (log or exponent, if used), followed by iSTFT to return time-domain
        waveforms.

        Parameters:
        - spec (torch.Tensor): Complex spectrogram of shape (B, F, T)
        - length (int, optional): The target number of samples in the output signal.

        Returns:
        - audio (torch.Tensor): Time-domain waveform, shape (B, S)
        """
        return brain.istft(brain.spec_back(spec), length=length)

    def stft(self, sig):
        """
        Compute the short-time Fourier transform (STFT) of the given signal.

        Parameters:
        - sig (torch.Tensor): Time-domain signal of shape (B, S).

        Returns:
        - spec (torch.Tensor): Complex STFT, shape (B, F, T).
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

        Parameters:
        - spec (torch.Tensor): Complex STFT of shape (B, F, T).
        - length (int, optional): The desired number of samples in the output.

        Returns:
        - waveform (torch.Tensor): Time-domain signal of shape (B, S).
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

        Parameters:
        - spec_cplx (torch.Tensor): Complex spectrogram of shape (B, F, T).

        Returns:
        - spec_trans (torch.Tensor): Transformed complex spectrogram of the same shape.
        """
        # Access hyperparams from self.hparams
        transform_type = self.hparams.transform_type
        factor         = self.hparams.spec_factor
        e              = getattr(self.hparams, "spec_abs_exponent", 1.0)

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

        Parameters:
        - spec_cplx (torch.Tensor): Complex spectrogram of shape (B, F, T).

        Returns:
        - spec_orig (torch.Tensor): Original-like complex spectrogram of the same shape.
        """
        transform_type = self.hparams.transform_type
        factor         = self.hparams.spec_factor
        e              = getattr(self.hparams, "spec_abs_exponent", 1.0)

        if transform_type == "exponent":
            spec_cplx = spec_cplx / factor
            if e != 1.0:
                mag = spec_cplx.abs() ** (1.0/e)
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

        Parameters:
        - window_type (str): Type of window function to use (e.g., 'hann', 'sqrthann').
        - window_length (int): The length of the window (e.g., n_fft).

        Returns:
        - window (torch.Tensor): The generated window tensor of shape (window_length,).
        """
        if window_type == 'sqrthann':
            return torch.sqrt(torch.hann_window(window_length, periodic=True))
        elif window_type == 'hann':
            return torch.hann_window(window_length, periodic=True)
        else:
            raise NotImplementedError(f"Window type {window_type} not implemented!")