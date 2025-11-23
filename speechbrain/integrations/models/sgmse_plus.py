"""
Speech enhancement and dereverberation using score-based generative models.

References:
[1] Richter, J., Welker, S., Lemercier, J.-M., Lay, B., & Gerkmann, T. (2023).
    Speech Enhancement and Dereverberation with Diffusion-based Generative Models.
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31, 2351-2364.
    https:/oi.org/10.1109/TASLP.2023.3285241
"""

from math import ceil

import sgmse.sampling as sampling
import torch
import torch.nn as nn
from sgmse.backbones import BackboneRegistry
from sgmse.sdes import SDERegistry
from torch_ema import ExponentialMovingAverage
from torch_pesq import PesqLoss


class ScoreModel(nn.Module):
    """
    Score-based generative model for speech enhancement.
    Encapsulates a backbone neural network and a stochastic differential equation (SDE)
    to perform denoising or data prediction in the spectrogram domain.

    Arguments
    ---------
    backbone: str
        Name of the backbone network architecture.
    sde: str
        Identifier of the SDE to use for diffusion sampling.
    lr: float
        Learning rate for optimizer.
    ema_decay: float
        Exponential moving average decay rate.
    t_eps: float
        Minimum time offset for numerical stability.
    num_eval_files: int
        Number of files to evaluate during validation.
    loss_type: str
        One of "score_matching", "denoiser", or "data_prediction".
    loss_weighting: str
        Weighting scheme for the loss (e.g., "sigma^2").
    network_scaling: str or None
        Scaling applied to network output.
    c_in: str
    c_out: str
    c_skip: str
        Coefficients for signal combinations.
    sigma_data: float
        Data noise standard deviation for EDM.
    l1_weight: float
        Weight for L1 term in data_prediction loss.
    pesq_weight: float
        Weight for PESQ loss term.
    sr: int
        Sample rate of audio.
    num_frames: int
        Number of time-frequency frames.
    hop_length: int
        Hop length between frames.
    **kwargs
        Arguments for creation of backbone.

    Example
    -------
    >>> # Note, this model should be trained before using in inference
    >>> from sgmse.util.other import pad_spec
    >>> sample_rate = 16000
    >>> noisy_audio = torch.rand(1, sample_rate)  # One second fake audio
    >>> noisy_spec = torch.stft(noisy_audio, n_fft=510, return_complex=True)
    >>> # pad for U-Net down-/up-sampling constraints
    >>> noisy_spec = pad_spec(noisy_spec.unsqueeze(1), mode="reflection")
    >>> model = ScoreModel(theta=1.5, sigma_min=0.05, sigma_max=0.5).to("cuda")
    >>> cleaned_spec = model.enhance(noisy_spec.to("cuda"))
    >>> cleaned_spec.shape
    torch.Size([1, 1, 256, 128])
    """

    def __init__(
        self,
        backbone="ncsnpp_v2",
        sde="ouve",
        lr=1e-4,
        ema_decay=0.999,
        t_eps=0.03,
        num_eval_files=20,
        loss_type="score_matching",
        loss_weighting="sigma^2",
        network_scaling=None,
        c_in="1",
        c_out="1",
        c_skip="0",
        sigma_data=0.1,
        l1_weight=0.001,
        pesq_weight=0.0,
        sr=16000,
        num_frames=256,
        hop_length=128,
        **kwargs,
    ):
        super().__init__()
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)

        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)

        # Save hyperparams
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(
            self.parameters(), decay=self.ema_decay
        )
        self._error_loading_ema = False

        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.network_scaling = network_scaling
        self.c_in = c_in
        self.c_out = c_out
        self.c_skip = c_skip
        self.sigma_data = sigma_data
        self.num_eval_files = num_eval_files
        self.num_frames = num_frames
        self.hop_length = hop_length
        self.sr = sr
        self.l1_weight = l1_weight
        self.pesq_weight = pesq_weight

        # PESQ loss, if used
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False

    def forward(self, x_t, y, t):
        """
        Computes the score or predicted clean data for a given noisy input and time step.

        Arguments
        ---------
        x_t: torch.Tensor
            The perturbed spectrogram at time `t`, of shape (B, 1, F, T).
        y: torch.Tensor
            The noisy input spectrogram of shape (B, 1, F, T).
        t: torch.Tensor
            The time step, of shape (B,).

        Returns
        -------
        torch.Tensor
            The computed score or the predicted clean data `x_hat`,
            depending on `self.loss_type`. Shape is (B, 1, F, T).
        """

        # In [3], we use new code with backbone='ncsnpp_v2':
        if self.backbone == "ncsnpp_v2":
            F = self.dnn(self._c_in(t) * x_t, self._c_in(t) * y, t)

            # Scaling the network output, see below Eq. (7) in the paper
            if self.network_scaling == "1/sigma":
                std = self.sde._std(t)
                F = F / std[:, None, None, None]
            elif self.network_scaling == "1/t":
                F = F / t[:, None, None, None]

            # The loss type determines the output of the model
            if self.loss_type == "score_matching":
                score = self._c_skip(t) * x_t + self._c_out(t) * F
                return score
            elif self.loss_type == "denoiser":
                sigmas = self.sde._std(t)[:, None, None, None]
                score = (F - x_t) / sigmas.pow(2)
                return score
            elif self.loss_type == "data_prediction":
                x_hat = self._c_skip(t) * x_t + self._c_out(t) * F
                return x_hat

        # In [1] and [2], we use the old code:
        else:
            dnn_input = torch.cat([x_t, y], dim=1)
            score = -self.dnn(dnn_input, t)
            return score

    def _step(self, batch, batch_idx):
        x, y = batch
        t = (
            torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps)
            + self.t_eps
        )
        mean, std = self.sde.marginal_prob(x, y, t)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z
        forward_out = self(x_t, y, t)
        loss = self._loss(forward_out, x_t, z, t, mean, x)
        return loss

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        elif self.c_in == "edm":
            sigma = self.sde._std(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[
                :, None, None, None
            ]
        else:
            raise ValueError(f"Invalid c_in type: {self.c_in}")

    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        elif self.c_out == "sigma":
            return self.sde._std(t)[:, None, None, None]
        elif self.c_out == "1/sigma":
            return 1.0 / self.sde._std(t)[:, None, None, None]
        elif self.c_out == "edm":
            sigma = self.sde._std(t)
            return (
                (sigma * self.sigma_data)
                / torch.sqrt(self.sigma_data**2 + sigma**2)
            )[:, None, None, None]
        else:
            raise ValueError(f"Invalid c_out type: {self.c_out}")

    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        elif self.c_skip == "edm":
            sigma = self.sde._std(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[
                :, None, None, None
            ]
        else:
            raise ValueError(f"Invalid c_skip type: {self.c_skip}")

    def get_pc_sampler(
        self,
        predictor_name,
        corrector_name,
        y,
        N=None,
        minibatch=None,
        **kwargs,
    ):
        """
        Get a predictor-corrector sampler for the SGMSE model.

        Arguments
        ---------
        predictor_name: str
            The name of the predictor to use.
        corrector_name: str
            The name of the corrector to use.
        y: torch.Tensor
            The noisy input spectrogram of shape (B, 1, F, T).
        N: int, optional
            The number of discretization steps. Defaults to `self.sde.N`.
        minibatch: int, optional
            The size of minibatches for batched sampling. Defaults to None.
        **kwargs
            Additional keyword arguments for the sampler.

        Returns
        -------
        function
            A sampling function that returns the enhanced sample and the number of function evaluations.
        """
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(
                predictor_name,
                corrector_name,
                sde=sde,
                score_fn=self,
                y=y,
                **kwargs,
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                """Batched sampling function for large inputs."""
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_pc_sampler(
                        predictor_name,
                        corrector_name,
                        sde=sde,
                        score_fn=self,
                        y=y_mini,
                        **kwargs,
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        """
        Get an ODE sampler for the SGMSE model.

        Arguments
        ---------
        y: torch.Tensor
            The noisy input spectrogram of shape (B, 1, F, T).
        N: int, optional
            The number of discretization steps. Defaults to `self.sde.N`.
        minibatch: int, optional
            The size of minibatches for batched sampling. Defaults to None.
        **kwargs
            Additional keyword arguments for the sampler.

        Returns
        -------
        function
            A sampling function that returns the enhanced sample and the number of function evaluations.
        """
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                """Batched sampling function for large inputs."""
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_ode_sampler(
                        sde, self, y=y_mini, **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns

            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        """
        Get a Schrödinger bridge sampler for the SGMSE model.

        Arguments
        ---------
        sde: sgmse.sdes.SDE
            The SDE object for the Schrödinger bridge.
        y: torch.Tensor
            The noisy input spectrogram of shape (B, 1, F, T).
        sampler_type: str, optional
            The type of sampler to use ("ode" or "pc"). Defaults to "ode".
        N: int, optional
            The number of discretization steps. Defaults to `sde.N`.
        **kwargs
            Additional keyword arguments for the sampler.

        Returns
        -------
        function
            A sampling function that returns the enhanced sample and the number of function evaluations.
        """
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N

        return sampling.get_sb_sampler(
            sde, self, y=y, sampler_type=sampler_type, **kwargs
        )

    def enhance(
        self,
        y,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        timeit=False,
        **kwargs,
    ):
        """
        One-call speech enhancement from a noisy input.

        This method runs the chosen SGMSE sampler to produce an enhanced spectrogram (or
        other representation) from the input `y`, which is assumed to be a
        spectrogram.

        Arguments
        ---------
        y: torch.Tensor
            The noisy input spectrogram of shape
            (B, 1, F, T).
        sampler_type: str, optional
            The type of sampler to use, e.g. "pc" or "ode".
            Defaults to "pc".
        predictor: str, optional
            The predictor method used in the sampler,
            e.g. "reverse_diffusion". Defaults to "reverse_diffusion".
        corrector: str, optional
            The corrector method used in the sampler, e.g. "ald".
            Defaults to "ald".
        N: int, optional
            Number of discretization steps for the SDE solver. Defaults to 30.
        corrector_steps: int, optional
            Number of corrector steps per iteration.
            Defaults to 1.
        snr: float, optional
            Step-size adaptation factor for the sampler. Defaults to 0.5.
        timeit: bool, optional
            If True, measure the runtime for enhancement. Defaults to False.
        **kwargs
            Additional keyword arguments passed to the sampler.

        Returns
        -------
        sample: torch.Tensor
            The sampled (enhanced) output from the model. Retains
            the same shape (B, 1, F, T) as the input `y`.
        """
        # SGMSE sampling with OUVE SDE
        if self.sde.__class__.__name__ == "OUVESDE":
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(
                    predictor,
                    corrector,
                    y.cuda(),
                    N=N,
                    corrector_steps=corrector_steps,
                    snr=snr,
                    intermediate=False,
                    **kwargs,
                )
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(y.cuda(), N=N, **kwargs)
            else:
                raise ValueError(
                    f"Invalid sampler type for SGMSE sampling: {sampler_type}"
                )
        # Schrödinger bridge sampling with VE SDE
        elif self.sde.__class__.__name__ == "SBVESDE":
            sampler = self.get_sb_sampler(
                sde=self.sde, y=y.cuda(), sampler_type=self.sde.sampler_type
            )
        else:
            raise ValueError(
                f"Invalid SDE type for speech enhancement: {self.sde.__class__.__name__}"
            )
        sample, _ = sampler()
        return sample

    def compute_loss(
        self,
        forward_out,
        x_t,
        z,
        t,
        mean,
        x,
        reduction="mean",
        to_audio_func=None,
    ):
        """
        Compute the loss for the score-based generative model.

        This function computes the loss according to the specified loss type, which can be one of:
        "score_matching", "denoiser", or "data_prediction". For the "data_prediction" loss, the function
        requires a callable to transform spectrogram data back to the time domain.

        Arguments
        ---------
        forward_out: torch.Tensor
            Predicted output from the score model of shape (B, 1, F, T).
        x_t: torch.Tensor
            Noisy input signal at time t in the spectrogram domain of shape (B, 1, F, T).
        z: torch.Tensor
            Noise or perturbation tensor of shape (B, 1, F, T).
        t: torch.Tensor
            Time-step tensor for the diffusion process of shape (B,).
        mean: torch.Tensor
            Estimated mean (clean signal) from the model of shape (B, 1, F, T).
        x: torch.Tensor
            Ground-truth clean signal in the spectrogram domain of shape (B, 1, F, T).
        reduction: str
            Specifies the reduction to apply to the per-sample loss. "mean" returns a scalar loss,
            whereas "none" returns a tensor of shape (B,) with the loss for each sample.
        to_audio_func: callable
            Function that converts spectrogram data to time-domain audio. This must be provided
            when using the "data_prediction" loss type.

        Returns
        -------
        loss: torch.Tensor
            Computed loss. If reduction is "mean", the returned tensor is a scalar; if "none",
            the returned tensor is of shape (B,) representing the loss per sample.
        """
        sigma = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "score_matching":
            score = forward_out
            if self.loss_weighting == "sigma^2":
                losses = torch.square(torch.abs(score * sigma + z))  # Eq. (7)
            else:
                raise ValueError(
                    f"Invalid loss weighting for loss_type=score_matching: {self.loss_weighting}"
                )
            # Compute per-sample losses by summing over spatial dimensions
            per_sample_loss = 0.5 * torch.sum(
                losses.reshape(losses.shape[0], -1), dim=-1
            )

        elif self.loss_type == "denoiser":
            score = forward_out
            D = score * sigma.pow(2) + x_t  # equivalent to Eq. (10)
            losses = torch.square(torch.abs(D - mean))  # Eq. (8)
            if self.loss_weighting == "1":
                pass
            elif self.loss_weighting == "sigma^2":
                losses = losses * sigma**2
            elif self.loss_weighting == "edm":
                losses = (
                    (sigma**2 + self.sigma_data**2)
                    / ((sigma * self.sigma_data) ** 2)
                )[:, None, None, None] * losses
            else:
                raise ValueError(
                    f"Invalid loss weighting for loss_type=denoiser: {self.loss_weighting}"
                )
            per_sample_loss = 0.5 * torch.sum(
                losses.reshape(losses.shape[0], -1), dim=-1
            )

        elif self.loss_type == "data_prediction":
            if to_audio_func is None:
                raise ValueError(
                    "to_audio_func must be provided for data prediction loss"
                )

            x_hat = forward_out
            B, C, F, T = x.shape

            # losses in the time-frequency domain (tf)
            losses_tf = (1 / (F * T)) * torch.square(torch.abs(x_hat - x))
            losses_tf = 0.5 * torch.sum(
                losses_tf.reshape(losses_tf.shape[0], -1), dim=-1
            )

            # losses in the time domain (td)
            target_len = (self.num_frames - 1) * self.hop_length
            x_hat_td = to_audio_func(x_hat.squeeze(), target_len)
            x_td = to_audio_func(x.squeeze(), target_len)
            losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
            losses_l1 = 0.5 * torch.sum(
                losses_l1.reshape(losses_l1.shape[0], -1), dim=-1
            )

            if self.pesq_weight > 0.0:
                losses_pesq = self.pesq_loss(x_td, x_hat_td)
                losses_pesq = torch.mean(
                    losses_pesq
                )  # Assuming pesq_loss returns per-sample losses
                per_sample_loss = (
                    losses_tf
                    + self.l1_weight * losses_l1
                    + self.pesq_weight * losses_pesq
                )
            else:
                per_sample_loss = losses_tf + self.l1_weight * losses_l1
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        if reduction == "mean":
            return torch.mean(per_sample_loss)
        elif reduction == "none":
            return per_sample_loss
        else:
            raise ValueError("Invalid reduction type")

    def update_ema(self):
        """Call this after each optimizer step to update the EMA weights."""
        self.ema.update(self.dnn.parameters())

    def store_ema(self):
        """Call this before evaluation if you want to switch to EMA weights."""
        self.ema.store(self.dnn.parameters())
        self.ema.copy_to(self.dnn.parameters())

    def restore_ema(self):
        """Call this after evaluation if you stored EMA weights and want to restore normal weights."""
        self.ema.restore(self.dnn.parameters())

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)
