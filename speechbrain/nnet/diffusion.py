"""An implementation of Denoising Diffusion

https://arxiv.org/pdf/2006.11239.pdf

Certain parts adopted from / inspired by denoising-diffusion-pytorch
https://github.com/lucidrains/denoising-diffusion-pytorch

Authors
 * Artem Ploujnikov 2022
"""

from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from speechbrain.utils.data_utils import unsqueeze_as
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils import data_utils


class Diffuser(nn.Module):
    """A base diffusion implementation

    Arguments
    ---------
    model: nn.Module
        the underlying model
    """

    def __init__(self, model, timesteps, noise=None):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        if noise is None:
            noise = "gaussian"
        if isinstance(noise, str):
            self.noise = _NOISE_FUNCTIONS[noise]
        else:
            self.noise = noise

    def distort(self, x, timesteps=None):
        """Adds noise to a batch of data

        Arguments
        ---------
        x: torch.Tensor
            the original data sample
        timesteps: torch.Tensor
            a 1-D integer tensor of a length equal to the number of
            batches in x, where each entry corresponds to the timestep
            number for the batch. If omitted, timesteps will be randomly
            sampled

        Returns
        -------
        result: torch.Tensor
            a tensor of the same dimension as x
        """
        raise NotImplementedError

    def train_sample(self, x, timesteps=None, condition=None, **kwargs):
        """Creates a sample for the training loop with a
        corresponding target
        Arguments
        ---------
        x: torch.Tensor
            the original data sample
        timesteps: torch.Tensor
            a 1-D integer tensor of a length equal to the number of
            batches in x, where each entry corresponds to the timestep
            number for the batch. If omitted, timesteps will be randomly
            sampled
        condition: torch.tensor
            the condition used for conditional generation
            Should be omitted during unconditional generation
        Returns
        -------
        pred: torch.Tensor
            the model output 0 prdicted noise
        noise: torch.Tensor
            the noise being applied
        noisy_sample
            the sample with the noise applied
        """
        if timesteps is None:
            timesteps = sample_timesteps(x, self.timesteps)
        noisy_sample, noise = self.distort(x, timesteps=timesteps, **kwargs)

        # in case that certain models do not have any condition as input
        if condition is None:
            pred = self.model(noisy_sample, timesteps, **kwargs)
        else:
            pred = self.model(noisy_sample, timesteps, condition, **kwargs)
        return pred, noise, noisy_sample

    def sample(self, shape, **kwargs):
        """Generates the number of samples indicated by the
        count parameter

        Arguments
        ---------
        shape: enumerable
            the shape of the sample to generate


        Returns
        -------
        result: torch.Tensor
            the generated sample(s)
        """
        raise NotImplementedError

    def forward(self, x, timesteps=None):
        """Computes the forward pass, calls distort()
        """
        return self.distort(x, timesteps)


DDPM_DEFAULT_BETA_START = 0.0001
DDPM_DEFAULT_BETA_END = 0.02
DDPM_REF_TIMESTEPS = 1000
DESC_SAMPLING = "Diffusion Sampling"


class DenoisingDiffusion(Diffuser):
    """An implementation of a classic Denoising Diffusion Probabilistic Model (DDPM)

    Arguments
    ---------
    model: nn.Module
        the underlying model

    timesteps: int
        the number of timesteps

    noise: str|nn.Module
        the type of noise being used
        "gaussian" will produce standard Gaussian noise


    beta_start: float
        the value of the "beta" parameter at the beginning at the end of the process
        (see the paper)

    beta_end: float
        the value of the "beta" parameter at the end of the process

    show_progress: bool
        whether to show progress during inference
    """

    def __init__(
        self,
        model,
        timesteps=None,
        noise=None,
        beta_start=None,
        beta_end=None,
        sample_min=None,
        sample_max=None,
        show_progress=False,
    ):
        if timesteps is None:
            timesteps = DDPM_REF_TIMESTEPS
        super().__init__(model, timesteps=timesteps, noise=noise)
        if beta_start is None or beta_end is None:
            scale = DDPM_REF_TIMESTEPS / timesteps
            if beta_start is None:
                beta_start = scale * DDPM_DEFAULT_BETA_START
            if beta_end is None:
                beta_end = scale * DDPM_DEFAULT_BETA_END
        self.beta_start = beta_start
        self.beta_end = beta_end
        alphas, betas = self.compute_coefficients()
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)
        alphas_cumprod = self.alphas.cumprod(dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        signal_coefficients = torch.sqrt(alphas_cumprod)
        noise_coefficients = torch.sqrt(1.0 - alphas_cumprod)
        self.register_buffer("signal_coefficients", signal_coefficients)
        self.register_buffer("noise_coefficients", noise_coefficients)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", posterior_variance.log())
        posterior_mean_weight_start = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_weight_step = (
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_weight_start", posterior_mean_weight_start
        )
        self.register_buffer(
            "posterior_mean_weight_step", posterior_mean_weight_step
        )
        sample_pred_model_coefficient = (1.0 / alphas_cumprod).sqrt()

        self.register_buffer(
            "sample_pred_model_coefficient", sample_pred_model_coefficient
        )
        sample_pred_noise_coefficient = (1.0 / alphas_cumprod - 1).sqrt()
        self.register_buffer(
            "sample_pred_noise_coefficient", sample_pred_noise_coefficient
        )
        self.sample_min = sample_min
        self.sample_max = sample_max
        self.show_progress = show_progress

    def compute_coefficients(self):
        """Computes diffusion coefficients (alphas and betas)"""
        betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        alphas = 1.0 - betas
        return alphas, betas

    def distort(self, x, noise=None, timesteps=None, **kwargs):
        """Adds noise to the sample, in a forward diffusion process,

        Arguments
        ---------
        x: torch.Tensor
            a data sample of 2 or more dimensions, with the
            first dimension representing the batch
        noise: torch.Tensor
            the noise to add
        timesteps: torch.Tensor
            a 1-D integer tensor of a length equal to the number of
            batches in x, where each entry corresponds to the timestep
            number for the batch. If omitted, timesteps will be randomly
            sampled

        Returns
        -------
        result: torch.Tensor
            a tensor of the same dimension as x
        """
        if timesteps is None:
            timesteps = sample_timesteps(x, self.timesteps)
        if noise is None:
            noise = self.noise(x, **kwargs)
        signal_coefficients = self.signal_coefficients[timesteps]
        noise_coefficients = self.noise_coefficients[timesteps]
        noisy_sample = (
            unsqueeze_as(signal_coefficients, x) * x
            + unsqueeze_as(noise_coefficients, noise) * noise
        )
        return noisy_sample, noise

    @torch.no_grad()
    def sample(self, shape, **kwargs):
        """Generates the number of samples indicated by the
        count parameter

        Arguments
        ---------
        shape: enumerable
            the shape of the sample to generate


        Returns
        -------
        result: torch.Tensor
            the generated sample(s)
        """
        sample = self.noise(torch.zeros(*shape, device=self.alphas.device))
        steps = reversed(range(self.timesteps))
        if self.show_progress:
            steps = tqdm(steps, desc=DESC_SAMPLING, total=self.timesteps)
        for timestep_number in steps:
            timestep = (
                torch.ones(
                    shape[0], dtype=torch.long, device=self.alphas.device
                )
                * timestep_number
            )
            sample = self.sample_step(sample, timestep, **kwargs)
        return sample

    @torch.no_grad()
    def sample_step(self, sample, timestep, **kwargs):
        """Processes a single timestep for the sampling
        process

        Arguments
        ---------
        sample: torch.Tensor
            the sample for the following timestep
        timestep: int
            the timestep number

        Arguments
        ---------
        predicted_sample: torch.Tensor
            the predicted sample (denoised by one step`)
        """
        model_out = self.model(sample, timestep, **kwargs)
        noise = self.noise(sample)
        sample_start = (
            unsqueeze_as(self.sample_pred_model_coefficient[timestep], sample)
            * sample
            - unsqueeze_as(
                self.sample_pred_noise_coefficient[timestep], model_out
            )
            * model_out
        )
        weight_start = unsqueeze_as(
            self.posterior_mean_weight_start[timestep], sample_start
        )
        weight_step = unsqueeze_as(
            self.posterior_mean_weight_step[timestep], sample
        )
        mean = weight_start * sample_start + weight_step * sample
        log_variance = unsqueeze_as(
            self.posterior_log_variance[timestep], noise
        )
        predicted_sample = mean + (0.5 * log_variance).exp() * noise
        if self.sample_min is not None or self.sample_max is not None:
            predicted_sample.clip_(min=self.sample_min, max=self.sample_max)
        return predicted_sample

    @torch.no_grad()
    def diffwave_inference(
        self,
        unconditional,
        scale,
        condition=None,
        fast_sampling=False,
        fast_sampling_noise_schedule=None,
        device=torch.device("cuda"),
    ):
        """Processes the inference for diffwave
        One inference function for all the locally/globally conditional
        generation and unconditional generation tasks
        Arguments
        ---------
        unconditional: bool
            do unconditional generation if True, else do conditional generation
        scale: int
            scale to get the final output wave length
            for conditional genration, the output wave length is scale * condition.shape[-1]
            for example, if the condition is spectrogram (bs, n_mel, time), scale should be hop length
            for unconditional generation, scale should be the desired audio length
        condition: torch.tensor
            input spectrogram for vocoding or other conditions for other
            conditional generation, should be None for unconditional generation
        fast_sampling: bool
            whether to do fast sampling
        fast_sampling_noise_schedule: list
            the noise schedules used for fast sampling
        device:
            inference device
        Returns
        ---------
        predicted_sample: torch.Tensor
            the predicted audio (bs, 1, t)
        """
        # either condition or uncondition
        if unconditional:
            assert condition is None
        else:
            assert condition is not None
            device = condition.device

        # must define fast_sampling_noise_schedule during fast sampling
        if fast_sampling:
            assert fast_sampling_noise_schedule is not None

        if fast_sampling and fast_sampling_noise_schedule is not None:
            inference_noise_schedule = fast_sampling_noise_schedule
            inference_alphas = 1 - torch.tensor(inference_noise_schedule)
            inference_alpha_cum = inference_alphas.cumprod(dim=0)
        else:
            inference_noise_schedule = self.betas
            inference_alphas = self.alphas
            inference_alpha_cum = self.alphas_cumprod

        inference_steps = []
        for s in range(len(inference_noise_schedule)):
            for t in range(self.timesteps - 1):
                if (
                    self.alphas_cumprod[t + 1]
                    <= inference_alpha_cum[s]
                    <= self.alphas_cumprod[t]
                ):
                    twiddle = (
                        self.alphas_cumprod[t] ** 0.5
                        - inference_alpha_cum[s] ** 0.5
                    ) / (
                        self.alphas_cumprod[t] ** 0.5
                        - self.alphas_cumprod[t + 1] ** 0.5
                    )
                    inference_steps.append(t + twiddle)
                    break

        if not unconditional:
            if (
                len(condition.shape) == 2
            ):  # Expand rank 2 tensors by adding a batch dimension.
                condition = condition.unsqueeze(0)
            audio = torch.randn(
                condition.shape[0], scale * condition.shape[-1], device=device,
            )
        else:
            audio = torch.randn(1, scale, device=device)
        # noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

        for n in range(len(inference_alphas) - 1, -1, -1):
            c1 = 1 / inference_alphas[n] ** 0.5
            c2 = (
                inference_noise_schedule[n]
                / (1 - inference_alpha_cum[n]) ** 0.5
            )
            # predict noise
            noise_pred = self.model(
                audio,
                torch.tensor([inference_steps[n]], device=device),
                condition,
            ).squeeze(1)
            # mean
            audio = c1 * (audio - c2 * noise_pred)
            # add variance
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = (
                    (1.0 - inference_alpha_cum[n - 1])
                    / (1.0 - inference_alpha_cum[n])
                    * inference_noise_schedule[n]
                ) ** 0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)
        return audio


class LatentDiffusion(nn.Module):
    """A latent diffusion wrapper. Latent diffusion is denoising diffusion
    applied to a latent space instead of the original data space

    Arguments
    ---------
    autoencoder: speechbrain.nnet.autoencoder.Autoencoder
        An autoencoder converting the original space to a latent space

    diffusion: speechbrian.nnet.diffusion.Diffuser
        A diffusion wrapper

    latent_downsample_factor: int
        The factor that latent space dimensions need to be divisible
        by. This is useful if the underlying model for the diffusion
        wrapper is based on a UNet-like architecture where the inputs
        are progressively downsampled and upsampled by factors of two
    
    latent_pad_dims: int|list[int]
        the dimension(s) along which the latent space will be
        padded
    """

    def __init__(
        self,
        autoencoder,
        diffusion,
        latent_downsample_factor=None,
        latent_pad_dim=1
    ):
        super().__init__()
        self.autencoder = autoencoder
        self.diffusion = diffusion
        self.latent_downsample_factor = latent_downsample_factor
        if isinstance(latent_pad_dim, int):
            latent_pad_dim = [latent_pad_dim]
        self.latent_pad_dim = latent_pad_dim

    def train_sample(self, x, **kwargs):
        """Creates a sample for the training loop with a
        corresponding target

        Arguments
        ---------
        x: torch.Tensor
            the original data sample
        timesteps: torch.Tensor
            a 1-D integer tensor of a length equal to the number of
            batches in x, where each entry corresponds to the timestep
            number for the batch. If omitted, timesteps will be randomly
            sampled

        Returns
        -------
        pred: torch.Tensor
            the model output 0 prdicted noise
        noise: torch.Tensor
            the noise being applied
        noisy_sample
            the sample with the noise applied
        """

        latent = self.autoencoder.encode(x)
        latent = self._pad_latent(latent)
        return self.diffusion.train_sample(latent, **kwargs)

    def _pad_latent(self, latent):
        """Pads the latent space to the desired dimension
        
        Arguments
        ---------
        latent: torch.Tensor
            the latent representation
            
        Returns
        -------
        result: torch.Tensor
            the latent representation, with padding"""
        
        # TODO: Check whether masking will need to be adjusted
        if (
            self.latent_downsample_factor is not None
            and self.latent_downsample_factor > 1
        ):
            for dim in self.latent_pad_dim:
                latent, _ = data_utils.pad_divisible(
                    latent,
                    factor=self.latent_downsample_factor,
                    len_dim=dim
                )
        return latent

    def train_sample_latent(self, x, **kwargs):
        """Returns a train sample with autoencoder output - can be used to jointly
        training the diffusion model and the autoencoder

        Arguments
        ---------
        x: torch.Tensor
            the original data sample
        """
        # TODO: Make this generic
        length = kwargs.get("length")
        out_mask_value = kwargs.get("out_mask_value")
        latent_mask_value = kwargs.get("latent_mask_value")
        autoencoder_out = self.autencoder.train_sample(
            x,
            length=length,
            out_mask_value=out_mask_value,
            latent_mask_value=latent_mask_value,
        )
        latent = self._pad_latent(autoencoder_out.latent)
        diffusion_train_sample = self.diffusion.train_sample(
            latent, **kwargs
        )
        return LatentDiffusionTrainSample(
            diffusion=diffusion_train_sample, autoencoder=autoencoder_out
        )

    def distort(self, x):
        """Adds noise to the sample, in a forward diffusion process,

        Arguments
        ---------
        x: torch.Tensor
            a data sample of 2 or more dimensions, with the
            first dimension representing the batch
        noise: torch.Tensor
            the noise to add
        timesteps: torch.Tensor
            a 1-D integer tensor of a length equal to the number of
            batches in x, where each entry corresponds to the timestep
            number for the batch. If omitted, timesteps will be randomly
            sampled

        Returns
        -------
        result: torch.Tensor
            a tensor of the same dimension as x
        """

        latent = self.autencoder.encode(x)
        return self.diffusion.distort(latent)

    def sample(self, shape):
        # TODO: Auto-compute the latent shape
        latent = self.diffusion.sample(shape)
        latent = self._pad_latent(latent)
        return self.autencoder.decode(latent)


def sample_timesteps(x, num_timesteps):
    """Returns a random sample of timesteps as a 1-D tensor
    (one dimension only)

    Arguments
    ---------
    x: torch.Tensor
        a tensor of samples of any dimension
    num_timesteps: int
        the total number of timesteps"""
    return torch.randint(num_timesteps, (x.size(0),), device=x.device)


class GaussianNoise(nn.Module):
    """Adds ordinary Gaussian noise"""

    def forward(self, sample, **kwargs):
        """Forward pass

        Arguments
        ---------
        sample: the original sample
        """
        return torch.randn_like(sample)


class LengthMaskedGaussianNoise(nn.Module):
    """Gaussian noise applied to padded samples. No
    noise is added to positions that are part of padding

    Arguments
    ---------
    length_dim: int
        the
    """

    def __init__(self, length_dim=2):
        super().__init__()
        self.length_dim = length_dim

    def forward(self, sample, length=None, **kwargs):
        """Creates Gaussian noise. If a tensor of lengths is
        provided, no noise is added to the padding positions.

        sample: torch.Tensor
            a batch of data
        lens: torch.Tensor
            relative lengths
        """
        noise = torch.randn_like(sample)
        if length is not None:
            max_len = sample.size(self.length_dim)
            mask = length_to_mask(length * max_len, max_len).bool()
            mask_shape = self._compute_mask_shape(noise, max_len)
            mask = mask.view(mask_shape)
            noise.masked_fill_(~mask, 0.0)
        return noise

    def _compute_mask_shape(self, noise, max_len):
        return (
            (noise.shape[0],)
            + ((1,) * (self.length_dim - 1))  # Between the batch and len_dim
            + (max_len,)
            + ((1,) * (noise.dim() - 3))  # Unsqueeze at the end
        )


_NOISE_FUNCTIONS = {
    "gaussian": GaussianNoise(),
}

DiffusionTrainSample = namedtuple(
    "DiffusionTrainSample", ["pred", "noise", "noisy_sample"]
)
LatentDiffusionTrainSample = namedtuple(
    "LatentDiffusionTrainSample", ["diffusion", "autoencoder"]
)
