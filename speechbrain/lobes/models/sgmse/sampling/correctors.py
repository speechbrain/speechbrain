import abc
import torch

from ..util.registry import Registry

CorrectorRegistry = Registry("Corrector")


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, y, t, *args):
        """One update of the corrector.

        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@CorrectorRegistry.register(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        self.score_fn = score_fn
        self.n_steps = n_steps
        self.snr = snr

    def update_fn(self, x, y, t, *args):
        target_snr = self.snr
        for _ in range(self.n_steps):
            grad = self.score_fn(x, y, t, *args)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = ((target_snr * noise_norm / grad_norm) ** 2 * 2).unsqueeze(0)
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@CorrectorRegistry.register(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    def update_fn(self, x, y, t, *args):
        n_steps = self.n_steps
        target_snr = self.snr
        std = self.sde.marginal_prob(x, y, t, *args)[1]

        for _ in range(n_steps):
            grad = self.score_fn(x, y, t, *args)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@CorrectorRegistry.register(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, *args, **kwargs):
        self.snr = 0
        self.n_steps = 0
        pass

    def update_fn(self, x, t, *args):
        return x, x