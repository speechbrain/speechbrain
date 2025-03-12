"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import warnings

import numpy as np
from .util.tensors import batch_broadcast
import torch

from .util.registry import Registry


SDERegistry = Registry("SDE")


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, y, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, y, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, y, t, stepsize):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = stepsize
        drift, diffusion = self.sde(x, y, t)
        f = drift * dt
        G = diffusion * torch.sqrt(dt)
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, y, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, y, t, *args)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                return total_drift, diffusion

            def rsde_parts(self, x, y, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, y, t, *args)
                score = score_model(x, y, t, *args)
                score_drift = -sde_diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift + score_drift
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score,
                }

            def discretize(self, x, y, t, stepsize):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, y, t, stepsize)
                rev_f = f - G[:, None, None, None] ** 2 * score_model(x, y, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass


@SDERegistry.register("ouve")
class OUVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--theta", type=float, default=1.5, help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.5 by default.")
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--sigma-max", type=float, default=0.5, help="The maximum sigma to use. 0.5 by default.")
        parser.add_argument("--N", type=int, default=30, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--sampler_type", type=str, default="pc", help="Type of sampler to use. 'pc' by default.")
        return parser

    def __init__(self, theta, sigma_min, sigma_max, N=30, sampler_type="pc", **ignored_kwargs):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = -theta (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N
        self.sampler_type = sampler_type

    def copy(self):
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max, N=self.N, sampler_type=self.sampler_type)

    @property
    def T(self):
        return 1

    def sde(self, x, y, t):
        drift = self.theta * (y - x)
        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, y, t):
        theta = self.theta
        exp_interp = torch.exp(-theta * t)[:, None, None, None]
        return exp_interp * x0 + (1 - exp_interp) * y

    def alpha(self, t):
        return torch.exp(-self.theta * t)

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            /
            (theta + logsig)
        )

    def marginal_prob(self, x0, y, t):
        return self._mean(x0, y, t), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        x_T = y + torch.randn_like(y) * std[:, None, None, None]
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")


@SDERegistry.register("sbve")
class SBVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--N", type=int, default=50, help="The number of timesteps in the SDE discretization. 50 by default")
        parser.add_argument("--k", type=float, default=2.6, help="Parameter of the diffusion coefficient. 2.6 by default.")
        parser.add_argument("--c", type=float, default=0.4, help="Parameter of the diffusion coefficient. 0.4 by default.")
        parser.add_argument("--eps", type=float, default=1e-8, help="Small constant to avoid numerical instability. 1e-8 by default.")
        parser.add_argument("--sampler_type", type=str, default="ode")
        return parser

    def __init__(self, k, c, N=50, eps=1e-8, sampler_type="ode", **ignored_kwargs):
        """Construct a Schrodinger Bridge with Variance Exploding SDE.

        As described in Jukić et al., „Schrödinger Bridge for Generative Speech Enhancement“, 2024.

        Args:
            k: stiffness parameter.
            c: diffusion parameter.
            N: number of discretization steps
        """
        super().__init__(N)
        self.k = k
        self.c = c
        self.N = N
        self.eps = eps
        self.sampler_type = sampler_type

    def copy(self):
        return SBVESDE(self.k, self.c, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, y, t):
        f = 0.0                                                                 # Table 1
        g = torch.sqrt(torch.tensor(self.c)) * self.k**(t)                      # Table 1
        return f, g

    def _sigmas_alphas(self, t):
        alpha_t = torch.ones_like(t)
        alpha_T = torch.ones_like(t)
        sigma_t = torch.sqrt((self.c*(self.k**(2*t)-1.0)) \
            / (2*torch.log(torch.tensor(self.k))))                              # Table 1
        sigma_T = torch.sqrt((self.c*(self.k**(2*self.T)-1.0)) \
            / (2*torch.log(torch.tensor(self.k))))                              # Table 1   
        
        alpha_bart = alpha_t / (alpha_T + self.eps)                             # below Eq. (9)
        sigma_bart = torch.sqrt(sigma_T**2 - sigma_t**2 + self.eps)             # below Eq. (9)

        return sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart

    def _mean(self, x0, y, t):
        sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)

        w_xt = alpha_t * sigma_bart**2 / (sigma_T**2 + self.eps)                # below Eq. (11)
        w_yt = alpha_bart * sigma_t**2 / (sigma_T**2 + self.eps)                # below Eq. (11)

        mu = w_xt[:, None, None, None] * x0 + w_yt[:, None, None, None] * y     # Eq. (11)
        return mu

    def _std(self, t):
        sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)

        sigma_xt = (alpha_t * sigma_bart * sigma_t) / (sigma_T + self.eps) 
        return sigma_xt

    def marginal_prob(self, x0, y, t):  
        return self._mean(x0, y, t), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        x_T = y
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for SBVE SDE not yet implemented!")  