# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch

from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry


__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_pc_sampler(
    predictor_name, corrector_name, sde, score_fn, y,
    denoise=True, eps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False,
    intermediate=False, **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps)

    def pc_sampler():
        """The PC sampler function."""
        with torch.no_grad():
            xt = sde.prior_sampling(y.shape, y).to(y.device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
            for i in range(sde.N):
                t = timesteps[i]
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i+1]
                else:
                    stepsize = timesteps[-1] # from eps to 0
                vec_t = torch.ones(y.shape[0], device=y.device) * t
                xt, xt_mean = corrector.update_fn(xt, y, vec_t)
                xt, xt_mean = predictor.update_fn(xt, y, vec_t, stepsize)
            x_result = xt_mean if denoise else xt
            ns = sde.N * (corrector.n_steps + 1)
            return x_result, ns
    
    return pc_sampler


def get_ode_sampler(
    sde, score_fn, y, inverse_scaler=None,
    denoise=True, rtol=1e-5, atol=1e-5,
    method='RK45', eps=3e-2, device='cuda', **kwargs
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    rsde = sde.reverse(score_fn, probability_flow=True)

    def denoise_update_fn(x):
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor.update_fn(x, y, vec_eps)
        return x

    def drift_fn(x, y, t):
        """Get the drift function of the reverse-time SDE."""
        return rsde.sde(x, y, t)[0]

    def ode_sampler(z=None, **kwargs):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # If not represent, sample the latent code from the prior distibution of the SDE.
            x = sde.prior_sampling(y.shape, y).to(device)

            def ode_func(t, x):
                x = from_flattened_numpy(x, y.shape).to(device).type(torch.complex64)
                vec_t = torch.ones(y.shape[0], device=x.device) * t
                drift = drift_fn(x, y, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (sde.T, eps), to_flattened_numpy(x),
                rtol=rtol, atol=atol, method=method, **kwargs
            )
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(y.shape).to(device).type(torch.complex64)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(x)

            if inverse_scaler is not None:
                x = inverse_scaler(x)
            return x, nfe

    return ode_sampler

def get_sb_sampler(sde, model, y, eps=1e-4, n_steps=50, sampler_type="ode", **kwargs):
    # adapted from https://github.com/NVIDIA/NeMo/blob/78357ae99ff2cf9f179f53fbcb02c88a5a67defb/nemo/collections/audio/parts/submodules/schroedinger_bridge.py#L382
    def sde_sampler():
        """The SB-SDE sampler function."""
        with torch.no_grad():
            xt = y[:, [0], :, :] # special case for storm_2ch
            time_steps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)

            # Initial values
            time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = sde._sigmas_alphas(time_prev)

            for t in time_steps[1:]:
                # Prepare time steps for the whole batch
                time = t * torch.ones(xt.shape[0], device=xt.device)

                # Get noise schedule for current time
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = sde._sigmas_alphas(time)

                # Run DNN
                current_estimate = model(xt, y, time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t**2 / (alpha_prev * sigma_prev**2 + sde.eps)
                tmp = 1 - sigma_t**2 / (sigma_prev**2 + sde.eps)
                weight_estimate = alpha_t * tmp
                weight_z = alpha_t * sigma_t * torch.sqrt(tmp)

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_z = weight_z[:, None, None, None]

                # Random sample
                z_norm = torch.randn_like(xt)
                
                if t == time_steps[-1]:
                    weight_z = 0.0

                # Update state: weighted sum of previous state, current estimate and noise
                xt = weight_prev * xt + weight_estimate * current_estimate + weight_z * z_norm

                # Save previous values
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt, n_steps

    def ode_sampler():
        """The SB-ODE sampler function."""
        with torch.no_grad():
            xt = y
            time_steps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)

            # Initial values
            time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = sde._sigmas_alphas(time_prev)

            for t in time_steps[1:]:
                # Prepare time steps for the whole batch
                time = t * torch.ones(xt.shape[0], device=xt.device)

                # Get noise schedule for current time
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = sde._sigmas_alphas(time)

                # Run DNN
                current_estimate = model(xt, y, time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t * sigma_bart / (alpha_prev * sigma_prev * sigma_bar_prev + sde.eps)
                weight_estimate = (
                    alpha_t
                    / (sigma_T**2 + sde.eps)
                    * (sigma_bart**2 - sigma_bar_prev * sigma_t * sigma_bart / (sigma_prev + sde.eps))
                )
                weight_prior_mean = (
                    alpha_t
                    / (alpha_T * sigma_T**2 + sde.eps)
                    * (sigma_t**2 - sigma_prev * sigma_t * sigma_bart / (sigma_bar_prev + sde.eps))
                )

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_prior_mean = weight_prior_mean[:, None, None, None]

                # Update state: weighted sum of previous state, current estimate and prior
                xt = weight_prev * xt + weight_estimate * current_estimate + weight_prior_mean * y

                # Save previous values
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt, n_steps
    
    if sampler_type == "sde":
        return sde_sampler
    elif sampler_type == "ode":
        return ode_sampler
    else:
        raise ValueError("Invalid type. Choose 'ode' or 'sde'.")
