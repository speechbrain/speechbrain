import torch
from torch import nn


def test_denoising_diffusion_distort():
    from speechbrain.nnet.diffusion import DenoisingDiffusion

    dummy_model = nn.Linear(2, 2)
    diffusion = DenoisingDiffusion(model=dummy_model)

    x = torch.tensor([[1.0, 2.0], [4.0, 5.0], [3.0, 2.0]])

    # Fake, non-Gaussian noise
    noise = torch.tensor([[5.0, 5.0], [10.0, 10.0], [20.0, 20.0]])
    timesteps = [0, 200, 900]

    x_noisy, x_noise = diffusion.distort(x, noise=noise, timesteps=timesteps)
    assert x_noisy[0].allclose(torch.tensor([1.05, 2.05]), atol=0.01)
    assert x_noisy[1].allclose(torch.tensor([9.103, 9.913]), atol=0.01)
    assert x_noisy[2].allclose(torch.tensor([20.05, 20.03]), atol=0.01)
    assert x_noise.allclose(noise)

    torch.manual_seed(42)
    x = torch.ones(3, 2).float() * 10.0
    timesteps = torch.tensor([0, 0, 999])
    x_noisy = diffusion.distort(x, timesteps=timesteps)
    assert (x_noisy[0].mean() - 10.0) < 1.0
    assert (x_noisy[1].mean() - 10.0) < 1.0


class DummyModel(nn.Module):
    def forward(self, x, timesteps):
        return x * 1.001 + timesteps.unsqueeze(-1) * 0.00001


def test_denoising_diffusion_sample():
    from speechbrain.nnet.diffusion import DenoisingDiffusion

    dummy_model = DummyModel()
    gen = torch.manual_seed(42)

    def noise(x):
        return torch.randn(*x.shape, generator=gen)

    diffusion = DenoisingDiffusion(
        model=dummy_model,
        timesteps=1000,
        noise=noise,
        beta_start=0.0001,
        beta_end=0.02,
    )

    sample = diffusion.sample((3, 2))
    sample_ref = torch.tensor(
        [[-0.4607, -0.3638], [0.4681, 0.3358], [-0.1250, -0.2619]]
    )
    assert sample.allclose(sample_ref, atol=0.0001)
