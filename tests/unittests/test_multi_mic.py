import torch


def test_gccphat(device):

    from speechbrain.processing.features import STFT
    from speechbrain.processing.multi_mic import Covariance, GccPhat

    # Creating the test signal
    fs = 16000

    delay = 60

    sig = torch.randn([10, fs], device=device)
    sig_delayed = torch.cat(
        (torch.zeros([10, delay], device=device), sig[:, 0:-delay]), 1
    )

    xs = torch.stack((sig_delayed, sig), -1)

    stft = STFT(sample_rate=fs).to(device)
    Xs = stft(xs)

    # Computing the covariance matrix for GCC-PHAT
    cov = Covariance().to(device)
    gccphat = GccPhat().to(device)

    XXs = cov(Xs).to(device)
    tdoas = torch.abs(gccphat(XXs))

    n_valid_tdoas = torch.sum(torch.abs(tdoas[..., 1] - delay) < 1e-3)
    assert n_valid_tdoas == Xs.shape[0] * Xs.shape[1]
    assert torch.jit.trace(stft, xs)
    assert torch.jit.trace(cov, Xs)
    assert torch.jit.trace(gccphat, XXs)
