import torch


def test_gccphat():

    from speechbrain.processing.features import STFT
    from speechbrain.processing.multi_mic import Covariance, GccPhat

    # Creating the test signal
    fs = 16000

    delay = 120

    sig = torch.randn([10, fs])
    sig_delayed = torch.cat((torch.zeros([10, delay]), sig[:, 0:-delay]), 1)

    xs = torch.stack((sig_delayed, sig), -1)

    stft = STFT(sample_rate=fs)
    Xs = stft(xs)

    # Computing the covariance matrix for GCC-PHAT
    cov = Covariance()
    gccphat = GccPhat()

    XXs = cov(Xs)
    tdoas = gccphat(XXs)

    n_valid_tdoas = torch.sum(torch.abs(tdoas[..., 1] - delay) < 1e-3)
    assert n_valid_tdoas == Xs.shape[0] * Xs.shape[1]
