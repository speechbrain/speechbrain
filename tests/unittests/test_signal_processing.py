import torch


def test_gccphat():

    from speechbrain.processing.features import STFT
    import speechbrain.processing.signal_processing as sp

    # Creating the test signal
    fs = 16000

    padding = 120
    n_fft = 400
    delay = n_fft - padding

    sig = torch.randn([10, fs])
    sig_delayed = torch.cat((torch.zeros([10, padding]), sig[:, 0:-padding]), 1)

    xs = torch.stack((sig, sig_delayed), -1)

    compute_stft = STFT(sample_rate=fs)
    xs = compute_stft(xs)

    # Computing the covariance matrix for GCC-PHAT
    rxx = sp.cov(xs)

    compute_gccphat = sp.GCCPHAT()
    xxs = compute_gccphat(rxx)

    # Extracting every delay found by GCC-PHAT
    _, gccphat_delays = torch.max(xxs[3, :, :, 1], 1)

    assert torch.sum(gccphat_delays == delay) == xs.shape[1]
