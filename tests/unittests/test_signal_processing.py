import torch


def test_gccphat():

    from speechbrain.processing.features import STFT
    import speechbrain.processing.signal_processing as sp

    # Creating the test signal
    fs = 16000

    delay = 120

    sig = torch.randn([10, fs])
    sig_delayed = torch.cat((torch.zeros([10, delay]), sig[:, 0:-delay]), 1)

    xs = torch.stack((sig_delayed, sig), -1)

    compute_stft = STFT(sample_rate=fs)
    xs = compute_stft(xs)

    # Computing the covariance matrix for GCC-PHAT
    rxx = sp.cov(xs)

    gccphat = sp.GCCPHAT()
    xxs = gccphat(rxx)

    # Extracting every delay found by GCC-PHAT
    gccphat_delays = gccphat.find_tdoa(xxs, tdoa_max=125)

    assert torch.sum(gccphat_delays[3, :, 1] == delay) == xs.shape[1]
