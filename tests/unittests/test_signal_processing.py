import torch


# WORK IN PROGRESS
def test_gccphat():

    from speechbrain.processing.features import STFT
    from speechbrain.processing.signal_processing import GCCPHAT

    fs = 16000
    inp = torch.randn([2, 16000])
    inp = torch.stack(3 * [inp], -1)

    compute_stft = STFT(sample_rate=fs)
    compute_gccphat = GCCPHAT()

    xs = compute_stft(inp)
    compute_gccphat(xs)

    # TODO: Change condition and complete test
    assert torch.sum(xs == xs) == xs.numel()
