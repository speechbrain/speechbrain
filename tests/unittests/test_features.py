import torch


def test_deltas():

    from speechbrain.processing.features import Deltas

    size = torch.Size([10, 101, 20])
    inp = torch.ones(size)
    compute_deltas = Deltas(input_size=20)
    out = torch.zeros(size)
    assert torch.sum(compute_deltas(inp) == out) == out.numel()

    assert torch.jit.trace(compute_deltas, inp)


def test_context_window():

    from speechbrain.processing.features import ContextWindow

    inp = torch.tensor([1, 2, 3]).unsqueeze(0).unsqueeze(-1).float()
    compute_cw = ContextWindow(left_frames=1, right_frames=1)
    out = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 0]]).unsqueeze(0).float()
    assert torch.sum(compute_cw(inp) == out) == 9

    inp = torch.rand([2, 10, 5])
    compute_cw = ContextWindow(left_frames=0, right_frames=0)
    assert torch.sum(compute_cw(inp) == inp) == inp.numel()

    assert torch.jit.trace(compute_cw, inp)


def test_istft():
    from speechbrain.processing.features import STFT
    from speechbrain.processing.features import ISTFT

    fs = 16000
    inp = torch.randn([10, 16000])
    inp = torch.stack(3 * [inp], -1)

    compute_stft = STFT(sample_rate=fs)
    compute_istft = ISTFT(sample_rate=fs)
    out = compute_istft(compute_stft(inp), sig_length=16000)

    assert torch.sum(torch.abs(inp - out) < 1e-6) >= inp.numel() - 5

    assert torch.jit.trace(compute_stft, inp)
    assert torch.jit.trace(compute_istft, compute_stft(inp))


def test_filterbank():

    from speechbrain.processing.features import Filterbank

    compute_fbanks = Filterbank()
    inputs = torch.ones([10, 101, 201])
    assert torch.jit.trace(compute_fbanks, inputs)


def test_dtc():

    from speechbrain.processing.features import DCT

    compute_dct = DCT(input_size=40)
    inputs = torch.randn([10, 101, 40])
    assert torch.jit.trace(compute_dct, inputs)


def test_input_normalization():

    from speechbrain.processing.features import InputNormalization

    norm = InputNormalization()
    inputs = torch.randn([10, 101, 20])
    inp_len = torch.ones([10])
    assert torch.jit.trace(norm, (inputs, inp_len))

    norm = InputNormalization()
    inputs = torch.FloatTensor([1, 2, 3, 0, 0, 0]).unsqueeze(0).unsqueeze(2)
    inp_len = torch.FloatTensor([0.5])
    out_norm = norm(inputs, inp_len).squeeze()
    target = torch.FloatTensor([-1, 0, 1, -2, -2, -2])
    assert torch.equal(out_norm, target)


def test_features_multimic():

    from speechbrain.processing.features import Filterbank

    compute_fbanks = Filterbank()
    inputs = torch.rand([10, 101, 201])
    output = compute_fbanks(inputs)
    inputs_ch2 = torch.stack((inputs, inputs), -1)
    output_ch2 = compute_fbanks(inputs_ch2)
    output_ch2 = output_ch2[..., 0]
    assert torch.sum(output - output_ch2) < 1e-05
