import torch


def test_deltas(device):

    from speechbrain.processing.features import Deltas

    size = torch.Size([10, 101, 20], device=device)
    inp = torch.ones(size, device=device)
    compute_deltas = Deltas(input_size=20).to(device)
    out = torch.zeros(size, device=device)
    assert torch.sum(compute_deltas(inp) == out) == out.numel()

    assert torch.jit.trace(compute_deltas, inp)


def test_context_window(device):

    from speechbrain.processing.features import ContextWindow

    inp = (
        torch.tensor([1, 2, 3], device=device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .float()
    )
    compute_cw = ContextWindow(left_frames=1, right_frames=1).to(device)
    out = (
        torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 0]], device=device)
        .unsqueeze(0)
        .float()
    )
    assert torch.sum(compute_cw(inp) == out) == 9

    inp = torch.rand([2, 10, 5], device=device)
    compute_cw = ContextWindow(left_frames=0, right_frames=0).to(device)
    assert torch.sum(compute_cw(inp) == inp) == inp.numel()

    assert torch.jit.trace(compute_cw, inp)


def test_istft(device):
    from speechbrain.processing.features import STFT
    from speechbrain.processing.features import ISTFT

    fs = 16000
    inp = torch.randn([10, 16000], device=device)
    inp = torch.stack(3 * [inp], -1)

    compute_stft = STFT(sample_rate=fs).to(device)
    compute_istft = ISTFT(sample_rate=fs).to(device)
    out = compute_istft(compute_stft(inp), sig_length=16000)

    assert torch.sum(torch.abs(inp - out) < 5e-5) >= inp.numel() - 5

    assert torch.jit.trace(compute_stft, inp)
    assert torch.jit.trace(compute_istft, compute_stft(inp))


def test_filterbank(device):

    from speechbrain.processing.features import Filterbank

    compute_fbanks = Filterbank().to(device)
    inputs = torch.ones([10, 101, 201], device=device)
    assert torch.jit.trace(compute_fbanks, inputs)

    # Check amin (-100 dB)
    inputs = torch.zeros([10, 101, 201], device=device)
    fbanks = compute_fbanks(inputs)
    assert torch.equal(fbanks, torch.ones_like(fbanks) * -100)

    # Check top_db
    fbanks = torch.zeros([1, 1, 1], device=device)
    expected = torch.Tensor([[[-100]]]).to(device)
    fbanks_db = compute_fbanks._amplitude_to_DB(fbanks)
    assert torch.equal(fbanks_db, expected)

    # Making sure independent computation gives same results
    # as the batch computation
    input1 = torch.rand([1, 101, 201], device=device) * 10
    input2 = torch.rand([1, 101, 201], device=device)
    input3 = torch.cat([input1, input2], dim=0)
    fbank1 = compute_fbanks(input1)
    fbank2 = compute_fbanks(input2)
    fbank3 = compute_fbanks(input3)
    assert torch.sum(torch.abs(fbank1[0] - fbank3[0])) < 8e-05
    assert torch.sum(torch.abs(fbank2[0] - fbank3[1])) < 8e-05


def test_dtc(device):

    from speechbrain.processing.features import DCT

    compute_dct = DCT(input_size=40)
    inputs = torch.randn([10, 101, 40], device=device)
    assert torch.jit.trace(compute_dct, inputs)


def test_input_normalization(device):

    from speechbrain.processing.features import InputNormalization

    norm = InputNormalization().to(device)
    inputs = torch.randn([10, 101, 20], device=device)
    inp_len = torch.ones([10], device=device)
    assert torch.jit.trace(norm, (inputs, inp_len))

    norm = InputNormalization().to(device)
    inputs = (
        torch.FloatTensor([1, 2, 3, 0, 0, 0])
        .to(device)
        .unsqueeze(0)
        .unsqueeze(2)
    )
    inp_len = torch.FloatTensor([0.5]).to(device)
    out_norm = norm(inputs, inp_len).squeeze()
    target = torch.FloatTensor([-1, 0, 1, -2, -2, -2]).to(device)
    assert torch.equal(out_norm, target)


def test_features_multimic(device):

    from speechbrain.processing.features import Filterbank

    compute_fbanks = Filterbank().to(device)
    inputs = torch.rand([10, 101, 201], device=device)
    output = compute_fbanks(inputs)
    inputs_ch2 = torch.stack((inputs, inputs), -1)
    output_ch2 = compute_fbanks(inputs_ch2)
    output_ch2 = output_ch2[..., 0]
    assert torch.sum(output - output_ch2) < 5e-05
