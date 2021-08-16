import torch


def test_stft():

    import torchaudio
    from speechbrain.processing.features import STFT
    from torch.nn.utils.rnn import pad_sequence

    audio_file_1 = "samples/audio_samples/example1.wav"
    audio_file_2 = "samples/audio_samples/example2.flac"

    sig_1, fs = torchaudio.load(audio_file_1)

    # Select just a random chunk
    rand_end = torch.randint(size=(1,), low=1000, high=8000)
    sig_1 = sig_1[:, 0:rand_end]

    # Padding
    sig_pad = torch.zeros([1, 16000])
    sig_pad[:, 0:rand_end] = sig_1

    compute_stft = STFT(fs)
    out = compute_stft(sig_1)
    out_pad = compute_stft(sig_pad)

    wav_len = torch.Tensor([sig_1.shape[1] / sig_pad.shape[1]])
    out_pad = compute_stft(sig_pad, wav_len=wav_len)

    # Padded version must be equal to unpadded version
    assert (
        torch.sum(out == out_pad[:, 0 : out.shape[1]])
        == torch.Tensor(list(out.shape)).prod()
    )

    # All the other elements of the padded tensors must be zero
    assert torch.sum(out_pad[:, out.shape[1] :]) == 0

    # Let's see what happens within the batch
    sig_2, fs = torchaudio.load(audio_file_2)

    # Let's just select a part
    rand_end = torch.randint(size=(1,), low=1000, high=8000)
    sig_2 = sig_2[:, 0:rand_end]
    out2 = compute_stft(sig_2)

    # Batch creation
    batch = pad_sequence(
        [sig_1.squeeze(), sig_2.squeeze()], batch_first=True, padding_value=0.0
    )
    wav_len = torch.Tensor(
        [sig_1.shape[1] / batch.shape[1], sig_2.shape[1] / batch.shape[1]]
    )
    out_batch = compute_stft(batch, wav_len)

    # Making sure the STFT is the same even when the sentence is withing a batch
    assert (
        torch.sum(out == out_batch[0, 0 : out.shape[1]])
        == torch.Tensor(list(out.shape)).prod()
    )
    assert torch.sum(out_batch[0, out.shape[1] :]) == 0
    assert (
        torch.sum(out2 == out_batch[1, 0 : out2.shape[1]])
        == torch.Tensor(list(out2.shape)).prod()
    )
    assert torch.sum(out_batch[1, out2.shape[1] :]) == 0


def test_deltas():

    from speechbrain.processing.features import Deltas

    size = torch.Size([10, 101, 20])
    inp = torch.ones(size)
    compute_deltas = Deltas(input_size=20)
    out = torch.zeros(size)
    assert (
        torch.sum(compute_deltas(inp)[:, 2:-2, :] == out[:, 2:-2, :])
        == out[:, 2:-2, :].numel()
    )

    # Check batch vs non-batch computations
    input1 = torch.rand([1, 101, 40]) * 10
    input2 = torch.rand([1, 101, 40])
    input3 = torch.cat([input1, input2], dim=0)

    compute_deltas = Deltas(input_size=input1.size(-1))

    fea1 = compute_deltas(input1)
    fea2 = compute_deltas(input2)
    fea3 = compute_deltas(input3)

    assert (fea1[0] - fea3[0]).abs().sum() == 0
    assert (fea2[0] - fea3[1]).abs().sum() == 0

    rand_end = torch.randint(size=(1,), low=15, high=80)

    input1 = input1[:, 0:rand_end]
    input3[0, rand_end:] = 0

    wav_len = torch.Tensor([rand_end / input3.shape[1], 1.0])
    fea1 = compute_deltas(input1)
    fea3 = compute_deltas(input3, wav_len)

    assert (fea1[0, :, :] - fea3[0, 0 : fea1.shape[1], :]).abs().sum() == 0
    assert (fea3[0, fea1.shape[1] :, :]).sum() == 0
    assert (fea2[0, :, :] - fea3[1, 0 : fea2.shape[1], :]).abs().sum() == 0

    # JIT check
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

    import torchaudio
    from speechbrain.processing.features import Filterbank
    from speechbrain.processing.features import STFT, spectral_magnitude
    from torch.nn.utils.rnn import pad_sequence

    torch.use_deterministic_algorithms(True)

    # A tollerance is needed because sometimes a non-deterministic behavior of
    # torch.mm occurs when processing sentences of different lengths
    # (see https://colab.research.google.com/drive/1lklKrRRYTKTXwMbFMh62biQokjmA95is?usp=sharing).
    tollerance_th = 5e-07

    compute_fbanks = Filterbank()
    inputs = torch.ones([10, 101, 201])
    assert torch.jit.trace(compute_fbanks, inputs)

    # Check amin (-100 dB)
    inputs = torch.zeros([10, 101, 201])
    fbanks = compute_fbanks(inputs)
    assert torch.equal(fbanks, torch.ones_like(fbanks) * -100)

    # Check top_db
    fbanks = torch.zeros([1, 1, 1])
    expected = torch.Tensor([[[-100]]])
    fbanks_db = compute_fbanks._amplitude_to_DB(fbanks)
    assert torch.equal(fbanks_db, expected)

    # Making sure independent computation gives same results
    # as the batch computation
    input1 = torch.rand([1, 101, 201]) * 10
    input2 = torch.rand([1, 101, 201])
    input3 = torch.cat([input1, input2], dim=0)
    fbank1 = compute_fbanks(input1)
    fbank2 = compute_fbanks(input2)
    fbank3 = compute_fbanks(input3)
    assert torch.mean(torch.abs(fbank1[0] - fbank3[0])) <= tollerance_th
    assert torch.mean(torch.abs(fbank2[0] - fbank3[1])) <= tollerance_th

    # Tests using read signals (including STFT)
    audio_file_1 = "samples/audio_samples/example1.wav"
    audio_file_2 = "samples/audio_samples/example2.flac"

    sig_1, fs = torchaudio.load(audio_file_1)

    # Let's just select a part
    rand_end = torch.randint(size=(1,), low=3500, high=8000)
    sig_1 = sig_1[:, 0:rand_end]

    compute_stft = STFT(16000)
    compute_fbanks = Filterbank()

    out = compute_stft(sig_1)
    out = spectral_magnitude(out)
    out_1 = compute_fbanks(out)

    # let's not add padding
    sig_pad = torch.zeros([1, 16000])
    sig_pad[:, 0:rand_end] = sig_1

    wav_len = torch.Tensor([sig_1.shape[1] / sig_pad.shape[1]])

    out = compute_stft(sig_pad, wav_len)
    out = spectral_magnitude(out)
    out_1_pad = compute_fbanks(out)

    # Padded version must be equal to unpadded version
    assert (
        torch.mean(
            torch.abs(out_1[0, :, :] - out_1_pad[0, 0 : out_1.shape[1], :])
        )
        <= tollerance_th
    )

    # Let's see what happens within the batch
    sig_2, fs = torchaudio.load(audio_file_2)

    # Let's just select a part
    rand_end = torch.randint(size=(1,), low=3500, high=8000)
    sig_2 = sig_2[:, 0:rand_end]

    batch = pad_sequence(
        [sig_1.squeeze(), sig_2.squeeze()], batch_first=True, padding_value=0.0
    )

    wav_len = torch.Tensor(
        [sig_1.shape[1] / batch.shape[1], sig_2.shape[1] / batch.shape[1]]
    )

    out = compute_stft(batch, wav_len)
    out = spectral_magnitude(out)
    out_batch = compute_fbanks(out)

    # Padded version must be equal to unpadded version
    assert (
        torch.mean(
            torch.abs(out_1[0, :, :] - out_batch[0, 0 : out_1.shape[1], :])
        )
        <= tollerance_th
    )

    out = compute_stft(sig_2)
    out = spectral_magnitude(out)
    out_2 = compute_fbanks(out)

    assert (
        torch.mean(
            torch.abs(out_2[0, :, :] - out_batch[1, 0 : out_2.shape[1], :])
        )
        <= tollerance_th
    )


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
