import torch


def test_lengths_capable_sequential_lengths_dispatch():
    from speechbrain.nnet.containers import LengthsCapableSequential

    class LengthsModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received = None

        def forward(self, x, lengths=None):
            self.received = lengths
            return x

    class WavLensModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received = None

        def forward(self, x, wav_lens=None):
            self.received = wav_lens
            return x

    class PlainModule(torch.nn.Module):
        def forward(self, x):
            return x

    lengths_layer = LengthsModule()
    wav_lens_layer = WavLensModule()
    model = LengthsCapableSequential(
        lengths_layer, PlainModule(), wav_lens_layer
    )

    x = torch.randn(2, 4)
    lengths = torch.tensor([0.5, 1.0])
    model(x, lengths=lengths)

    assert lengths_layer.received is lengths
    assert wav_lens_layer.received is lengths
