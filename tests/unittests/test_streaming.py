import torch

def test_streaming_feature_wrapper(device):
    from speechbrain.lobes.features import StreamingFeatureWrapper
    from speechbrain.utils.filter_analysis import FilterProperties
    from speechbrain.utils.streaming import split_fixed_chunks

    class DummySumModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x[:, 2:-2]

    m = StreamingFeatureWrapper(
        DummySumModule(),
        FilterProperties(window_size=5, stride=2)
    )

    TEST_CHUNK_SIZE = 4
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], device=device)

    chunks = split_fixed_chunks(x, TEST_CHUNK_SIZE)
    assert len(chunks) == 2

    ctx = m.make_streaming_context()
    outs = [m(chunk, ctx) for chunk in chunks]

    print(outs)