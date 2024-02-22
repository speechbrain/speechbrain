import torch


def test_streaming_feature_wrapper(device):
    from speechbrain.lobes.features import StreamingFeatureWrapper
    from speechbrain.utils.filter_analysis import FilterProperties
    from speechbrain.utils.streaming import split_fixed_chunks

    # dummy filter that lies about its properties
    class DummySumModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x

    props = FilterProperties(window_size=5, stride=2)

    m = StreamingFeatureWrapper(DummySumModule(), props)

    chunk_size = 3
    chunk_size_frames = (props.stride - 1) * chunk_size

    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]], device=device
    )

    chunks = split_fixed_chunks(x, chunk_size_frames)
    assert len(chunks) == 3

    ctx = m.make_streaming_context()
    outs = [m(chunk, ctx) for chunk in chunks]

    # the streaming feature wrapper will truncate output module frames that are
    # centered on "padding" frames (as described in the code)

    # currently, the expected output is as follows:
    assert torch.allclose(outs[0], torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    assert torch.allclose(outs[1], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert torch.allclose(outs[2], torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0]))

    # thus we have outputs centered on:
    # [0, 0, 2]
    # [1, 3, 5]
    # [4, 6, 8]
    # which preserves the filter properties as expected, and the chunk size we
    # requested.
