import torch


def test_deltas():

    from speechbrain.processing.features import Deltas

    size = torch.Size([10, 101, 20])
    inp = torch.ones(size)
    compute_deltas = Deltas()
    compute_deltas.init_params(inp)
    out = torch.zeros(size)
    assert torch.sum(compute_deltas(inp) == out) == out.numel()


def test_context_window():

    from speechbrain.processing.features import ContextWindow

    inp = torch.tensor([1, 2, 3]).unsqueeze(0).unsqueeze(-1).float()
    compute_cw = ContextWindow(left_frames=1, right_frames=1)
    out = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 0]]).unsqueeze(0).float()
    assert torch.sum(compute_cw(inp) == out) == 9

    inp = torch.rand([2, 10, 5])
    compute_cw = ContextWindow(left_frames=0, right_frames=0)
    assert torch.sum(compute_cw(inp) == inp) == inp.numel()
