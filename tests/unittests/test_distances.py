import math

import torch


def test_cosine_similarity_matrix_simple(device):
    from speechbrain.utils.distances import cosine_similarity_matrix

    a = torch.tensor([[1.0, 0.5], [0.0, 0.5]], device=device)
    b = torch.tensor([[-1.0, -0.5], [1.0, 0.0]], device=device)

    sim = cosine_similarity_matrix(a, b)

    assert list(sim.shape) == [2, 2]

    assert math.isclose(sim[0, 0].cpu().item(), -1.0, abs_tol=1e-3)
    assert math.isclose(sim[0, 1].cpu().item(), 0.8944, abs_tol=1e-3)
    assert math.isclose(sim[1, 0].cpu().item(), -0.4472, abs_tol=1e-3)
    assert math.isclose(sim[1, 1].cpu().item(), 0.0, abs_tol=1e-3)


def test_cosine_similarity_matrix_batched(device):
    from speechbrain.utils.distances import cosine_similarity_matrix

    a = torch.tensor([[1.0, 0.5], [0.0, 0.5]], device=device)
    b = torch.tensor([[-1.0, -0.5], [1.0, 0.0]], device=device)

    a = a.unsqueeze(0).repeat(4, 1, 1)
    b = b.unsqueeze(0).repeat(4, 1, 1)
    a[0, ...] = 0.0  # zero out some batch to see if it affects others
    b[0, ...] = 0.0

    sim = cosine_similarity_matrix(a, b)

    assert list(sim.shape) == [4, 2, 2]

    assert math.isclose(sim[1, 0, 0].cpu().item(), -1.0, abs_tol=1e-3)
    assert math.isclose(sim[1, 0, 1].cpu().item(), 0.8944, abs_tol=1e-3)
    assert math.isclose(sim[1, 1, 0].cpu().item(), -0.4472, abs_tol=1e-3)
    assert math.isclose(sim[1, 1, 1].cpu().item(), 0.0, abs_tol=1e-3)
