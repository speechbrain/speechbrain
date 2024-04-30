"""Distance metrics and related functions"""

import torch


def cosine_similarity_matrix(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    """Computes a matrix evaluating all pairwise cosine similarities.
    The cosine similarity can otherwise be determined with
    :class:`torch.nn.CosineSimilarity`.

    Arguments
    ---------
    a : torch.Tensor
        Tensor of shape `[..., X, dim]` where `dim` is the dimension where the
        cosine similarity will be computed and `X` is any value `>= 0`.
    b : torch.Tensor
        Tensor of shape `[..., Y, dim]`, where other dimensions are otherwise
        identical to `a`'s and `Y` is any value `>= 0`.
    eps : float
        Epsilon value for numerical stability, in order to avoid a division by
        zero. Does not significantly affect results.

    Returns
    -------
    torch.Tensor
        Tensor of shape `[..., X, Y]` living on the same device and dtype as the
        input tensors. e.g. ignoring first dimensions `out[3, 0]` would be the
        cosine similarity of `a[3]` and `b[0]`.
    """

    assert a.dim() == b.dim(), "Inputs must be of the same dim"
    assert a.dim() >= 2, "Expected at least 2 dims [X, cos_sim_dim]"
    assert (
        a.shape[:-2] == b.shape[:-2]
    ), "Input shape must match until last 2 dims"

    a_norm = torch.linalg.vector_norm(a, dim=-1).unsqueeze(-1)  # [..., X, 1]
    b_norm = torch.linalg.vector_norm(b, dim=-1).unsqueeze(-1)  # [..., Y, 1]

    # dim -1 of *_norm gets broadcasted
    a_normalized = a / torch.clamp(a_norm, min=eps)
    b_normalized = b / torch.clamp(b_norm, min=eps)

    # here the matrix multiply effectively results, for [..., x, y], in the dot
    # product of the normalized `a[..., x, :]` and `b[..., y, :]` vectors, thus
    # giving us the proper cosine similarity.
    # multiplication shape: a[..., X, 1] @ b[..., 1, Y]
    return a_normalized @ b_normalized.transpose(-1, -2)
