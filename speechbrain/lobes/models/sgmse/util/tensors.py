def batch_broadcast(a, x):
    """Broadcasts a over all dimensions of x, except the batch dimension, which must match."""

    if len(a.shape) != 1:
        a = a.squeeze()
        if len(a.shape) != 1:
            raise ValueError(
                f"Don't know how to batch-broadcast tensor `a` with more than one effective dimension (shape {a.shape})"
            )

    if a.shape[0] != x.shape[0] and a.shape[0] != 1:
        raise ValueError(
            f"Don't know how to batch-broadcast shape {a.shape} over {x.shape} as the batch dimension is not matching")

    out = a.view((x.shape[0], *(1 for _ in range(len(x.shape)-1))))
    return out
