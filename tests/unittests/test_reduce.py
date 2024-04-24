import os

import torch
import torch.multiprocessing as mp
import torch.nn

import speechbrain as sb


def test_reduce(device):
    tensor = torch.tensor([1, 2], device=device)

    out_tensor = sb.utils.distributed.reduce(tensor, reduction="sum")
    assert (out_tensor == torch.Tensor([1, 2])).all()

    out_tensor = sb.utils.distributed.reduce(tensor, reduction="mean")
    assert (out_tensor == torch.Tensor([1, 2])).all()


def _test_reduce_ddp(rank, size, backend="gloo"):  # noqa
    """Initialize the distributed environment."""
    os.environ["WORLD_SIZE"] = f"{size}"
    os.environ["RANK"] = f"{rank}"
    os.environ["LOCAL_RANK"] = f"{rank}"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    run_opts = dict()
    run_opts["distributed_launch"] = True
    run_opts["distributed_backend"] = backend

    sb.utils.distributed.ddp_init_group(run_opts)

    # test reduce
    tensor = torch.arange(2) + 1 + 2 * rank
    out_tensor = sb.utils.distributed.reduce(tensor.float(), reduction="sum")
    assert (out_tensor == torch.Tensor([4, 6])).all()

    out_tensor = sb.utils.distributed.reduce(tensor.float(), reduction="mean")
    assert (out_tensor == torch.Tensor([2, 3])).all()

    obj = [{"a": [(torch.arange(2) + 1 + 2 * rank).float() for _ in range(4)]}]
    out_obj = sb.utils.distributed.reduce(obj, reduction="sum")
    for i in range(4):
        assert (out_obj[0]["a"][i] == torch.Tensor([4, 6])).all()

    out_obj = sb.utils.distributed.reduce(obj, reduction="mean")
    for i in range(4):
        assert (out_obj[0]["a"][i] == torch.Tensor([2, 3])).all()


def test_ddp_reduce():
    size = 2
    processes = []

    mp.set_start_method("spawn", force=True)

    for rank in range(size):
        p = mp.Process(target=_test_reduce_ddp, args=(rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        assert p.exitcode == 0


if __name__ == "__main__":
    test_ddp_reduce()
