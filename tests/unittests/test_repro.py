"""Unit tests for reproducibility utilities"""

import warnings

import torch


def test_repro(tmpdir):
    from speechbrain.utils.checkpoints import Checkpointer
    from speechbrain.utils.repro import SaveableGenerator

    gen1 = torch.Generator()
    gen2 = torch.Generator()
    gen = SaveableGenerator({"gen1": gen1, "gen2": gen2})
    checkpointer = Checkpointer(tmpdir)
    checkpointer.add_recoverable("gen", gen)
    # NOTE: Move the state a bit
    torch.randint(1, 10, (10,), generator=gen1)
    torch.randn((3, 3), generator=gen2)

    # NOTE: Save the checkpoint and get a reference
    checkpointer.save_checkpoint()
    x1_ref = torch.randint(1, 10, (10,), generator=gen1)
    x2_ref = torch.randn((3, 3), generator=gen2)
    # NOTE: Move the state even more, simulate usage
    for _ in range(5):
        torch.randint(1, 10, (10,), generator=gen1)
        torch.randn((3, 3), generator=gen2)

    # NOTE: Recover and compare
    checkpointer.recover_if_possible()
    x1 = torch.randint(1, 10, (10,), generator=gen1)
    x2 = torch.randn((3, 3), generator=gen2)
    assert (x1 == x1_ref).all()
    assert x2.allclose(x2_ref)


def test_repro_with_device(tmpdir, device):
    from speechbrain.utils.checkpoints import Checkpointer
    from speechbrain.utils.repro import SaveableGenerator

    if device == "cpu" or device.startswith("cuda"):
        gen = SaveableGenerator()
        checkpointer = Checkpointer(tmpdir, recoverables={"gen": gen})
        for _ in range(10):
            torch.randint(0, 10, (20, 20), device=device)
            torch.rand((10, 10))
        checkpointer.save_checkpoint()
        x = torch.randint(0, 10, (20, 20), device=device)
        y = torch.rand((10, 10))
        checkpointer.recover_if_possible()
        x_check = torch.randint(0, 10, (20, 20), device=device)
        y_check = torch.rand((10, 10))
        assert (x == x_check).all()
        assert y.allclose(y_check)

    else:
        warnings.warn(
            f"Device {device} is currently unsupported for saveable generations"
        )
