"""Unit tests for reproducibility utilities"""

import torch


def test_repro(tmpdir):
    from speechbrain.utils.repro import SaveableGenerator
    from speechbrain.utils.checkpoints import Checkpointer
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
