import pytest
import torch


def test_checkpointer(tmpdir, device):
    from speechbrain.utils.checkpoints import Checkpointer

    class Recoverable(torch.nn.Module):
        def __init__(self, param):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor([param]))

        def forward(self, x):
            return x * self.param

    recoverable = Recoverable(2.0)
    recoverables = {"recoverable": recoverable}
    recoverer = Checkpointer(tmpdir, recoverables)
    recoverable.param.data = torch.tensor([1.0], device=device)
    # Should not be possible since no checkpoint saved yet:
    assert not recoverer.recover_if_possible()
    result = recoverable(10.0)
    # Check that parameter has not been loaded from original value:
    assert recoverable.param.data == torch.tensor([1.0], device=device)

    ckpt = recoverer.save_checkpoint()
    # Check that the name recoverable has a save file:
    # NOTE: Here assuming .pt filename; if convention changes, change test
    assert (ckpt.path / "recoverable.ckpt").exists()
    # Check that saved checkpoint is found, and location correct:
    assert recoverer.list_checkpoints()[0] == ckpt
    assert recoverer.list_checkpoints()[0].path.parent == tmpdir
    recoverable.param.data = torch.tensor([2.0], device=device)
    recoverer.recover_if_possible()
    # Check that parameter has been loaded immediately:
    assert recoverable.param.data == torch.tensor([1.0], device=device)
    result = recoverable(10.0)
    # And result correct
    assert result == 10.0

    other = Recoverable(2.0)
    recoverer.add_recoverable("other", other)
    # Check that both objects are now found:
    assert recoverer.recoverables["recoverable"] == recoverable
    assert recoverer.recoverables["other"] == other
    new_ckpt = recoverer.save_checkpoint()
    # Check that now both recoverables have a save file:
    assert (new_ckpt.path / "recoverable.ckpt").exists()
    assert (new_ckpt.path / "other.ckpt").exists()
    assert new_ckpt in recoverer.list_checkpoints()
    recoverable.param.data = torch.tensor([2.0], device=device)
    other.param.data = torch.tensor([10.0], device=device)
    chosen_ckpt = recoverer.recover_if_possible()
    # Should choose newest by default:
    assert chosen_ckpt == new_ckpt
    # Check again that parameters have been loaded immediately:
    assert recoverable.param.data == torch.tensor([1.0], device=device)
    assert other.param.data == torch.tensor([2.0], device=device)
    other_result = other(10.0)
    # And again we should have the correct computations:
    assert other_result == 20.0

    # Recover from oldest, which does not have "other":
    # This also tests a custom sort
    # Raises by default:
    with pytest.raises(RuntimeError):
        chosen_ckpt = recoverer.recover_if_possible(
            importance_key=lambda x: -x.meta["unixtime"]
        )
    # However this operation may have loaded the first object
    # so let's set the values manually:
    recoverable.param.data = torch.tensor([2.0], device=device)
    other.param.data = torch.tensor([10.0], device=device)
    recoverer.allow_partial_load = True
    chosen_ckpt = recoverer.recover_if_possible(
        importance_key=lambda x: -x.meta["unixtime"]
    )
    # Should have chosen the original:
    assert chosen_ckpt == ckpt
    # And should recover recoverable:
    assert recoverable.param.data == torch.tensor([1.0], device=device)
    # But not other:
    other_result = other(10.0)
    assert other.param.data == torch.tensor([10.0], device=device)
    assert other_result == 100.0

    # Test saving names checkpoints with meta info, and custom filter
    epoch_ckpt = recoverer.save_checkpoint(name="ep1", meta={"loss": 2.0})
    assert "ep1" in epoch_ckpt.path.name
    other.param.data = torch.tensor([2.0], device=device)
    recoverer.save_checkpoint(meta={"loss": 3.0})
    chosen_ckpt = recoverer.recover_if_possible(
        ckpt_predicate=lambda ckpt: "loss" in ckpt.meta,
        importance_key=lambda ckpt: -ckpt.meta["loss"],
    )
    assert chosen_ckpt == epoch_ckpt
    assert other.param.data == torch.tensor([10.0], device=device)

    # Make sure checkpoints can't be name saved by the same name
    # with pytest.raises(FileExistsError):
    #    recoverer.save_checkpoint(name="ep1")


def test_recovery_custom_io(tmpdir):
    from speechbrain.utils.checkpoints import (
        Checkpointer,
        mark_as_loader,
        mark_as_saver,
        register_checkpoint_hooks,
    )

    @register_checkpoint_hooks
    class CustomRecoverable:
        def __init__(self, param):
            self.param = int(param)

        @mark_as_saver
        def save(self, path):
            with open(path, "w", encoding="utf-8") as fo:
                fo.write(str(self.param))

        @mark_as_loader
        def load(self, path, end_of_epoch):
            del end_of_epoch  # Unused
            with open(path, encoding="utf-8") as fi:
                self.param = int(fi.read())

    custom_recoverable = CustomRecoverable(0)
    recoverer = Checkpointer(tmpdir, {"custom_recoverable": custom_recoverable})
    custom_recoverable.param = 1
    # First, make sure no checkpoints are found
    # (e.g. somehow tmpdir contaminated)
    ckpt = recoverer.recover_if_possible()
    assert ckpt is None
    ckpt = recoverer.save_checkpoint()
    custom_recoverable.param = 2
    loaded_ckpt = recoverer.recover_if_possible()
    # Make sure we got the same thing:
    assert ckpt == loaded_ckpt
    # With this custom recoverable, the load is instant:
    assert custom_recoverable.param == 1


def test_checkpoint_deletion(tmpdir, device):
    from speechbrain.utils.checkpoints import Checkpointer

    class Recoverable(torch.nn.Module):
        def __init__(self, param):
            super().__init__()
            self.param = torch.nn.Parameter(
                torch.tensor([param], device=device)
            )

        def forward(self, x):
            return x * self.param

    recoverable = Recoverable(1.0)
    recoverables = {"recoverable": recoverable}
    recoverer = Checkpointer(tmpdir, recoverables)
    first_ckpt = recoverer.save_checkpoint()
    recoverer.delete_checkpoints()
    # Will not delete only checkpoint by default:
    assert first_ckpt in recoverer.list_checkpoints()
    second_ckpt = recoverer.save_checkpoint()
    recoverer.delete_checkpoints()
    # Oldest checkpoint is deleted by default:
    assert first_ckpt not in recoverer.list_checkpoints()
    # Other syntax also should work:
    recoverer.save_and_keep_only()
    assert second_ckpt not in recoverer.list_checkpoints()
    # Can delete all checkpoints:
    recoverer.delete_checkpoints(num_to_keep=0)
    assert not recoverer.list_checkpoints()

    # Now each should be kept:
    # Highest foo
    c1 = recoverer.save_checkpoint(meta={"foo": 2})
    # Latest CKPT after filtering
    c2 = recoverer.save_checkpoint(meta={"foo": 1})
    # Filtered out
    c3 = recoverer.save_checkpoint(meta={"epoch_ckpt": True})
    recoverer.delete_checkpoints(
        num_to_keep=1,
        max_keys=["foo"],
        importance_keys=[lambda c: c.meta["unixtime"]],
        ckpt_predicate=lambda c: "epoch_ckpt" not in c.meta,
    )
    assert all(c in recoverer.list_checkpoints() for c in [c1, c2, c3])
    # Reset:
    recoverer.delete_checkpoints(num_to_keep=0)
    assert not recoverer.list_checkpoints()

    # Test the keeping multiple checkpoints without predicate:
    # This should be deleted:
    c_to_delete = recoverer.save_checkpoint(meta={"foo": 2})
    # Highest foo
    c1 = recoverer.save_checkpoint(meta={"foo": 3})
    # Latest CKPT after filtering
    c2 = recoverer.save_checkpoint(meta={"foo": 1})
    recoverer.delete_checkpoints(
        num_to_keep=1,
        importance_keys=[lambda c: c.meta["unixtime"], lambda c: c.meta["foo"]],
    )
    assert all(c in recoverer.list_checkpoints() for c in [c1, c2])
    assert c_to_delete not in recoverer.list_checkpoints()


def test_multiple_ckpts_and_criteria(tmpdir):
    from speechbrain.utils.checkpoints import Checkpointer

    class Recoverable(torch.nn.Module):
        def __init__(self, param):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor([param]))

        def forward(self, x):
            return x * self.param

    recoverable = Recoverable(1.0)
    recoverables = {"recoverable": recoverable}
    recoverer = Checkpointer(tmpdir, recoverables)

    # Here testing multiple checkpoints with equal meta criteria
    recoverer.save_and_keep_only(
        meta={"error": 5}, min_keys=["error"], keep_recent=True
    )
    # By default, get the most recent one:
    first_ckpt = recoverer.find_checkpoint()
    recoverer.save_and_keep_only(
        meta={"error": 5}, min_keys=["error"], keep_recent=True
    )
    second_ckpt = recoverer.find_checkpoint()
    assert first_ckpt.meta["unixtime"] < second_ckpt.meta["unixtime"]
    recoverer.save_and_keep_only(
        meta={"error": 6}, min_keys=["error"], keep_recent=True
    )
    third_ckpt = recoverer.find_checkpoint()
    remaining_ckpts = recoverer.list_checkpoints()
    assert first_ckpt not in remaining_ckpts
    assert second_ckpt in remaining_ckpts
    assert third_ckpt in remaining_ckpts

    # With equal importance criteria, the latest checkpoint should always be
    # returned
    fourth_ckpt = recoverer.save_checkpoint(meta={"error": 5})
    found_ckpt = recoverer.find_checkpoint(min_key="error")
    assert found_ckpt == fourth_ckpt
    fifth_ckpt = recoverer.save_checkpoint(meta={"error": 5})
    # Similarly for getting multiple checkpoints:
    found_ckpts = recoverer.find_checkpoints(
        min_key="error", max_num_checkpoints=2
    )
    assert found_ckpts == [fifth_ckpt, fourth_ckpt]


def test_average_ckpts(tmpdir):
    from speechbrain.utils.checkpoints import Checkpointer, average_checkpoints

    class Recoverable(torch.nn.Module):
        def __init__(self, param):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor([param]))

        def forward(self, x):
            return x * self.param

    N_avg = 2
    recoverable = Recoverable(1.0)
    recoverables = {"recoverable": recoverable}
    recoverer = Checkpointer(tmpdir, recoverables)

    # save first checkpoint
    recoverer.save_and_keep_only(
        meta={"error": 5},
        min_keys=["error"],
        keep_recent=True,
        num_to_keep=N_avg,
    )

    # Save another checkpoint
    recoverable.param = torch.nn.Parameter(torch.tensor([3.0]))

    recoverer.save_and_keep_only(
        meta={"error": 4},
        min_keys=["error"],
        keep_recent=True,
        num_to_keep=N_avg,
    )

    recoverer.recover_if_possible()

    checkpoints = recoverer.find_checkpoints(max_num_checkpoints=N_avg)

    model_state_dict = average_checkpoints(checkpoints, "recoverable")

    assert model_state_dict["param"] == 2.0


def test_torch_meta(tmpdir, device):
    from speechbrain.utils.checkpoints import Checkpointer

    class Recoverable(torch.nn.Module):
        def __init__(self, param):
            super().__init__()
            self.param = torch.nn.Parameter(
                torch.tensor([param], device=device)
            )

        def forward(self, x):
            return x * self.param

    recoverable = Recoverable(1.0)
    recoverables = {"recoverable": recoverable}
    recoverer = Checkpointer(tmpdir, recoverables)
    saved = recoverer.save_checkpoint(
        meta={"loss": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)}
    )
    loaded = recoverer.recover_if_possible()
    assert saved.meta["loss"].allclose(loaded.meta["loss"])


def test_checkpoint_hook_register(tmpdir):
    from speechbrain.utils.checkpoints import (
        Checkpointer,
        mark_as_loader,
        mark_as_saver,
        register_checkpoint_hooks,
    )

    # First a proper interface:
    @register_checkpoint_hooks
    class CustomRecoverable:
        def __init__(self, param):
            self.param = int(param)

        @mark_as_saver
        def save(self, path):
            with open(path, "w", encoding="utf-8") as fo:
                fo.write(str(self.param))

        @mark_as_loader
        def load(self, path, end_of_epoch):
            del end_of_epoch  # Unused
            with open(path, encoding="utf-8") as fi:
                self.param = int(fi.read())

    recoverable = CustomRecoverable(1.0)
    checkpointer = Checkpointer(tmpdir, {"recoverable": recoverable})
    checkpointer.save_checkpoint()
    recoverable.param = 2.0
    checkpointer.recover_if_possible()
    assert recoverable.param == 1.0

    # Improper interfaces:
    with pytest.raises(TypeError):

        class BadRecoverable:
            def __init__(self, param):
                self.param = int(param)

            def save(self, path):
                with open(path, "w", encoding="utf-8") as fo:
                    fo.write(str(self.param))

            @mark_as_loader
            def load(self, path):  # MISSING end_of_epoch
                with open(path, encoding="utf-8") as fi:
                    self.param = int(fi.read())

    with pytest.raises(TypeError):

        class BadRecoverable:  # noqa: F811
            def __init__(self, param):
                self.param = int(param)

            @mark_as_saver
            def save(self, path, extra_arg):  # Extra argument
                with open(path, "w", encoding="utf-8") as fo:
                    fo.write(str(self.param))

            def load(self, path, end_of_epoch):
                del end_of_epoch  # Unused
                with open(path, encoding="utf-8") as fi:
                    self.param = int(fi.read())


def test_torch_defaults(tmpdir, device):
    from speechbrain.utils.checkpoints import Checkpointer

    module = torch.nn.Linear(10, 10).to(device)
    optimizer = torch.optim.Adam(module.parameters())
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 0.1, 1.0, cycle_momentum=False
    )
    # ReduceLROnPlateau is on an _LRScheduler for some reason, so have a separate test for it
    another_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    checkpointer = Checkpointer(
        tmpdir,
        recoverables={
            "module": module,
            "optimizer": optimizer,
            "scheduler": lr_scheduler,
            "scheduler2": another_scheduler,
        },
    )
    ckpt = checkpointer.save_checkpoint()
    # test the module:
    inp = torch.randn((3, 10), device=device)
    prev_output = module(inp)

    # Re-initialize everything
    module = torch.nn.Linear(10, 10, device=device)
    optimizer = torch.optim.Adam(module.parameters())
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 0.1, 1.0, cycle_momentum=False
    )
    another_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    checkpointer = Checkpointer(
        tmpdir,
        recoverables={
            "module": module,
            "optimizer": optimizer,
            "scheduler": lr_scheduler,
            "scheduler2": another_scheduler,
        },
    )
    checkpointer.load_checkpoint(ckpt)
    assert torch.allclose(module(inp), prev_output)


def parallel_checkpoint(rank, world_size, tmpdir):
    import os

    from speechbrain.utils.checkpoints import Checkpointer

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # initialize the process group
    sync_file = f"file://{tmpdir}/sync"
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=sync_file
    )

    model = torch.nn.Linear(10, 10, device="cpu")
    checkpointer = Checkpointer(tmpdir, recoverables={"model": model})
    ckpt = checkpointer.save_checkpoint()

    if rank != 0:
        assert ckpt is None
    if rank == 0:
        # Check that only a single checkpoint is saved, even in ddp
        # Second file is the DDP synchronization file
        assert len(os.listdir(tmpdir)) == 2

        # Check that the model is saved
        inp = torch.randn((3, 10), device="cpu")
        prev_output = model(inp)
        checkpointer.load_checkpoint(ckpt)
        assert torch.allclose(model(inp), prev_output)


def test_parallel_checkpoint(tmpdir):
    world_size = 2
    torch.multiprocessing.spawn(
        parallel_checkpoint,
        args=(world_size, tmpdir),
        nprocs=world_size,
        join=True,
    )
