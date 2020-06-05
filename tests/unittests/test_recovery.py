import pytest


def test_checkpointer(tmpdir):
    from speechbrain.utils.checkpoints import Checkpointer
    import torch

    class Recoverable(torch.nn.Module):
        def __init__(self, param):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor([param]))

        def forward(self, x):
            return x * self.param

    recoverable = Recoverable(2.0)
    recoverables = {"recoverable": recoverable}
    recoverer = Checkpointer(tmpdir, recoverables)
    recoverable.param.data = torch.tensor([1.0])
    # Should not be possible since no checkpoint saved yet:
    assert not recoverer.recover_if_possible()
    result = recoverable(10.0)
    # Check that parameter has not been loaded from original value:
    assert recoverable.param.data == torch.tensor([1.0])

    ckpt = recoverer.save_checkpoint()
    # Check that the name recoverable has a save file:
    # NOTE: Here assuming .pt filename; if convention changes, change test
    assert (ckpt.path / "recoverable.ckpt").exists()
    # Check that saved checkpoint is found, and location correct:
    assert recoverer.list_checkpoints()[0] == ckpt
    assert recoverer.list_checkpoints()[0].path.parent == tmpdir
    recoverable.param.data = torch.tensor([2.0])
    recoverer.recover_if_possible()
    # Check that parameter has been loaded immediately:
    assert recoverable.param.data == torch.tensor([1.0])
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
    recoverable.param.data = torch.tensor([2.0])
    other.param.data = torch.tensor([10.0])
    chosen_ckpt = recoverer.recover_if_possible()
    # Should choose newest by default:
    assert chosen_ckpt == new_ckpt
    # Check again that parameters have been loaded immediately:
    assert recoverable.param.data == torch.tensor([1.0])
    assert other.param.data == torch.tensor([2.0])
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
    recoverable.param.data = torch.tensor([2.0])
    other.param.data = torch.tensor([10.0])
    recoverer.allow_partial_load = True
    chosen_ckpt = recoverer.recover_if_possible(
        importance_key=lambda x: -x.meta["unixtime"]
    )
    # Should have chosen the original:
    assert chosen_ckpt == ckpt
    # And should recover recoverable:
    assert recoverable.param.data == torch.tensor([1.0])
    # But not other:
    other_result = other(10.0)
    assert other.param.data == torch.tensor([10.0])
    assert other_result == 100.0

    # Test saving names checkpoints with meta info, and custom filter
    epoch_ckpt = recoverer.save_checkpoint(name="ep1", meta={"loss": 2.0})
    assert "ep1" in epoch_ckpt.path.name
    other.param.data = torch.tensor([2.0])
    recoverer.save_checkpoint(meta={"loss": 3.0})
    chosen_ckpt = recoverer.recover_if_possible(
        ckpt_predicate=lambda ckpt: "loss" in ckpt.meta,
        importance_key=lambda ckpt: -ckpt.meta["loss"],
    )
    assert chosen_ckpt == epoch_ckpt
    assert other.param.data == torch.tensor([10.0])

    # Make sure checkpoints can't be name saved by the same name
    with pytest.raises(FileExistsError):
        recoverer.save_checkpoint(name="ep1")


def test_recovery_custom_io(tmpdir):
    from speechbrain.utils.checkpoints import register_checkpoint_hooks
    from speechbrain.utils.checkpoints import mark_as_saver
    from speechbrain.utils.checkpoints import mark_as_loader
    from speechbrain.utils.checkpoints import Checkpointer

    @register_checkpoint_hooks
    class CustomRecoverable:
        def __init__(self, param):
            self.param = int(param)

        @mark_as_saver
        def save(self, path):
            with open(path, "w") as fo:
                fo.write(str(self.param))

        @mark_as_loader
        def load(self, path, end_of_epoch):
            del end_of_epoch  # Unused
            with open(path) as fi:
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


def test_checkpoint_deletion(tmpdir):
    from speechbrain.utils.checkpoints import Checkpointer
    import torch

    class Recoverable(torch.nn.Module):
        def __init__(self, param):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor([param]))

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
        importance_keys=[lambda c: c.meta["unixtime"], lambda c: c.meta["foo"]],
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
