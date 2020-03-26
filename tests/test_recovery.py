import pytest

def test_recoverer(tmpdir_factory):
    from speechbrain.utils.recovery import Recoverer
    import torch

    class recoverable(torch.nn.Module):
        def __init__(self, param):
            super().__init__()
            self.param = torch.nn.Parameter(torch.Tensor([param]))
    
    recoverables = {"recoverable": recoverable(1.)}
    tmpdir = tmpdir_factory.mktemp("checkpoint")
    recoverer = Recoverer(tmpdir, recoverables)
    recoverer.save_checkpoint()
    assert recoverer.list_checkpoint_dirs()[0].parent == tmpdir
