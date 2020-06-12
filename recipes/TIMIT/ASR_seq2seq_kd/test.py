import os
import sys
import torch
import speechbrain as sb
from speechbrain.utils.checkpoints import Checkpoint

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

 #SETUP:
tempdir = '/nfs-share/yan/test'
class Recoverable(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor([param]))
    def forward(self, x):
        return x * self.param
recoverable = Recoverable(1.)
recoverables = {'recoverable': recoverable}
# SETUP DONE.
checkpointer = Checkpoint(tempdir, recoverables, )
first_ckpt = checkpointer.save_checkpoint()
# recoverable.param.data = torch.tensor([2.])
# loaded_ckpt = checkpointer.recover_if_possible()
# # Parameter has been loaded:
# assert recoverable.param.data == torch.tensor([1.])
# # With this call, by default, oldest checkpoints are deleted:
# checkpointer.save_and_keep_only()
# assert first_ckpt not in checkpointer.list_checkpoints()