"""Classes and utils needed for the cooperative-learning recipe
"""
import torch
import logging
from collections import OrderedDict
from speechbrain.utils import checkpoints

from torch import nn
from speechbrain.processing.features import STFT
from speechbrain.lobes.models.transformer.Transformer import get_lookahead_mask
from speechbrain.lobes.models.transformer.TransformerSE import CNNTransformerSE
from speechbrain.lobes.features import Fbank

logger = logging.getLogger(__name__)


def torch_save(obj, path):
    """Saves the obj's parameters to path.

    Default save hook for torch.nn.Modules
    For saving torch.nn.Module state_dicts.

    Arguments
    ---------
    obj : torch.nn.Module
        Instance to save.
    path : str, pathlib.Path
        Path where to save to.

    Returns
    -------
    None
        State dict is written to disk.
    """
    state_dict = obj.state_dict()
    if not state_dict:
        logger.warning(f"Saving an empty state_dict for {obj} in {path}.")
    torch.save(state_dict, path)


def torch_recovery(obj, path, device=None):
    """Loads a torch.nn.Module state_dict from the given path instantly.

    This can be made the default for torch.nn.Modules with:
    >>> DEFAULT_LOAD_HOOKS[torch.nn.Module] = torch_recovery

    Arguments
    ---------
    obj : torch.nn.Module
        Instance for which to load the parameters.
    path : str, pathlib.Path
        Path where to load from.
    end_of_epoch : bool
        Whether the recovery comes from an end of epoch checkpoint.
    device : str
        Torch device, where to map the loaded parameters.

    Returns
    -------
    None
        Given object is modified in place.
    """
    try:
        obj.load_state_dict(torch.load(path, map_location=device), strict=True)
    except TypeError:
        obj.load_state_dict(torch.load(path, map_location=device))


@checkpoints.register_checkpoint_hooks
class MultipleOptimizer(object):
    """Class that allows the initialization of multiple optimization methods, and handles gradient updates."""

    def __init__(self, params, init_keys=["se_opt", "asr_opt"], **optim):
        self.optimizers = OrderedDict(
            {key: optim[key](params[key]) for key in init_keys}
        )

    def zero_grad(self):
        for op in self.optimizers.values():
            op.zero_grad()

    def step(self):
        for op in self.optimizers.values():
            op.step()

    @checkpoints.mark_as_saver
    def save(self, path):
        key = path.stem.split("optimizer_")[1]
        torch_save(self.optimizers[key], path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch, device):
        del end_of_epoch
        del device
        key = path.stem.split("optimizer_")[1]
        torch_recovery(self.optimizers[key], path)


def update_learning_rate(optimizer, new_lrs, param_group=None):
    """Change the learning rate value within an optimizer.

    Arguments
    ---------
    optimizer : torch.optim object
        Updates the learning rate for this optimizer.
    new_lr : float
        The new value to use for the learning rate.
    param_group : dict of list of int
        The param group indices to update. If not provided, all groups updated.

    Example
    -------
    >>> from torch.optim import SGD
    >>> from speechbrain.nnet.linear import Linear
    >>> model = Linear(n_neurons=10, input_size=10)
    >>> optimizer = SGD(model.parameters(), lr=0.1)
    >>> update_learning_rate(optimizer, 0.2)
    >>> optimizer.param_groups[0]["lr"]
    0.2
    """
    for key, new_lr in new_lrs.items():
        # Iterate all groups if none is provided
        if param_group is None:
            groups = range(len(optimizer.optimizers[key].param_groups))

        for i in groups:
            old_lr = optimizer.optimizers[key].param_groups[i]["lr"]

            # Change learning rate if new value is different from old.
            if new_lr != old_lr:
                optimizer.optimizers[key].param_groups[i]["lr"] = new_lr
                optimizer.optimizers[key].param_groups[i]["prev_lr"] = old_lr
                logger.info(
                    "Changing %s lr from %.2g to %.2g" % (key, old_lr, new_lr)
                )


@checkpoints.register_checkpoint_hooks
class MultipleOptimizerAnnealing(object):
    """Class that allows using multiple optimizer annealing methods, as well as saving their state."""

    def __init__(self, init_keys=["se_opt", "asr_opt"], **ann):
        self.annealers = OrderedDict({key: ann[key] for key in init_keys})

    def __call__(self, opt):
        new_value = {}
        old_value = {}
        for key, annealer in self.annealers.items():
            old_value[key], new_value[key] = annealer(opt)
        return old_value, new_value

    @checkpoints.mark_as_saver
    def save(self, path):
        for k, v in self.annealers.items():
            p = path.name.split(".")
            p[0] += f"_{k}_annealer"
            p = "".join(p)
            v.save(path.with_name(p))

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch, device):
        del end_of_epoch
        del device
        for k, v in self.annealers.items():
            p = path.name.split(".")
            p[0] += f"_{k}_annealer"
            p = "".join(p)
            v.load(path.with_name(p))


class MaskedCNNTransformerSE(CNNTransformerSE):
    """Wrapper around CNNTransformerSE that gives access to attn_mask on forward method"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, src_key_padding_mask=None, attn_mask=None):
        if attn_mask is None:
            if self.causal:
                self.attn_mask = get_lookahead_mask(x)
            else:
                self.attn_mask = None
        else:
            self.attn_mask = attn_mask

        if self.custom_emb_module is not None:
            x = self.custom_emb_module(x)

        encoder_output, _ = self.encoder(
            src=x,
            src_mask=self.attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        output = self.output_layer(encoder_output)
        output = self.output_activation(output)

        return output


class ScaleLayer(nn.Module):
    """Simple scaling parameter"""

    def __init__(self, init_value=1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModFbank(Fbank):
    """Quick hack to pass win_length and hop_length to STFT"""

    def __init__(self, win_length=36, hop_length=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sample_rate = self.compute_STFT.sample_rate
        n_fft = self.compute_STFT.n_fft
        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
