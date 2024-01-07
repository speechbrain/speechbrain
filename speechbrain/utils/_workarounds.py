"""This module implements some workarounds for dependencies

Authors
 * Aku Rouhe 2022
"""
import torch
import weakref
import warnings

WEAKREF_MARKER = "WEAKREF"


def _cycliclrsaver(obj, path):
    state_dict = obj.state_dict()
    if state_dict.get("_scale_fn_ref") is not None:
        state_dict["_scale_fn_ref"] = WEAKREF_MARKER
    torch.save(state_dict, path)


def _cycliclrloader(obj, path, end_of_epoch):
    del end_of_epoch  # Unused
    device = "cpu"
    state_dict = torch.load(path, map_location=device)
    if state_dict.get("_scale_fn_ref") == WEAKREF_MARKER:
        if not isinstance(obj._scale_fn_ref, weakref.WeakMethod):
            MSG = "Loading CyclicLR scheduler and the _scale_ref_fn did not exist in instance."
            MSG += " You did not construct it with the same parameters it was created!"
            MSG += " Looks like you changed the scale function!"
            MSG += " If this was not intentional, the scheduler might not work correctly."
            warnings.warn(MSG)
    try:
        obj.load_state_dict(torch.load(path, map_location=device), strict=True)
    except TypeError:
        obj.load_state_dict(torch.load(path, map_location=device))
