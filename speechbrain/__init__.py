"""Comprehensive speech processing toolkit
"""
from .core import Stage, Brain, create_experiment_directory, parse_arguments
from .core import ddp_init_group, if_main_process, ddp_barrier
from .yaml import load_extended_yaml, resolve_references
from speechbrain import (
    alignment,
    data_io,
    decoders,
    lobes,
    lm,
    nnet,
    processing,
    tokenizers,
    utils,
)

__all__ = [
    "Stage",
    "Brain",
    "create_experiment_directory",
    "parse_arguments",
    "ddp_init_group",
    "if_main_process",
    "ddp_barrier",
    "load_extended_yaml",
    "resolve_references",
    "alignment",
    "data_io",
    "decoders",
    "lobes",
    "lm",
    "nnet",
    "processing",
    "tokenizers",
    "utils",
]


class TestThing:
    # Purely for test purposes.
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_keys(cls, args, kwargs):
        obj = cls()
        obj.specific_key = kwargs["thing1"]
        obj.args = args
        obj.kwargs = kwargs
        return obj
