"""Comprehensive speech processing toolkit
"""
from .core import Stage, Brain, create_experiment_directory, parse_arguments
from .yaml import load_extended_yaml, resolve_references
import speechbrain.alignment
import speechbrain.data_io
import speechbrain.decoders
import speechbrain.lobes
import speechbrain.lm
import speechbrain.nnet
import speechbrain.processing
import speechbrain.tokenizers
import speechbrain.utils  # noqa

from speechbrain.utils.Accuracy import AccuracyStats

__all__ = [
    "Stage",
    "Brain",
    "create_experiment_directory",
    "parse_arguments",
    "load_extended_yaml",
    "resolve_references",
    "AccuracyStats",
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
