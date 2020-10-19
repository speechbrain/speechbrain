"""This package contains support functions.
"""
import os


def condition(filename):
    filename = os.path.basename(filename)
    return filename.endswith(".py") and not filename.startswith("__")


files = os.listdir(os.path.dirname(__file__))
__all__ = [filename[:-3] for filename in files if condition(filename)]

from . import *  # noqa
