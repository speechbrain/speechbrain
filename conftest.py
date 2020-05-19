import sys

collect_ignore = ["setup.py"]
try:
    import numba
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/nnet/transducer/transducer_loss.py")

