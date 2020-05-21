collect_ignore = ["setup.py"]
try:
    import numba  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/nnet/transducer/transducer_loss.py")
