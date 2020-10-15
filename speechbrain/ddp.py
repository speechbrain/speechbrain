"""Python module for starting an experiment on multiple processes for DDP.

This module is based off of the ``multiproc.py`` module in:
https://github.com/NVIDIA/tacotron2

To start a DDP experiment with this module, do:

> python -m speechbrain.ddp experiment.py hyperparams.yaml

Authors:
 * Peter Plantinga 2020
"""
import sys
import torch
import subprocess

original_argslist = list(sys.argv)[1:]
num_gpus = torch.cuda.device_count()
workers = []

for i in range(num_gpus):
    argslist = original_argslist + [f"--device=cuda:{i}"]
    stdout = None if i == 0 else open(f"log_{i}.log", "w")
    p = subprocess.Popen([str(sys.executable)] + argslist, stdout=stdout)
    workers.append(p)
    argslist = argslist

for p in workers:
    p.wait()
