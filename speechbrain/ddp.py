"""Python module for starting an experiment on multiple processes for DDP.

This module is based off of the ``multiproc.py`` module in:
https://github.com/NVIDIA/tacotron2

To start a DDP experiment with this module, do:

> python -m speechbrain.ddp experiment.py hyperparams.yaml

Authors:
 * Peter Plantinga 2020
"""
import os
import sys
import torch
import subprocess

original_argslist = list(sys.argv)[1:]
num_gpus = torch.cuda.device_count()
original_argslist += [f"--multigpu_count={num_gpus}"]
original_argslist += ["--multigpu_backend=ddp_nccl"]
workers = []
ddp_logs_folder = "ddp_logs"
if not os.path.isdir(ddp_logs_folder):
    os.mkdir(ddp_logs_folder)

for i in range(num_gpus):
    argslist = original_argslist + [f"--device=cuda:{i}", f"--rank={i}"]
    logfile = os.path.join(ddp_logs_folder, f"log_{i}.log")
    stdout = None if i == 0 else open(logfile, "w")
    p = subprocess.Popen([str(sys.executable)] + argslist, stdout=stdout)
    workers.append(p)
    argslist = argslist

for p in workers:
    p.wait()
