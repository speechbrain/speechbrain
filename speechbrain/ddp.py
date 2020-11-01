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
import argparse
import subprocess

# Parse arguments relevant to ddp, while ignoring the rest
parser = argparse.ArgumentParser()
parser.add_argument(
    "--multigpu_backend",
    choices=["ddp_nccl", "ddp_gloo", "ddp_mpi"],
    default="ddp_nccl",
    help="Backend for training with Distributed Data Parallel",
)
parser.add_argument(
    "--multigpu_count",
    type=int,
    default=None,
    help="Number of devices to run this experiment on",
)
parser.add_argument(
    "--master_addr",
    default="127.0.0.1",
    help="IP address of the master process",
)
parser.add_argument(
    "--master_port",
    default="12321",
    help="Port to use for communicating between processes",
)
ddp_args, passed_arglist = parser.parse_known_args(sys.argv[1:])

# Pass backend unchanged
passed_arglist += [f"--multigpu_backend={ddp_args.multigpu_backend}"]

# Pass number of gpus to use, using all if not specified
if ddp_args.multigpu_count is None:
    ddp_args.multigpu_count = torch.cuda.device_count()
passed_arglist += [f"--multigpu_count={ddp_args.multigpu_count}"]

# Set appropriate environ objects
current_env = os.environ.copy()
current_env["MASTER_ADDR"] = ddp_args.master_addr
current_env["MASTER_PORT"] = ddp_args.master_port

# Create logs folder
ddp_logs_folder = "ddp_logs"
print(f"Creating logs directory in {ddp_logs_folder} for non-root process logs")
if not os.path.isdir(ddp_logs_folder):
    os.mkdir(ddp_logs_folder)

print(f"Starting {ddp_args.multigpu_count} processes")
workers = []
for rank in range(ddp_args.multigpu_count):

    # Rank is passed by environment variable to bypass the yaml
    current_env["RANK"] = str(rank)

    # Build command
    cmd = [sys.executable] + passed_arglist
    cmd += [f"--device=cuda:{rank}"]

    # Logfile location for non-root processes
    outfile = os.path.join(ddp_logs_folder, f"log_{rank}.out")
    errfile = os.path.join(ddp_logs_folder, f"log_{rank}.err")
    stdout = None if rank == 0 else open(outfile, "w")
    stderr = None if rank == 0 else open(errfile, "w")

    # Spawn process
    p = subprocess.Popen(cmd, env=current_env, stdout=stdout, stderr=stderr)
    workers.append(p)

for p in workers:
    p.wait()
