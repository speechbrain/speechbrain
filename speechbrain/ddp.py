"""Python module for starting an experiment on multiple processes for DDP.

This module is based off of the ``multiproc.py`` module in:
https://github.com/NVIDIA/tacotron2

To start a DDP experiment with this module, do:

> python -m speechbrain.ddp experiment.py hyperparams.yaml

Authors:
 * Peter Plantinga 2020
 * Abdel HEBA 2020
"""
import os
import sys
import torch
import argparse
import subprocess
import speechbrain as sb

# Parse arguments relevant to ddp, while ignoring the rest
supported_ddp = ["ddp_nccl", "ddp_gloo", "ddp_mpi"]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--multigpu_backend",
    choices=supported_ddp,
    default=None,
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
# passed_arglist[1] = 'hyperparams.yaml' use it
# to check the multigpu_{count,backend}
hparams_file, overrides = sb.parse_arguments(passed_arglist[1:])
with open(hparams_file) as fin:
    hparams = sb.load_extended_yaml(fin, overrides)

# Handle the multigpu_backend input
if ddp_args.multigpu_backend is None:
    if hparams["multigpu_backend"] is not None:
        if hparams["multigpu_backend"] in supported_ddp:
            ddp_args.multigpu_backend = hparams["multigpu_backend"]
        else:
            print(
                "Attribute multigpu_backend="
                + str(hparams["multigpu_backend"])
                + " in the yaml file"
            )
            print("We do support the following ddp:" + str(supported_ddp))
            print("ddp_nccl is selected by default")
            ddp_args.multigpu_backend = "ddp_nccl"
    else:
        print("No input for multigpu_backend, ddp_nccl is selected")
        ddp_args.multigpu_backend = "ddp_nccl"

# Pass backend unchanged
passed_arglist += [f"--multigpu_backend={ddp_args.multigpu_backend}"]

# Handle the multigpu_count input
if ddp_args.multigpu_count is None and int(hparams["multigpu_count"]) > 0:
    ddp_args.multigpu_count = int(hparams["multigpu_count"])
else:
    print("No input for multigpu_count, using all GPUs")
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
