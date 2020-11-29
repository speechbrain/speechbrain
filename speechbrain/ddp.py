r"""
This is a copy of ``torch.distributed.launch`` git hash f7a8bf2

For docs, see that file.
"""


import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER
from typing import Optional, IO

node_local_rank_stdout_filename = "node_{}_local_rank_{}_stdout"
node_local_rank_stderr_filename = "node_{}_local_rank_{}_stderr"


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="PyTorch distributed training launch "
        "helper utility that will spawn up "
        "multiple distributed processes"
    )

    # Optional arguments for the launch helper
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="The number of nodes to use for distributed " "training",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The rank of the node for multi-node distributed " "training",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="The number of processes to launch on each node, "
        "for GPU training, this is recommended to be set "
        "to the number of GPUs in your system so that "
        "each process can be bound to a single GPU.",
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        type=str,
        help="Master node (rank 0)'s address, should be either "
        "the IP address or the hostname of node 0, for "
        "single node multi-proc training, the "
        "--master_addr can simply be 127.0.0.1",
    )
    parser.add_argument(
        "--master_port",
        default=29500,
        type=int,
        help="Master node (rank 0)'s free port that needs to "
        "be used for communication during distributed "
        "training",
    )
    parser.add_argument(
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local_rank as argument, and will instead set LOCAL_RANK.",
    )
    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        "'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        default=False,
        action="store_true",
        help='Do not prepend the training script with "python" - just exec '
        "it directly. Useful when the script is not a Python script.",
    )
    parser.add_argument(
        "--logdir",
        default=None,
        type=str,
        help=f"""Relative path to write subprocess logs to. Passing in a relative
        path will create a directory if needed, and write the stdout and stderr to files
        {node_local_rank_stdout_filename} and {node_local_rank_stderr_filename}. Note that
        successive runs with the  same path to write logs to will overwrite existing logs,
        so be sure to save logs as needed.""",
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if "OMP_NUM_THREADS" not in os.environ and args.nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print(
            "*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process "
            "to be {} in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************".format(
                current_env["OMP_NUM_THREADS"]
            )
        )

    if args.logdir:
        # Possibly create the directory to write subprocess log output to.
        if os.path.exists(args.logdir):
            if not os.path.isdir(args.logdir):
                raise ValueError(
                    "argument --logdir must be a path to a directory."
                )
        else:
            # create the relative directory
            os.mkdir(os.path.join(os.getcwd(), args.logdir))

    subprocess_file_handles = []

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        else:
            if not args.use_env:
                raise ValueError(
                    "When using the '--no_python' flag, you must also set the '--use_env' flag."
                )
            if args.module:
                raise ValueError(
                    "Don't use both the '--no_python' flag and the '--module' flag at the same time."
                )

        cmd.append(args.training_script)

        if not args.use_env:
            cmd.append("--local_rank={}".format(local_rank))

        cmd.extend(args.training_script_args)

        stdout_handle: Optional[IO]
        stderr_handle: Optional[IO]
        if args.logdir:
            directory_path = os.path.join(os.getcwd(), args.logdir)
            node_rank = args.node_rank
            stdout_file_name = node_local_rank_stdout_filename.format(
                node_rank, local_rank
            )
            stderr_file_name = node_local_rank_stderr_filename.format(
                node_rank, local_rank
            )
            stdout_handle = open(
                os.path.join(directory_path, stdout_file_name), "w"
            )
            stderr_handle = open(
                os.path.join(directory_path, stderr_file_name), "w"
            )
            subprocess_file_handles.append((stdout_handle, stderr_handle))
            stdout_name = stdout_handle.name
            stderr_name = stderr_handle.name
            print(
                f"""Note: Stdout and stderr for node {node_rank} rank {local_rank} will
            be written to {stdout_name}, {stderr_name} respectively."""
            )

        stdout_handle = (
            None
            if not subprocess_file_handles
            else subprocess_file_handles[local_rank][0]
        )
        stderr_handle = (
            None
            if not subprocess_file_handles
            else subprocess_file_handles[local_rank][1]
        )
        process = subprocess.Popen(
            cmd, env=current_env, stdout=stdout_handle, stderr=stderr_handle
        )
        processes.append(process)

    try:
        for process in processes:
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=process.returncode, cmd=cmd
                )
    finally:
        # close open file descriptors
        for (stdout_handle, stderr_handle) in subprocess_file_handles:
            stdout_handle.close()
            stderr_handle.close()


if __name__ == "__main__":
    main()
