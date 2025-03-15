from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
import os
import sys
import argparse
import torch


@dataclass
class RunOptions:
    test_only: bool = False
    debug: bool = False
    debug_batches: int = 2
    debug_epochs: int = 2
    debug_persistently: bool = False
    device: str = "cpu"
    data_parallel_backend: bool = False
    data_parallel_count: int = -1
    distributed_backend: str = "nccl"
    distributed_launch: bool = False
    find_unused_parameters: bool = False
    jit: bool = False
    jit_module_keys: Optional[None] = None
    compile: bool = False
    compile_module_keys: Optional[None] = None
    compile_mode: str = "default"
    compile_using_fullgraph: bool = False
    compile_using_dynamic_shape_tracing: bool = True
    precision: str = "fp32"
    eval_precision: str = "fp32"
    auto_mix_prec: bool = False
    bfloat16_mix_prec: bool = False
    max_grad_norm: float = 5.0
    skip_nonfinite_grads: bool = False
    nonfinite_patience: int = 3
    noprogressbar: bool = False
    ckpt_interval_minutes: int = 0
    ckpt_interval_steps: int = 0
    grad_accumulation_factor: int = 1
    optimizer_step_limit: Optional[None] = None
    tqdm_colored_bar: bool = False
    tqdm_barcolor: Dict[str, str] = field(
        default_factory=lambda: {"train": "GREEN", "valid": "MAGENTA", "test": "CYAN"}
    )
    remove_vector_weight_decay: bool = False
    profile_training: bool = False
    profile_warmup: int = 5
    profile_steps: int = 5
    log_config: Optional[str] = None

    def as_dict(self) -> Dict:
        """
        Converts the instance into a dictionary.

        Returns:
            Dict: A dictionary representation of the instance.
        """
        return asdict(self)

    @classmethod
    def from_command_line_args(cls, arg_list=None):
        """Parse command-line arguments to the experiment.

        Arguments
        ---------
        arg_list : list, None
            A list of arguments to parse.  If not given, this is read from
            `sys.argv[1:]`

        Returns
        -------
        param_file : str
            The location of the parameters file.
        run_opts : dict
            Run options, such as distributed, device, etc.
        overrides : dict
            The overrides to pass to ``load_hyperpyyaml``.

        Example
        -------
        >>> argv = ['hyperparams.yaml', '--device', 'cuda:1', '--seed', '10']
        >>> filename, run_opts, overrides = parse_arguments(argv)
        >>> filename
        'hyperparams.yaml'
        >>> run_opts["device"]
        'cuda:1'
        >>> overrides
        'seed: 10'
        """
        if arg_list is None:
            arg_list = sys.argv[1:]
        parser = argparse.ArgumentParser(description="Run a SpeechBrain experiment")
        parser.add_argument(
            "param_file",
            type=str,
            help="A yaml-formatted file using the extended YAML syntax. "
            "defined by SpeechBrain.",
        )
        parser.add_argument(
            "--test_only",
            default=False,
            action="store_true",
            help="Run the experiment in evaluate only mode."
            "It skips the training and goes directly to the evaluation."
            "The model is expected to be already trained.",
        )
        parser.add_argument(
            "--debug",
            default=False,
            action="store_true",
            help="Run the experiment with only a few batches for all "
            "datasets, to ensure code runs without crashing.",
        )
        parser.add_argument(
            "--debug_batches",
            type=int,
            default=2,
            help="Number of batches to run in debug mode.",
        )
        parser.add_argument(
            "--debug_epochs",
            type=int,
            default=2,
            help="Number of epochs to run in debug mode. "
            "If a non-positive number is passed, all epochs are run.",
        )
        parser.add_argument(
            "--debug_persistently",
            default=False,
            action="store_true",
            help="Keep data stored during debug mode (not using /tmp).",
        )
        parser.add_argument(
            "--log_config",
            type=str,
            help="A file storing the configuration options for logging",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="The device to run the experiment on (e.g. 'cuda:0')",
        )
        parser.add_argument(
            "--data_parallel_backend",
            default=False,
            action="store_true",
            help="This flag enables training with data_parallel.",
        )
        parser.add_argument(
            "--distributed_backend",
            type=str,
            default="nccl",
            help="One of {nccl, gloo, mpi}",
        )
        parser.add_argument(
            "--find_unused_parameters",
            default=False,
            action="store_true",
            help="This flag disable unused parameters detection",
        )
        parser.add_argument(
            "--jit",
            default=False,
            action="store_true",
            help="Enables jit compilation for all modules. "
            "Compilation may fail depending on the modules. "
            "Use --jit_module_keys to compile a subset of modules.",
        )
        parser.add_argument(
            "--jit_module_keys",
            type=str,
            nargs="*",
            help="A list of keys in the 'modules' dict to jitify",
        )
        parser.add_argument(
            "--compile",
            default=False,
            action="store_true",
            help="Enabling this flag compiles all modules using torch.compile (if available). "
            "Beta feature. Use --compile_module_keys to compile a subset of modules. "
            "Set the compilation flags below properly. "
            "Compilation can be time-consuming and might fail.",
        )
        parser.add_argument(
            "--compile_module_keys",
            type=str,
            nargs="*",
            help="A list of keys in the 'modules' dict to compile using "
            "TorchInductor. If a module also has a JIT key specified, "
            "TorchInductor will take precedence when available.",
        )
        parser.add_argument(
            "--compile_mode",
            type=str,
            nargs="*",
            help="One of {default, reduce-overhead, max-autotune}",
        )
        parser.add_argument(
            "--compile_using_fullgraph",
            type=bool,
            nargs="*",
            help="Whether it is ok to break model into several subgraphs",
        )
        parser.add_argument(
            "--compile_using_dynamic_shape_tracing",
            type=bool,
            nargs="*",
            help="Use dynamic shape tracing for compilation",
        )
        parser.add_argument(
            "--precision",
            type=str,
            help="This flag enables training with automatic mixed-precision."
            "It can be set to `fp32`, `fp16`, or `bf16`.",
        )
        parser.add_argument(
            "--eval_precision",
            type=str,
            help="This flag enables inference with automatic mixed-precision."
            "It can be set to `fp32`, `fp16`, or `bf16`.",
        )
        parser.add_argument(
            "--auto_mix_prec",
            default=None,
            action="store_true",
            help="This flag enables training with automatic mixed-precision.",
        )
        parser.add_argument(
            "--bfloat16_mix_prec",
            default=None,
            action="store_true",
            help="This flag enables training with bfloat16 mixed-precision.",
        )
        parser.add_argument(
            "--max_grad_norm",
            type=float,
            help="Gradient norm will be clipped to this value, "
            "enter negative value to disable.",
        )
        parser.add_argument(
            "--skip_nonfinite_grads",
            default=False,
            action="store_true",
            help="Set the gradients to None if they are nonfinite (inf or nan).",
        )
        parser.add_argument(
            "--nonfinite_patience",
            type=int,
            help="Max number of batches per epoch to skip if loss is nonfinite.",
        )
        parser.add_argument(
            "--noprogressbar",
            default=None,
            action="store_true",
            help="This flag disables the data loop progressbars.",
        )
        parser.add_argument(
            "--ckpt_interval_minutes",
            type=float,
            help="Amount of time between saving intra-epoch checkpoints "
            "in minutes. If non-positive, intra-epoch checkpoints are not saved.",
        )
        parser.add_argument(
            "--ckpt_interval_steps",
            type=int,
            help="Save an intra-epoch checkpoint after this many steps."
            "If non-positive, intra-epoch checkpoints are not saved.",
        )
        parser.add_argument(
            "--grad_accumulation_factor",
            type=int,
            help="Number of batches to accumulate gradients before optimizer step",
        )
        parser.add_argument(
            "--optimizer_step_limit",
            type=int,
            help="Number of optimizer steps to run. If not passed, all epochs are run.",
        )
        parser.add_argument(
            "--tqdm_colored_bar",
            default=False,
            action="store_true",
            help="Enable colored progress-bar in tqdm. If this is "
            "false, tqdm shall use default colors.",
        )
        parser.add_argument(
            "--remove_vector_weight_decay",
            default=False,
            action="store_true",
            help="Make vectors (e.g. norms and biases) a separate parameter group without weight_decay.",
        )
        parser.add_argument(
            "--profile_training",
            default=False,
            action="store_true",
            help=(
                "If set to True, a profiler will be initiated and tensorboard logs will be generated. "
                "Please ensure you have installed the torch.TensorBoard profiler with 'pip install torch_tb_profiler'."
            ),
        )
        parser.add_argument(
            "--profile_warmup",
            default=5,
            type=int,
            help="Number of warmup steps before logging for the profiler.",
        )
        parser.add_argument(
            "--profile_steps",
            default=5,
            type=int,
            help="Number of steps of logging for the profiler",
        )

        # Accept extra args to override yaml
        parsed_args, overrides = parser.parse_known_args(arg_list)
        args_dict = vars(parsed_args)
        param_file = args_dict.pop("param_file")
        run_opts = cls(**args_dict)

        overrides = cls._convert_to_yaml(overrides)

        # Checking that DataParallel use the right number of GPU
        if run_opts.data_parallel_backend and torch.cuda.device_count() == 0:
            raise ValueError("You must have at least 1 GPU.")

        # force device arg to be the same as local_rank from torchrun
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None and "cuda" in run_opts.device:
            run_opts.devide = run_opts.devide[:-1] + str(local_rank)

        return param_file, cls, overrides

    @staticmethod
    def _convert_to_yaml(overrides):
        """
        Convert a list of override arguments to a YAML formatted string.
        Args:
            overrides (list): A list of strings representing override arguments in the form '--arg=val'.
        Returns:
            str: A YAML formatted string representing the overrides.
        """
        """Convert args to yaml for overrides"""
        yaml_string = ""

        # Handle '--arg=val' type args
        joined_args = "=".join(overrides)
        split_args = joined_args.split("=")

        for arg in split_args:
            if arg.startswith("--"):
                yaml_string += "\n" + arg[len("--") :] + ":"
            else:
                yaml_string += " " + arg

        return yaml_string.strip()
