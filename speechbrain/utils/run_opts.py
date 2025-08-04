"""
Contains the defaults and parsing code for run-time controls

Authors
 * Nouran Ali 2025
 * Peter Plantinga 2025
"""

import argparse
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, Literal, Optional

HELP_TEXTS = {
    "test_only": "Run the experiment in evaluate only mode, which skips the training and "
    "goes directly to the evaluation. The model is expected to be already trained.",
    "debug": "Run with only a few batches and few epochs to ensure code runs without crashing.",
    "debug_batches": "Number of batches to run in debug mode.",
    "debug_epochs": "Number of epochs to run in debug mode. If a non-positive number is passed, all epochs are run.",
    "debug_persistently": "Keep data stored during debug mode (not using /tmp).",
    "log_config": "A file storing the configuration options for logging",
    "device": "The device to run the experiment on (e.g. 'cuda:0')",
    "data_parallel_backend": "This flag enables training with data_parallel.",
    "distributed_backend": "One of {nccl, gloo, mpi}",
    "find_unused_parameters": "This flag disable unused parameters detection",
    "jit": "Enables jit compilation for all modules. Compilation may fail for some modules. "
    "Use 'jit_module_keys' to compile a subset of modules.",
    "compile": "Enabling this flag compiles all modules using torch.compile (if available). "
    "Beta feature. Use 'compile_module_keys' to compile a subset of modules. "
    "Compilation can be time-consuming and might fail. Additional options provided are "
    "'compile_mode', 'compile_using_fullgraph', and 'compile_using_dynamic_shape_tracing'",
    "compile_mode": "One of {default, reduce-overhead, max-autotune}",
    "compile_using_fullgraph": "Whether it is ok to break model into several subgraphs",
    "compile_using_dynamic_shape_tracing": "Use dynamic shape tracing for compilation",
    "precision": "Floating-point precision for training with automatic mixed-precision.",
    "eval_precision": "Floating-point precision for inference with automatic mixed-precision.",
    "auto_mix_prec": "This flag enables training with automatic mixed-precision (deprecated).",
    "bfloat16_mix_prec": "This flag enables training with bfloat16 mixed-precision (deprecated).",
    "max_grad_norm": "Gradient norm will be clipped to this value, enter a negative value to disable.",
    "skip_nonfinite_grads": "Set the gradients to None if they are nonfinite (inf or nan).",
    "nonfinite_patience": "Max number of batches per epoch to skip if loss is nonfinite.",
    "noprogressbar": "This flag disables the data loop progressbars.",
    "ckpt_interval_minutes": "Amount of time between saving intra-epoch checkpoints "
    "in minutes. If non-positive, intra-epoch checkpoints are not saved.",
    "ckpt_interval_steps": "Save an intra-epoch checkpoint after this many steps. "
    "If non-positive, intra-epoch checkpoints are not saved.",
    "grad_accumulation_factor": "Number of batches to accumulate gradients before optimizer step",
    "optimizer_step_limit": "Number of optimizer steps to run. If not passed, all epochs are run.",
    "tqdm_colored_bar": "Enable colored progress-bar in tqdm. If this is false, tqdm shall use default colors.",
    "remove_vector_weight_decay": "Make vectors (e.g. norms and biases) a separate parameter group without weight_decay.",
    "profile_training": "If set to True, a profiler will be initiated and tensorboard logs will be generated. "
    "Please ensure you have installed the torch.TensorBoard profiler with 'pip install torch_tb_profiler'.",
    "profile_warmup": "Number of warmup steps before logging for the profiler.",
    "profile_steps": "Number of steps of logging for the profiler",
}


@dataclass(frozen=True)
class RunOptions:
    """
    Holds configuration options and runtime controls for SpeechBrain experiments.

    This dataclass encapsulates all tunable parameters and flags that affect
    the behavior of a SpeechBrain experiment, including device selection,
    debugging, distributed training, mixed-precision settings, checkpointing,
    profiling, and more. It provides default values for each option and can be
    constructed directly or via command-line argument parsing.

    Attributes
    ----------
    test_only : bool
        Run in evaluation-only mode, skipping training.
    debug : bool
        Enable debugging mode with reduced dataset size.
    debug_batches : int
        Number of batches to run in debug mode.
    debug_epochs : int
        Number of epochs to run in debug mode.
    debug_persistently : bool
        Keep debug data persistent (not using /tmp).
    device : str
        The device on which to run (e.g., "cpu", "cuda:0").
        Default of None may be handled with `speechbrain.utils.distributed.infer_device()`
    data_parallel_backend : bool
        Enable data parallel training.
    data_parallel_count : int
        Number of devices for data parallelism.
    distributed_backend : Literal["nccl", "gloo", "mpi"]
        Backend for distributed training.
    distributed_launch : bool
        Use distributed launch for training.
    find_unused_parameters : bool
        Detect unused parameters during distributed training.
    jit : bool
        Enable JIT compilation for modules.
    jit_module_keys : Optional[list]
        Module keys to compile with JIT.
    compile : bool
        Enable torch.compile for modules (if available).
    compile_module_keys : Optional[list]
        Module keys to compile with torch.compile.
    compile_mode : Literal["default", "reduce-overhead", "max-autotune"]
        Compilation mode.
    compile_using_fullgraph : bool
        Use fullgraph compilation.
    compile_using_dynamic_shape_tracing : bool
        Use dynamic shape tracing in compilation.
    precision : Literal["fp32", "fp16", "bf16"]
        Training precision.
    eval_precision : Literal["fp32", "fp16", "bf16"]
        Inference precision.
    auto_mix_prec : bool
        Enable automatic mixed-precision training.
    bfloat16_mix_prec : bool
        Enable bfloat16 mixed-precision training.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    skip_nonfinite_grads : bool
        Skip non-finite gradients.
    nonfinite_patience : int
        Number of tolerated non-finite batches per epoch.
    noprogressbar : bool
        Disable progress bars.
    ckpt_interval_minutes : int
        Minutes between intra-epoch checkpoints.
    ckpt_interval_steps : int
        Steps between intra-epoch checkpoints.
    grad_accumulation_factor : int
        Batches to accumulate before optimizer step.
    optimizer_step_limit : None or int
        Maximum number of optimizer steps.
    tqdm_colored_bar : bool
        Enable colored progress bars.
    tqdm_barcolor : dict of str
        Color mapping for progress bars.
    remove_vector_weight_decay : bool
        Separate parameter group for vectors without weight decay.
    profile_training : bool
        Enable profiling and tensorboard logging.
    profile_warmup : int
        Profiler warmup steps.
    profile_steps : int
        Profiler logging steps.
    log_config : None or str
        Path to logging configuration file.
    param_file : str
        Path to experiment parameter YAML file.
    overridden_args : dict
        The args that have been manually specified on the command line.
    """

    test_only: bool = False
    debug: bool = False
    debug_batches: int = 2
    debug_epochs: int = 2
    debug_persistently: bool = False
    device: Optional[str] = None
    data_parallel_backend: bool = False
    data_parallel_count: int = -1
    distributed_backend: Literal["nccl", "gloo", "mpi"] = "nccl"
    distributed_launch: bool = False
    find_unused_parameters: bool = False
    jit: bool = False
    jit_module_keys: Optional[list[str]] = None
    compile: bool = False
    compile_module_keys: Optional[list[str]] = None
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = (
        "default"
    )
    compile_using_fullgraph: bool = False
    compile_using_dynamic_shape_tracing: bool = False
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    eval_precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    auto_mix_prec: bool = False
    bfloat16_mix_prec: bool = False
    max_grad_norm: float = 5.0
    skip_nonfinite_grads: bool = False
    nonfinite_patience: int = 3
    noprogressbar: bool = False
    ckpt_interval_minutes: int = 0
    ckpt_interval_steps: int = 0
    grad_accumulation_factor: int = 1
    optimizer_step_limit: Optional[int] = None
    tqdm_colored_bar: bool = False
    tqdm_barcolor: Dict[str, str] = field(
        default_factory=lambda: {
            "train": "GREEN",
            "valid": "MAGENTA",
            "test": "CYAN",
        }
    )
    remove_vector_weight_decay: bool = False
    profile_training: bool = False
    profile_warmup: int = 5
    profile_steps: int = 5
    log_config: Optional[str] = None
    param_file: str = ""
    overridden_args: set = field(default_factory=set)

    def as_dict(self) -> Dict:
        """
        Converts the instance into a dictionary.

        Returns:
            Dict: A dictionary representation of the instance.
        """
        return asdict(self)

    def __getitem__(self, key):
        """Make items accessible via dict notation, to maintain backwards compat."""
        return getattr(self, key)

    @classmethod
    def from_dictionary(cls, args_dict):
        """Set experimental arguments from a dictionary."""

        # All the specified arguments are marked as overridden
        return cls(**{**args_dict, "overridden_args": set(args_dict.keys())})

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
        >>> filename, run_opts, overrides = RunOptions.from_command_line_args(argv)
        >>> filename
        'hyperparams.yaml'
        >>> run_opts["device"]
        'cuda:1'
        >>> overrides
        'seed: 10'
        """
        if arg_list is None:
            arg_list = sys.argv[1:]

        # Create a mapping of all possible argument names (including short forms)
        parser = cls._create_parser()
        arg_mapping = {}
        for action in parser._actions:
            if action.dest != "help":
                for opt in action.option_strings:
                    arg_mapping[opt] = action.dest

        # Parse and accept extra args to override yaml
        parsed_args, overrides = parser.parse_known_args(arg_list)
        overrides = cls._convert_to_yaml(overrides)

        # Go through arg list to see which were set
        # NOTE: Slight risk of collisions if an arg value matches an arg name
        overridden_args = {
            arg_mapping[arg] for arg in arg_list if arg in arg_mapping
        }

        # Add a record of which args were specified
        run_opts = cls(
            **{**vars(parsed_args), "overridden_args": overridden_args}
        )

        return run_opts.param_file, run_opts, overrides

    @staticmethod
    def _create_parser():
        """Sets up the parser using the options in HELP_TEXTS & defaults"""
        parser = argparse.ArgumentParser(
            description="Run a SpeechBrain experiment"
        )

        # A few arguments don't fit the standard format, write them out first
        parser.add_argument(
            "param_file",
            type=str,
            help="A hyperparameters file. Recipes use HyperPyYAML syntax.",
        )
        parser.add_argument(
            "--jit_module_keys",
            type=str,
            nargs="*",
            help="A list of keys in the 'modules' dict to jit-ify",
        )
        parser.add_argument(
            "--compile_module_keys",
            type=str,
            nargs="*",
            help="A list of keys in the 'modules' dict to compile using "
            "TorchInductor. If a module also has a JIT key specified, "
            "TorchInductor will take precedence when available.",
        )

        # These ones follow a standard format, pull default from class directly
        # NOTE: Assumes all options that can be specified on command-line have
        # an entry in the HELP_TEXTS dictionary at the top of this file.
        defaults = RunOptions().as_dict()
        for option in HELP_TEXTS.keys() & defaults.keys():
            default = defaults[option]
            kwargs = {"help": HELP_TEXTS[option]}

            # Booleans are flags
            if default is False:
                kwargs["action"] = "store_true"
            elif default is not None:
                kwargs["type"] = type(default)
                kwargs["default"] = default

            # Any options with "precision" in the name can only take these values
            if "precision" in option:
                kwargs["choices"] = ["fp32", "fp16", "bf16"]

            parser.add_argument(f"--{option}", **kwargs)

        return parser

    @staticmethod
    def _convert_to_yaml(overrides):
        """
        Convert a list of override arguments to a YAML formatted string.

        Arguments
        ---------
        overrides: list[str]
            A list of strings representing override arguments in the form '--arg=val'.

        Returns
        -------
        A YAML formatted string representing the overrides.
        """
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
