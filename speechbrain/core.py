"""Core SpeechBrain code for running experiments.

Authors
 * Peter Plantinga 2020, 2023
 * Abdel Heba 2020
 * Mirco Ravanelli 2020
 * Aku Rouhe 2021
 * Andreas Nautsch 2022
 * Sylvain de Langen 2023
 * Adel Moumen 2023, 2024
"""

import argparse
import inspect
import logging
import os
import pathlib
import shutil
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from types import SimpleNamespace

import torch
import yaml
from hyperpyyaml import resolve_references
from torch.nn import DataParallel as DP
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from tqdm.contrib import tqdm

import speechbrain as sb
from speechbrain.dataio.dataloader import LoopedLoader, SaveableDataLoader
from speechbrain.dataio.sampler import (
    DistributedSamplerWrapper,
    ReproducibleRandomSampler,
)
from speechbrain.utils.distributed import is_distributed_initialized
from speechbrain.utils.logger import get_logger
from speechbrain.utils.optimizers import rm_vector_weight_decay
from speechbrain.utils.profiling import prepare_profiler

sb.utils.quirks.apply_quirks()

logger = get_logger(__name__)
DEFAULT_LOG_CONFIG = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_CONFIG = os.path.join(DEFAULT_LOG_CONFIG, "log-config.yaml")
INTRA_EPOCH_CKPT_FLAG = "brain_intra_epoch_ckpt"
PYTHON_VERSION_MAJOR = 3
PYTHON_VERSION_MINOR = 8

# Arguments passed via the run opts dictionary
run_opt_defaults = {
    "test_only": False,
    "debug": False,
    "debug_batches": 2,
    "debug_epochs": 2,
    "debug_persistently": False,
    "device": "cpu",
    "data_parallel_backend": False,
    "distributed_backend": "nccl",
    "find_unused_parameters": False,
    "jit": False,
    "jit_module_keys": None,
    "compile": False,
    "compile_module_keys": None,
    "compile_mode": "reduce-overhead",
    "compile_using_fullgraph": False,
    "compile_using_dynamic_shape_tracing": False,
    "precision": "fp32",
    "eval_precision": "fp32",
    "auto_mix_prec": False,
    "bfloat16_mix_prec": False,
    "max_grad_norm": 5.0,
    "skip_nonfinite_grads": False,
    "nonfinite_patience": 3,
    "noprogressbar": False,
    "ckpt_interval_minutes": 0,
    "ckpt_interval_steps": 0,
    "grad_accumulation_factor": 1,
    "optimizer_step_limit": None,
    "tqdm_colored_bar": False,
    "tqdm_barcolor": {"train": "GREEN", "valid": "MAGENTA", "test": "CYAN"},
    "remove_vector_weight_decay": False,
    "profile_training": False,
    "profile_warmup": 5,
    "profile_steps": 5,
}


@dataclass
class AMPConfig:
    """Configuration for automatic mixed precision (AMP).

    Arguments
    ---------
    dtype : torch.dtype
        The dtype to use for AMP.
    """

    dtype: torch.dtype

    @classmethod
    def from_name(self, name):
        """Create an AMPConfig from a string name.

        Arguments
        ---------
        name : str
            The name of the AMPConfig to create.  Must be one of `fp32`,
            `fp16`, or `bf16`.

        Returns
        -------
        AMPConfig
            The AMPConfig corresponding to the name.
        """
        if name is None or name == "fp32":
            return AMPConfig(torch.float32)
        elif name == "fp16":
            return AMPConfig(torch.float16)
        elif name == "bf16":
            return AMPConfig(torch.bfloat16)
        else:
            raise ValueError(
                f"Specified autocast mode ({name}) incorrect, expected one of `fp32`, `fp16`, `bf16`."
            )


def create_experiment_directory(
    experiment_directory,
    hyperparams_to_save=None,
    overrides={},
    log_config=DEFAULT_LOG_CONFIG,
    save_env_desc=True,
):
    """Create the output folder and relevant experimental files.

    Arguments
    ---------
    experiment_directory : str
        The place where the experiment directory should be created.
    hyperparams_to_save : str
        A filename of a yaml file representing the parameters for this
        experiment. If passed, references are resolved, and the result is
        written to a file in the experiment directory called "hyperparams.yaml".
    overrides : dict
        A mapping of replacements made in the yaml file, to save in yaml.
    log_config : str
        A yaml filename containing configuration options for the logger.
    save_env_desc : bool
        If True, an environment state description is saved to the experiment
        directory, in a file called env.log in the experiment directory.
    """
    try:
        # all writing command must be done with the main_process
        if sb.utils.distributed.if_main_process():
            if not os.path.isdir(experiment_directory):
                os.makedirs(experiment_directory)

            # Write the parameters file
            if hyperparams_to_save is not None:
                hyperparams_filename = os.path.join(
                    experiment_directory, "hyperparams.yaml"
                )
                with open(hyperparams_to_save, encoding="utf-8") as f:
                    resolved_yaml = resolve_references(f, overrides)
                with open(hyperparams_filename, "w", encoding="utf-8") as w:
                    print("# Generated %s from:" % date.today(), file=w)
                    print("# %s" % os.path.abspath(hyperparams_to_save), file=w)
                    print("# yamllint disable", file=w)
                    shutil.copyfileobj(resolved_yaml, w)

            # Copy executing file to output directory
            module = inspect.getmodule(inspect.currentframe().f_back)
            if module is not None:
                callingfile = os.path.realpath(module.__file__)
                shutil.copy(callingfile, experiment_directory)

            # Log exceptions to output automatically
            log_file = os.path.join(experiment_directory, "log.txt")
            logger_overrides = {
                "handlers": {"file_handler": {"filename": log_file}}
            }
            sb.utils.logger.setup_logging(log_config, logger_overrides)
            sys.excepthook = _logging_excepthook

            # Log quirks again so that it makes it to the log file.
            # Quirks are applied way earlier, before logging is properly setup,
            # so this gives a chance to the user to see them, lowering surprise.
            sb.utils.quirks.log_applied_quirks()

            # Log beginning of experiment!
            logger.info("Beginning experiment!")
            logger.info(f"Experiment folder: {experiment_directory}")

            # Save system description:
            if save_env_desc:
                description_str = sb.utils.logger.get_environment_description()
                with open(
                    os.path.join(experiment_directory, "env.log"),
                    "w",
                    encoding="utf-8",
                ) as fo:
                    fo.write(description_str)
    finally:
        # wait for main_process if ddp is used
        sb.utils.distributed.ddp_barrier()


def _logging_excepthook(exc_type, exc_value, exc_traceback):
    """Interrupt exception raising to log the error."""
    logger.error("Exception:", exc_info=(exc_type, exc_value, exc_traceback))


def parse_arguments(arg_list=None):
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
    run_opts, overrides = parser.parse_known_args(arg_list)

    # Ignore items that are "None", they were not passed
    run_opts = {k: v for k, v in vars(run_opts).items() if v is not None}

    param_file = run_opts["param_file"]
    del run_opts["param_file"]

    overrides = _convert_to_yaml(overrides)

    # Checking that DataParallel use the right number of GPU
    if run_opts["data_parallel_backend"]:
        if torch.cuda.device_count() == 0:
            raise ValueError("You must have at least 1 GPU.")

    # force device arg to be the same as local_rank from torchrun
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and "cuda" in run_opts["device"]:
        run_opts["device"] = run_opts["device"][:-1] + str(local_rank)

    return param_file, run_opts, overrides


def _convert_to_yaml(overrides):
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


class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()


@sb.utils.checkpoints.register_checkpoint_hooks
class Brain:
    """Brain class abstracts away the details of data loops.

    The primary purpose of the `Brain` class is the implementation of
    the ``fit()`` method, which iterates epochs and datasets for the
    purpose of "fitting" a set of modules to a set of data.

    In order to use the ``fit()`` method, one should sub-class the ``Brain``
    class and override any methods for which the default behavior does not
    match the use case. For a simple use case (e.g., training a single model
    with a single dataset) the only methods that need to be overridden are:

    * ``compute_forward()``
    * ``compute_objectives()``

    The example below illustrates how overriding these two methods is done.

    For more complicated use cases, such as multiple modules that need to
    be updated, the following methods can be overridden:

    * ``fit_batch()``
    * ``evaluate_batch()``

    Arguments
    ---------
    modules : dict of str:torch.nn.Module pairs
        These modules are passed to the optimizer by default if they have
        trainable parameters, and will have ``train()``/``eval()`` called on them.
    opt_class : torch.optim class
        A torch optimizer constructor that takes only the list of
        parameters (e.g. a lambda or partial function definition). By default,
        this will be passed all modules in ``modules`` at the
        beginning of the ``fit()`` method. This behavior can be changed
        by overriding the ``configure_optimizers()`` method.
    hparams : dict
        Each key:value pair should consist of a string key and a hyperparameter
        that is used within the overridden methods. These will
        be accessible via an ``hparams`` attribute, using "dot" notation:
        e.g., self.hparams.model(x).
    run_opts : dict
        A set of options to change the runtime environment, including

        debug (bool)
            If ``True``, this will only iterate a few batches for all
            datasets, to ensure code runs without crashing.
        debug_batches (int)
            Number of batches to run in debug mode, Default ``2``.
        debug_epochs (int)
            Number of epochs to run in debug mode, Default ``2``.
            If a non-positive number is passed, all epochs are run.
        debug_persistently (bool)
            Keep data stored during debug mode (not using /tmp), Default ``False``.
        jit (bool)
            Enable to compile all modules using jit, Default ``False``.
        jit_module_keys (list of str)
            List of keys in ``modules`` that should be jit compiled.
        compile (bool)
            Enable to compile all modules using torch.compile, Default ``False``.
        compile_module_keys (list of str)
            List of keys in ``modules`` that should be compiled using
            ``torch.compile``. If ``torch.compile`` is unavailable,
            an error is raised.
        compile_mode (str)
            One of ``default``, ``reduce-overhead``, ``max-autotune``, Default ``reduce-overhead``.
        compile_using_fullgraph (bool)
            Whether it is ok to break model into several subgraphs, Default ``False``.
        compile_using_dynamic_shape_tracing (bool)
            Use dynamic shape tracing for compilation, Default ``False``.
        distributed_backend (str)
            One of ``nccl``, ``gloo``, ``mpi``.
        device (str)
            The location for performing computations.
        precision (str)
            One of ``fp32``, ``fp16``, ``bf16``.
        eval_precision (str)
            One of ``fp32``, ``fp16``, ``bf16``.
        auto_mix_prec (bool)
            If ``True``, automatic mixed-precision (fp16) is used.
            Activate it only with cuda. Note: this is a
            deprecated feature, and will be removed in the future.
        bfloat16_mix_prec (bool)
            If ``True``, automatic mixed-precision (bf16) is used.
            Activate it only with cuda. Note: this is a
            deprecated feature, and will be removed in the future.
        max_grad_norm (float)
            Default implementation of ``fit_batch()`` uses
            ``clip_grad_norm_`` with this value. Default: ``5``.
        skip_nonfinite_grads (bool)
            If ``True``, sets gradients to zero if they are non-finite
            (e.g., NaN, Inf). Default: ``False``.
        nonfinite_patience (int)
            Number of times to ignore non-finite losses before stopping.
            Default: ``3``.
        noprogressbar (bool)
            Whether to turn off progressbar when training. Default: ``False``.
        ckpt_interval_minutes (float)
            Amount of time between saving intra-epoch checkpoints,
            in minutes, default: ``15.0``. If non-positive, these are not saved.
        ckpt_interval_steps (int)
            Number of steps between saving intra-epoch checkpoints.
            If non-positive, these are not saved. Default: ``0``.


        Typically in a script this comes from ``speechbrain.parse_args``, which
        has different defaults than Brain. If an option is not defined here
        (keep in mind that parse_args will inject some options by default),
        then the option is also searched for in hparams (by key).
    checkpointer : speechbrain.Checkpointer
        By default, this will be used to load checkpoints, and will have the
        optimizer added to continue training if interrupted.

    Example
    -------
    >>> from torch.optim import SGD
    >>> class SimpleBrain(Brain):
    ...     def compute_forward(self, batch, stage):
    ...         return self.modules.model(batch[0])
    ...     def compute_objectives(self, predictions, batch, stage):
    ...         return torch.nn.functional.l1_loss(predictions, batch[0])
    >>> model = torch.nn.Linear(in_features=10, out_features=10)
    >>> brain = SimpleBrain({"model": model}, opt_class=lambda x: SGD(x, 0.1))
    >>> brain.fit(range(1), ([torch.rand(10, 10), torch.rand(10, 10)],))
    """

    def __init__(  # noqa: C901
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
    ):
        self.optimizers_dict = None
        self.opt_class = opt_class
        self.checkpointer = checkpointer

        for arg, default in run_opt_defaults.items():
            if run_opts is not None and arg in run_opts:
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: "
                        + arg
                        + " arg overridden by command line input to: "
                        + str(run_opts[arg])
                    )
                setattr(self, arg, run_opts[arg])
            else:
                # If any arg from run_opt_defaults exist in hparams and
                # not in command line args "run_opts"
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: " + arg + " arg from hparam file is used"
                    )
                    setattr(self, arg, hparams[arg])
                else:
                    setattr(self, arg, default)

        # Check Python version
        if not (
            sys.version_info.major == PYTHON_VERSION_MAJOR
            and sys.version_info.minor >= PYTHON_VERSION_MINOR
        ):
            logger.warning(
                "Detected Python "
                + str(sys.version_info.major)
                + "."
                + str(sys.version_info.minor)
                + ". We suggest using SpeechBrain with Python >="
                + str(PYTHON_VERSION_MAJOR)
                + "."
                + str(PYTHON_VERSION_MINOR)
            )

        # Assume `torchrun` was used if `RANK` and `LOCAL_RANK` are set
        self.distributed_launch = (
            os.environ.get("RANK") is not None
            and os.environ.get("LOCAL_RANK") is not None
        )

        if self.data_parallel_backend and self.distributed_launch:
            raise ValueError(
                "To use data_parallel backend, start your script with:\n\t"
                "python experiment.py hyperparams.yaml "
                "--data_parallel_backend=True\n"
                "To use DDP backend, start your script with:\n\t"
                "torchrun [args] experiment.py hyperparams.yaml"
            )

        if self.ckpt_interval_minutes > 0 and self.ckpt_interval_steps > 0:
            sys.exit(
                "The options `ckpt_interval_minutes` and `ckpt_interval_steps` "
                "are mutually exclusive. "
                "Please keep only one active per experiment run."
            )

        # Switch to the right context
        if self.device == "cuda":
            torch.cuda.set_device(0)
        elif "cuda" in self.device:
            torch.cuda.set_device(int(self.device[-1]))

        # Put modules on the right device, accessible with dot notation
        self.modules = torch.nn.ModuleDict(modules).to(self.device)

        # The next line ensures that both tensors marked as parameters and standard tensors,
        # such as those used in InputNormalization, are placed on the right device.
        for module in self.modules:
            if hasattr(self.modules[module], "to"):
                self.modules[module] = self.modules[module].to(self.device)

        # Make hyperparams available with dot notation too
        if hparams is not None:
            self.hparams = SimpleNamespace(**hparams)

        # Checkpointer should point at a temporary directory in debug mode
        if (
            self.debug
            and not self.debug_persistently
            and self.checkpointer is not None
            and hasattr(self.checkpointer, "checkpoints_dir")
        ):
            tempdir = tempfile.TemporaryDirectory()
            logger.info(
                "Since debug mode is active, switching checkpointer "
                f"output to temporary directory: {tempdir.name}"
            )
            self.checkpointer.checkpoints_dir = pathlib.Path(tempdir.name)

            # Keep reference to tempdir as long as checkpointer exists
            self.checkpointer.tempdir = tempdir

        # Sampler should be handled by `make_dataloader`
        # or if you provide a DataLoader directly, you can set
        # this.train_sampler = your_sampler
        # to have your_sampler.set_epoch() called on each epoch.
        self.train_sampler = None

        if self.auto_mix_prec:
            logger.warning(
                "The option `--auto_mix_prec` is deprecated and will be removed in the future. "
                "Please use `--precision=fp16` instead."
            )
            self.precision = "fp16"

        if self.bfloat16_mix_prec:
            logger.warning(
                "The option `--bfloat16_mix_prec` is deprecated and will be removed in the future. "
                "Please use `--precision=bf16` instead."
            )
            self.precision = "bf16"

        if self.device == "cpu" and (
            self.precision == "fp16" or self.eval_precision == "fp16"
        ):
            raise ValueError(
                "The option `--precision` or `--eval_precision` is set to fp16. "
                "This option is not yet supported on CPU. "
                "Please use `--precision=bf16` or `--eval_precision=bf16` instead "
                "to enable mixed precision on CPU."
            )

        gradscaler_enabled = self.precision == "fp16" and "cuda" in self.device
        if self.skip_nonfinite_grads and gradscaler_enabled:
            logger.warning(
                "The option `skip_nonfinite_grads` will be ignored "
                "because GradScaler is enabled and will automatically "
                "skip nonfinite gradients."
            )

        logger.info(
            f"Gradscaler enabled: {gradscaler_enabled}. Using precision: {self.precision}."
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=gradscaler_enabled)

        self.use_amp = False
        if self.device == "cpu" and self.precision == "bf16":
            self.use_amp = True
        elif "cuda" in self.device and self.precision in ["fp16", "bf16"]:
            self.use_amp = True

        if self.use_amp and self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "scaler", self.scaler, optional_load=True
            )

        # List parameter count for the user
        self.print_trainable_parameters()

        if self.distributed_launch:
            self.rank = int(os.environ["RANK"])
            if not is_distributed_initialized():
                if self.rank > 0:
                    raise ValueError(
                        " ================ WARNING ==============="
                        "Please add sb.ddp_init_group() into your exp.py"
                        "To use DDP backend, start your script with:\n\t"
                        "torchrun [args] experiment.py hyperparams.yaml"
                    )
                else:
                    logger.warning(
                        "To use DDP, please add "
                        "sb.utils.distributed.ddp_init_group() into your exp.py"
                    )
                    logger.info(
                        "Only the main process is alive, "
                        "all other subprocess were killed."
                    )

        # Prepare iterating variables
        self.avg_train_loss = 0.0
        self.step = 0
        self.optimizer_step = 0

        # Add this class to the checkpointer for intra-epoch checkpoints
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("brain", self)

        # Force default color for tqdm progressbar
        if not self.tqdm_colored_bar:
            self.tqdm_barcolor = dict.fromkeys(self.tqdm_barcolor, "")

        # Profiler setup
        self.profiler = None
        if self.profile_training:
            logger.info("Pytorch profiler has been activated.")
            self.tot_prof_steps = (self.profile_steps + self.profile_warmup) - 1
            self.profiler = prepare_profiler(
                self.profile_warmup,
                self.profile_steps,
                self.hparams.output_folder,
            )

    def print_trainable_parameters(self):
        """Prints the number of trainable parameters in the model."""
        total_trainable_params = 0
        total_parameters = 0
        for parameter in self.modules.parameters():
            total_parameters += parameter.numel()
            if parameter.requires_grad:
                total_trainable_params += parameter.numel()
        class_name = self.__class__.__name__
        if total_parameters == 0:
            logger.warning("The model has no parameters!")
            logger.info(
                f"{class_name} Model Statistics:\n"
                f"* Total Number of Trainable Parameters: {total_trainable_params}\n"
                f"* Total Number of Parameters: {total_parameters}\n"
                f"* Trainable Parameters represent {0:.2f}% of the total size."
            )
        elif total_trainable_params == 0:
            logger.warning("The model has no trainable parameters!")
            formatted_total_params = sb.utils.logger.format_order_of_magnitude(
                total_parameters
            )
            logger.info(
                f"{class_name} Model Statistics:\n"
                f"* Total Number of Trainable Parameters: {total_trainable_params}\n"
                f"* Total Number of Parameters: {formatted_total_params}\n"
                f"* Trainable Parameters represent {0:.4f}% of the total size."
            )
        else:
            percentage_trainable = (
                100 * total_trainable_params / total_parameters
            )
            formatted_trainable_params = (
                sb.utils.logger.format_order_of_magnitude(
                    total_trainable_params
                )
            )
            formatted_total_params = sb.utils.logger.format_order_of_magnitude(
                total_parameters
            )
            logger.info(
                f"{class_name} Model Statistics:\n"
                f"* Total Number of Trainable Parameters: {formatted_trainable_params}\n"
                f"* Total Number of Parameters: {formatted_total_params}\n"
                f"* Trainable Parameters represent {percentage_trainable:.4f}% of the total size."
            )

    def compute_forward(self, batch, stage):
        """Forward pass, to be overridden by sub-classes.

        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        torch.Tensor or torch.Tensors
            The outputs after all processing is complete.
            Directly passed to ``compute_objectives()``.
        """
        raise NotImplementedError
        return

    def compute_objectives(self, predictions, batch, stage):
        """Compute loss, to be overridden by sub-classes.

        Arguments
        ---------
        predictions : torch.Tensor or torch.Tensors
            The output tensor or tensors to evaluate.
            Comes directly from ``compute_forward()``.
        batch : torch.Tensor or tensors
            An element from the dataloader, including targets for comparison.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        loss : torch.Tensor
            A tensor with the computed loss.
        """
        raise NotImplementedError
        return

    def on_stage_start(self, stage, epoch=None):
        """Gets called when a stage starts.

        Useful for defining class variables used during the stage.

        Arguments
        ---------
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        epoch : int
            The current epoch count.
        """
        pass

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage.

        Useful for computing stage statistics, saving checkpoints, etc.

        Arguments
        ---------
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        stage_loss : float
            The average loss over the completed stage.
        epoch : int
            The current epoch count.
        """
        pass

    def make_dataloader(
        self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs
    ):
        """Creates DataLoaders for Datasets.

        This is used by ``fit()`` and ``evaluate()`` if they just receive
        Datasets.

        Alternatively, this can be called from outside the Brain subclass.
        In that case, the DataLoader should be passed to ``fit()`` in place
        of the dataset.

        The Stage.TRAIN DataLoader is handled specially. It has extra args for
        shuffle and drop_last. In DDP a DistributedSampler is created (unless
        the dataset is an IterableDataset).

        NOTE
        ----
        Some important DataLoader arguments are passed via **loader_kwargs,
        e.g., batch_size, num_workers, pin_memory.

        NOTE
        ----
        By default, ``evaluate()`` specifies ckpt_prefix=None to stop the test
        DataLoader being added to the checkpointer. If you need to add a
        recoverable after saving checkpoints (e.g., at test time, after
        checkpointing the training), and still be able to recover reasonably,
        you should probably specify ``allow_partial_load=True``.

        Arguments
        ---------
        dataset : Dataset
            A set of data to use to create data loader. If the Dataset is a
            DynamicItemDataset, PaddedBatch is used as the default collate_fn,
            unless specified in loader_kwargs.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        ckpt_prefix : str, None
            Prefix to use for SaveableDataLoader Checkpoint name. The Stage
            name is added to this to create the full key. Set to None to not
            save the DataLoader.
        **loader_kwargs : dict
            Additional keyword arguments to the DataLoader.
            E.g., batch_size, num_workers, pin_memory.

        Returns
        -------
        DataLoader for the input dataset
        """
        # TRAIN stage is handled specially.
        if stage == sb.Stage.TRAIN:
            loader_kwargs = self._train_loader_specifics(dataset, loader_kwargs)
        # This commented-out code block is useful when one can ensure
        # metric reporting is DDP-valid for VALID & EVAL datasets.
        # elif self.distributed_launch:
        #     loader_kwargs = sb.dataio.dataloader.distributed_loader_specifics(
        #         self.distributed_launch, self.rank, dataset, loader_kwargs
        #     )
        dataloader = sb.dataio.dataloader.make_dataloader(
            dataset, **loader_kwargs
        )

        if (
            self.checkpointer is not None
            and ckpt_prefix is not None
            and (
                isinstance(dataloader, SaveableDataLoader)
                or isinstance(dataloader, LoopedLoader)
            )
        ):
            ckpt_key = ckpt_prefix + stage.name
            self.checkpointer.add_recoverable(ckpt_key, dataloader)
        return dataloader

    def _train_loader_specifics(self, dataset, loader_kwargs):
        sampler = loader_kwargs.get("sampler", None)
        # Shuffling should really only matter for the train stage. Shuffling
        # will also lead to more padding in batches if the order was otherwise
        # sorted by length.
        shuffle = loader_kwargs.get("shuffle", False)
        if shuffle and not self.distributed_launch:
            if sampler is not None:
                raise ValueError(
                    "Cannot specify both shuffle=True"
                    "and a sampler in loader_kwargs"
                )
            seed = os.environ.get("SB_GLOBAL_SEED", 563375142)
            sampler = ReproducibleRandomSampler(dataset, seed=seed)
            self.train_sampler = sampler
            loader_kwargs["sampler"] = self.train_sampler
            # Delete the shuffle flag, since you cannot specify both a sampler and
            # shuffling:
            del loader_kwargs["shuffle"]

        # Possibly make a DistributedSampler or a wrapper for some other sampler
        if self.distributed_launch and not isinstance(dataset, IterableDataset):
            # sort or not
            if hasattr(self.hparams, "sorting"):
                shuffle_ddp = (
                    self.hparams.sorting == "random"
                )  # False if 'ascending' or 'descending'
            else:
                shuffle_ddp = True

            drop_last = loader_kwargs.get("drop_last", False)
            # num_replicas arg is equal to world_size
            # and retrieved automatically within
            # DistributedSampler obj.
            if sampler is not None:
                self.train_sampler = DistributedSamplerWrapper(
                    sampler,
                    rank=self.rank,
                    drop_last=drop_last,
                    shuffle=shuffle,
                )

                # with DistributedSamplerWrapper, one must disable shuffling for dataloader
                loader_kwargs["shuffle"] = False
                loader_kwargs["sampler"] = self.train_sampler
            elif loader_kwargs.get("batch_sampler") is None:
                # no sampler and batch-sampler
                self.train_sampler = DistributedSampler(
                    dataset,
                    rank=self.rank,
                    shuffle=shuffle_ddp,
                    drop_last=drop_last,
                )

                # with DistributedSamplerWrapper, one must disable shuffling for dataloader
                loader_kwargs["shuffle"] = False
                loader_kwargs["sampler"] = self.train_sampler
            else:  # batch_sampler was specified
                self.train_sampler = DistributedSamplerWrapper(
                    loader_kwargs.get("batch_sampler", None),
                    rank=self.rank,
                    shuffle=shuffle_ddp,
                )
                loader_kwargs["batch_sampler"] = self.train_sampler
        elif self.distributed_launch and isinstance(dataset, IterableDataset):
            logger.warning(
                "Cannot automatically solve distributed sampling "
                "for IterableDataset."
            )
        return loader_kwargs

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit/compiled modules
        # cannot be pickled.
        self._compile()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible()

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).

        The default implementation of this method depends on an optimizer
        class being passed at initialization that takes only a list
        of parameters (e.g., a lambda or a partial function definition).
        This creates a single optimizer that optimizes all trainable params.

        Override this class if there are multiple optimizers.
        """

        all_params = self.modules.parameters()

        if self.opt_class is not None:
            if self.remove_vector_weight_decay:
                all_params = rm_vector_weight_decay(self.modules)

            self.optimizer = self.opt_class(all_params)

            self.optimizers_dict = {"opt_class": self.optimizer}

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        """Sets the gradients of all optimized ``torch.Tensor``s to zero
        if ``set_to_none=False`` (default) or to None otherwise.

        Setting gradients to None should save the memory, e.g.
        during ``evaluate()`` and thus larger batch might be used.
        """
        if self.optimizers_dict is not None:
            for opt in self.freeze_optimizers(self.optimizers_dict).values():
                opt.zero_grad(set_to_none=set_to_none)
        elif self.opt_class is not None:
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Gets called at the beginning of ``evaluate()``

        Default implementation loads the best-performing checkpoint for
        evaluation, based on stored metrics.

        Arguments
        ---------
        max_key : str
            Key to use for finding best checkpoint (higher is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        min_key : str
            Key to use for finding best checkpoint (lower is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        """

        # Recover best checkpoint for evaluation
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                max_key=max_key, min_key=min_key
            )

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``
        * ``optimizers_step()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        amp = AMPConfig.from_name(self.precision)
        should_step = (self.step % self.grad_accumulation_factor) == 0
        self.on_fit_batch_start(batch, should_step)

        with self.no_sync(not should_step):
            if self.use_amp:
                with torch.autocast(
                    dtype=amp.dtype, device_type=torch.device(self.device).type
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(
                        outputs, batch, sb.Stage.TRAIN
                    )
            else:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            scaled_loss = self.scaler.scale(
                loss / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_loss)
            scaled_loss.backward()

        if should_step:
            self.optimizers_step()

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def check_loss_isfinite(self, loss):
        """Check if the loss is finite.

        If the loss is not finite, log a helpful message and increment the `nonfinite_count`.
        If the `nonfinite_count` exceeds the `--nonfinite_patience` threshold, stop the training
        and raise an error.

        This check is particularly useful when the loss becomes NaN or inf, while the
        parameters and gradients remain finite. It helps prevent getting stuck in an
        infinite loop during training.

        Arguments
        ---------
        loss : tensor
            The loss tensor after ``backward()`` has been called but
            before the optimizers ``step()``.
        """
        if not torch.isfinite(loss):
            self.nonfinite_count += 1

            # Check if patience is exhausted
            if self.nonfinite_count > self.nonfinite_patience:
                raise ValueError(
                    "Loss is not finite and patience is exhausted. "
                    "To debug, wrap `fit()` with "
                    "autograd's `detect_anomaly()`, e.g.\n\nwith "
                    "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
                )
            else:
                logger.warning("Patience not yet exhausted.")

    def check_gradients(self):
        """Checks if the gradients are finite. If not, it will emit a warning and set them to zero."""
        for param in self.modules.parameters():
            if param.requires_grad and param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    param.grad = None
                    logger.warning(
                        f"Gradients {param.name} contain NaN or Inf. Setting to None."
                    )

    def freeze_optimizers(self, optimizers):
        """By default, this method returns the passed optimizers.
        Override this method if you want to freeze some optimizers
        during training. To do so, return a of active optimizers.
        """
        return optimizers

    def optimizers_step(self):
        """Performs a step of gradient descent on the optimizers. This method is called every
        ``grad_accumulation_factor`` steps."""
        # 1. get the valid optimizers, i.e., the ones that are not frozen during this step
        if self.optimizers_dict is not None:
            valid_optimizers = self.freeze_optimizers(self.optimizers_dict)
        elif self.opt_class is not None:
            # if valid_optimizers is not defined which could happen if a user is using an old
            # init_optimizers() method, then we assume that the only valid optimizer is
            # self.optimizer (which is the default behavior).
            valid_optimizers = {"optimizer": self.optimizer}
        else:
            # Note: in some cases you might want to only compute gradients statistics and
            # you do not need to call the optimizers.step() method. In this case, you can
            # simply return from this method and skip the rest of the code.
            return

        # 2. unscale the gradients of the valid optimizers
        for opt in valid_optimizers.values():
            self.scaler.unscale_(opt)

        # 3. clip gradients
        # We are clipping this way because clipping on self.modules.parameters()
        # can leads to NaN/Inf gradients norm as doing the concatenation
        # of all parameters in a single vector can lead to overflow/underflow.
        for opt in valid_optimizers.values():
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], self.max_grad_norm
            )

        # Note: no need to activate this flag if you are in fp16
        # since GradScaler is automatically handling the nonfinite gradients
        if not self.scaler.is_enabled() and self.skip_nonfinite_grads:
            self.check_gradients()

        # 4. step the valid optimizers
        # If the scaler is disable, it simply calls optimizer.step()
        for opt in valid_optimizers.values():
            self.scaler.step(opt)

        self.scaler.update()

        for opt in valid_optimizers.values():
            opt.zero_grad(set_to_none=True)

        self.optimizer_step += 1

    def on_fit_batch_start(self, batch, should_step):
        """Called at the beginning of ``fit_batch()``.

        This method is not called under the AMP context manager. Do not assume
        automatic casting of the input batch to a lower precision (e.g. fp16).

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        pass

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after ``fit_batch()``.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        outputs : list or dictionary of torch.Tensors
            Returned value of compute_forward().
        loss : torch.Tensor
            Returned value of compute_objectives().
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        pass

    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        amp = AMPConfig.from_name(self.eval_precision)
        if self.use_amp:
            with torch.autocast(
                dtype=amp.dtype, device_type=torch.device(self.device).type
            ):
                out = self.compute_forward(batch, stage=stage)
                loss = self.compute_objectives(out, batch, stage=stage)
        else:
            out = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu()

    def _fit_train(self, train_set, epoch, enable):
        # Training stage
        self.on_stage_start(Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(
            self.train_sampler, "set_epoch"
        ):
            self.train_sampler.set_epoch(epoch)

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        steps_since_ckpt = 0
        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            if self.profiler is not None:
                self.profiler.start()
            for batch in t:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                steps_since_ckpt += 1
                loss = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )
                t.set_postfix(train_loss=self.avg_train_loss)

                if self.profiler is not None:
                    self.profiler.step()
                    if self.profiler.step_num > self.tot_prof_steps:
                        logger.info(
                            "The profiler finished, training is stopped."
                        )
                        self.profiler.stop()
                        quit()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

                if self._should_save_intra_epoch_ckpt(
                    last_ckpt_time, steps_since_ckpt
                ):
                    # Checkpointer class will handle running this on main only
                    self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()
                    steps_since_ckpt = 0

        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        self.step = 0

    def _should_save_intra_epoch_ckpt(self, last_ckpt_time, steps_since_ckpt):
        """Determines if an intra-epoch checkpoint should be saved.

        Returns True if there's a checkpointer and time or steps has exceeded limit.
        """
        if self.checkpointer is None:
            return False

        # Return early if mid-epoch checkpoints are disabled to avoid sync
        if self.ckpt_interval_minutes <= 0 and self.ckpt_interval_steps <= 0:
            return False

        # Check if we've run for the requested amount of time
        elapsed_minutes = (time.time() - last_ckpt_time) / 60.0
        decision = 0 < self.ckpt_interval_minutes < elapsed_minutes

        # Save after requested # of steps
        decision = decision or 0 < self.ckpt_interval_steps <= steps_since_ckpt

        # If the program is not distributed, just return
        if not is_distributed_initialized():
            return decision

        # Otherwise, broadcast decision to all processes from main (rank 0)
        # This solves synchronization issues where main gets a different
        # timing result than the other processes.
        else:
            broadcast_list = [decision]
            torch.distributed.broadcast_object_list(broadcast_list, src=0)
            return broadcast_list[0]

    def _fit_valid(self, valid_set, epoch, enable):
        # Validation stage
        if valid_set is not None:
            self.on_stage_start(Stage.VALID, epoch)
            self.modules.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    valid_set,
                    dynamic_ncols=True,
                    disable=not enable,
                    colour=self.tqdm_barcolor["valid"],
                ):
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=Stage.VALID)
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                self.step = 0
                self.on_stage_end(Stage.VALID, avg_valid_loss, epoch)

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.

        Returns
        -------
        None
        """
        if self.test_only:
            logger.info(
                "Test only mode, skipping training and validation stages."
            )
            return

        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break

    @property
    def _optimizer_step_limit_exceeded(self):
        return (
            self.optimizer_step_limit is not None
            and self.optimizer_step >= self.optimizer_step_limit
        )

    def _save_intra_epoch_ckpt(self):
        """Saves a CKPT with specific intra-epoch flag."""
        self.checkpointer.save_and_keep_only(
            end_of_epoch=False,
            num_to_keep=1,
            ckpt_predicate=lambda c: INTRA_EPOCH_CKPT_FLAG in c.meta,
            meta={INTRA_EPOCH_CKPT_FLAG: True},
            verbosity=logging.DEBUG,
        )

    def _compile(self):
        """Compile requested modules with either JIT or TorchInductor."""
        compile_available = hasattr(torch, "compile")

        if not compile_available and self.compile_module_keys is not None:
            raise ValueError(
                "'compile_module_keys' specified, but this install of PyTorch "
                "seems to be too old to support it."
            )
        # Modules to compile with torch.compile
        compile_module_keys = set()
        if self.compile:
            if self.compile_module_keys is None:
                compile_module_keys = set(self.modules)
            else:
                compile_module_keys = set(self.compile_module_keys)
                logger.warning(
                    "--compile and --compile_module_keys are both specified. "
                    "Only modules specified in --compile_module_keys will be compiled."
                )

        # Modules to compile with jit
        jit_module_keys = set()
        if self.jit:
            if self.jit_module_keys is None:
                jit_module_keys = set(self.modules)
            else:
                jit_module_keys = set(self.jit_module_keys)
                logger.warning(
                    "--jit and --jit_module_keys are both specified. "
                    "Only modules specified in --jit_module_keys will be compiled."
                )

        # find missing keys
        for name in compile_module_keys | jit_module_keys:
            if name not in self.modules:
                raise ValueError(
                    f"module {name} is not defined in your hparams file."
                )

        # try 'torch.compile', remove successful compiles from JIT list
        for name in compile_module_keys:
            try:
                module = torch.compile(
                    self.modules[name],
                    mode=self.compile_mode,
                    fullgraph=self.compile_using_fullgraph,
                    dynamic=self.compile_using_dynamic_shape_tracing,
                )
            except Exception as e:
                logger.warning(
                    f"'{name}' in 'compile_module_keys' failed to compile "
                    f"and will be skipped (may fallback onto JIT, if "
                    f"specified): {e}"
                )
                continue

            self.modules[name] = module.to(self.device)
            jit_module_keys.discard(name)

        for name in jit_module_keys:
            module = torch.jit.script(self.modules[name])
            self.modules[name] = module.to(self.device)

    def _wrap_distributed(self):
        """Wrap modules with distributed wrapper when requested."""
        if not self.distributed_launch and not self.data_parallel_backend:
            return
        elif self.distributed_launch:
            for name, module in self.modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = SyncBatchNorm.convert_sync_batchnorm(module)
                    if self.distributed_backend == "gloo":
                        module = DDP(
                            module,
                            device_ids=None,
                            find_unused_parameters=self.find_unused_parameters,
                        )
                    else:
                        module = DDP(
                            module,
                            device_ids=[self.device],
                            find_unused_parameters=self.find_unused_parameters,
                        )
                    self.modules[name] = module
        else:
            # data_parallel_backend
            for name, module in self.modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = DP(module)
                    self.modules[name] = module

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not enable,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.on_stage_end(Stage.TEST, avg_test_loss, None)
        self.step = 0
        return avg_test_loss

    def update_average(self, loss, avg_loss):
        """Update running average of the loss.

        Arguments
        ---------
        loss : torch.tensor
            detached loss, a single float value.
        avg_loss : float
            current running average.

        Returns
        -------
        avg_loss : float
            The average loss.
        """
        if torch.isfinite(loss):
            avg_loss -= avg_loss / self.step
            avg_loss += float(loss) / self.step
        return avg_loss

    @contextmanager
    def no_sync(self, use=True):
        """Copies pytorch's implementation for doing no_sync across all modules.

        Explanation: nn.module.no_sync() is a context manager for when one does
        not want to sync gradients, which happens when using both DDP and gradient accumulation.
        Speechbrain brain's class can contain multiple modules and calling no_sync on these
        individually would be very awkward, therefore this contextmanager exists.

        Arguments
        ---------
        use : bool
            If set to `False` will still sync gradients, useful to make behavior toggleable.

        Yields
        ------
        None
        """
        if use:
            old_values_list = []
            for module in self.modules.values():
                if not hasattr(module, "require_backward_grad_sync"):
                    # if not using DDP
                    continue
                old_values_list.append(module.require_backward_grad_sync)
                module.require_backward_grad_sync = False
            yield
            i = 0
            for module in self.modules.values():
                if not hasattr(module, "require_backward_grad_sync"):
                    continue
                module.require_backward_grad_sync = old_values_list[i]
                i += 1
        else:
            yield

    @sb.utils.checkpoints.mark_as_saver
    def _save(self, path):
        save_dict = {
            "step": self.step,
            "avg_train_loss": self.avg_train_loss,
            "optimizer_step": self.optimizer_step,
        }
        with open(path, "w", encoding="utf-8") as w:
            w.write(yaml.dump(save_dict))

    @sb.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch):
        del end_of_epoch
        with open(path, encoding="utf-8") as f:
            save_dict = yaml.safe_load(f)
        self.step = save_dict["step"]
        self.avg_train_loss = save_dict["avg_train_loss"]
        # Ensure compatibility with checkpoints from before optimizer_step:
        if "optimizer_step" not in save_dict:
            clsname = self.__class__.__name__
            MSG = f"'optimizer_step' not found in {clsname} checkpoint."
            MSG += " Using the saved 'step' value (BACKWARDS COMPATIBILITY)"
            warnings.warn(MSG)
            self.optimizer_step = self.step
        else:
            self.optimizer_step = save_dict["optimizer_step"]
