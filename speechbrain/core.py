"""Core SpeechBrain code for running experiments.

Authors
 * Peter Plantinga 2020
 * Abdel Heba 2020
 * Mirco Ravanelli 2020
 * Aku Rouhe 2021
 * Andreas Nautsch 2022
"""

import os
import sys
import yaml
import time
import torch
import shutil
import logging
import inspect
import pathlib
import argparse
import tempfile
import warnings
from contextlib import contextmanager
import speechbrain as sb
from datetime import date
from enum import Enum, auto
from tqdm.contrib import tqdm
from types import SimpleNamespace
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader
from torch.nn import DataParallel as DP
from torch.utils.data import IterableDataset
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from hyperpyyaml import resolve_references
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.optimizers import rm_vector_weight_decay
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import DistributedSamplerWrapper
from speechbrain.dataio.sampler import ReproducibleRandomSampler

logger = logging.getLogger(__name__)
DEFAULT_LOG_CONFIG = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_CONFIG = os.path.join(DEFAULT_LOG_CONFIG, "log-config.yaml")
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
INTRA_EPOCH_CKPT_FLAG = "brain_intra_epoch_ckpt"
PYTHON_VERSION_MAJOR = 3
PYTHON_VERSION_MINOR = 7


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
                with open(hyperparams_to_save) as f:
                    resolved_yaml = resolve_references(f, overrides)
                with open(hyperparams_filename, "w") as w:
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

            # Log beginning of experiment!
            logger.info("Beginning experiment!")
            logger.info(f"Experiment folder: {experiment_directory}")

            # Save system description:
            if save_env_desc:
                description_str = sb.utils.logger.get_environment_description()
                with open(
                    os.path.join(experiment_directory, "env.log"), "w"
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
    # if use_env = False in torch.distributed.lunch then local_rank arg is given
    parser.add_argument(
        "--local_rank",
        "--local-rank",  # alias required for PyTorch 2.x
        type=int,
        help="Rank on local machine",
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
        "--distributed_launch",
        default=False,
        action="store_true",
        help="This flag enables training with DDP. Assumes script run with "
        "`torch.distributed.launch`",
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
        "--jit_module_keys",
        type=str,
        nargs="*",
        help="A list of keys in the 'modules' dict to jitify",
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

    # For DDP, the device args must equal to local_rank used by
    # torch.distributed.launch. If run_opts["local_rank"] exists,
    # use os.environ["LOCAL_RANK"]
    local_rank = None
    if "local_rank" in run_opts:
        local_rank = run_opts["local_rank"]
    else:
        if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] != "":
            local_rank = int(os.environ["LOCAL_RANK"])

    # force device arg to be the same as local_rank from torch.distributed.lunch
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
        jit_module_keys (list of str)
            List of keys in ``modules`` that should be jit compiled.
        distributed_backend (str)
            One of ``nccl``, ``gloo``, ``mpi``.
        device (str)
            The location for performing computations.
        auto_mix_prec (bool)
            If ``True``, automatic mixed-precision is used.
            Activate it only with cuda.
        max_grad_norm (float)
            Default implementation of ``fit_batch()`` uses
            ``clip_grad_norm_`` with this value. Default: ``5``.
        nonfinite_patience (int)
            Number of times to ignore non-finite losses before stopping.
            Default: ``3``.
        noprogressbar (bool)
            Whether to turn off progressbar when training. Default: ``False``.
        ckpt_interval_minutes (float)
            Amount of time between saving intra-epoch checkpoints,
            in minutes, default: ``15.0``. If non-positive, these are not saved.

        Typically in a script this comes from ``speechbrain.parse_args``, which
        has different defaults than Brain. If an option is not defined here
        (keep in mind that parse_args will inject some options by default),
        then the option is also searched for in hparams (by key).
    checkpointer : speechbrain.Checkpointer
        By default, this will be used to load checkpoints, and will have the
        optimizer added to continue training if interrupted.
    profiler : torch.profiler.profile
        Context manager for profiling and benchmarking of training/inference steps.
        Default: ``None`` (skip profiling).

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
        profiler=None,
    ):
        self.opt_class = opt_class
        self.checkpointer = checkpointer
        self.profiler = profiler

        # Arguments passed via the run opts dictionary
        run_opt_defaults = {
            "debug": False,
            "debug_batches": 2,
            "debug_epochs": 2,
            "debug_persistently": False,
            "device": "cpu",
            "data_parallel_backend": False,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "find_unused_parameters": False,
            "jit_module_keys": None,
            "auto_mix_prec": False,
            "bfloat16_mix_prec": False,
            "max_grad_norm": 5.0,
            "nonfinite_patience": 3,
            "noprogressbar": False,
            "ckpt_interval_minutes": 0,
            "grad_accumulation_factor": 1,
            "optimizer_step_limit": None,
            "tqdm_colored_bar": False,
            "tqdm_barcolor": {
                "train": "GREEN",
                "valid": "MAGENTA",
                "test": "CYAN",
            },
            "remove_vector_weight_decay": False,
        }

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

        if self.data_parallel_backend and self.distributed_launch:
            sys.exit(
                "To use data_parallel backend, start your script with:\n\t"
                "python experiment.py hyperparams.yaml "
                "--data_parallel_backend=True"
                "To use DDP backend, start your script with:\n\t"
                "python -m torch.distributed.lunch [args]\n"
                "experiment.py hyperparams.yaml --distributed_launch=True "
                "--distributed_backend=nccl"
            )

        # Switch to the right context
        if self.device == "cuda":
            torch.cuda.set_device(0)
        elif "cuda" in self.device:
            torch.cuda.set_device(int(self.device[-1]))

        # Put modules on the right device, accessible with dot notation
        self.modules = torch.nn.ModuleDict(modules).to(self.device)

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

        # Automatic mixed precision init
        if self.auto_mix_prec:
            self.scaler = torch.cuda.amp.GradScaler()
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("scaler", self.scaler)

        # List parameter count for the user
        total_params = sum(
            p.numel() for p in self.modules.parameters() if p.requires_grad
        )
        if total_params > 0:
            clsname = self.__class__.__name__
            fmt_num = sb.utils.logger.format_order_of_magnitude(total_params)
            logger.info(f"{fmt_num} trainable parameters in {clsname}")

        if self.distributed_launch:
            self.rank = int(os.environ["RANK"])
            if not torch.distributed.is_initialized():
                if self.rank > 0:
                    sys.exit(
                        " ================ WARNING ==============="
                        "Please add sb.ddp_init_group() into your exp.py"
                        "To use DDP backend, start your script with:\n\t"
                        "python -m torch.distributed.launch [args]\n\t"
                        "experiment.py hyperparams.yaml "
                        "--distributed_launch=True --distributed_backend=nccl"
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
        self.valid_step = 0
        self.optimizer_step = 0

        # Add this class to the checkpointer for intra-epoch checkpoints
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("brain", self)

        # Force default color for tqdm progrressbar
        if not self.tqdm_colored_bar:
            self.tqdm_barcolor = dict.fromkeys(self.tqdm_barcolor, "")

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
        torch.Tensor or Tensors
            The outputs after all processing is complete.
            Directly passed to ``compute_objectives()``.
        """
        raise NotImplementedError

    def compute_objectives(self, predictions, batch, stage):
        """Compute loss, to be overridden by sub-classes.

        Arguments
        ---------
        predictions : torch.Tensor or Tensors
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
            sampler = ReproducibleRandomSampler(dataset)
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
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device)
            )

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

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        """Sets the gradients of all optimized ``torch.Tensor``s to zero
        if ``set_to_none=False`` (default) or to None otherwise.

        Setting gradients to None should save the memory, e.g.
        during ``evaluate()`` and thus larger batch might be used.
        """
        if hasattr(self, "optimizer"):
            self.optimizer.zero_grad(set_to_none)

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
                max_key=max_key,
                min_key=min_key,
                device=torch.device(self.device),
            )

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

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
        valid_loss = False

        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(device_type=torch.device(self.device).type):
                outputs = self.compute_forward(batch, Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, Stage.TRAIN)

            if self.check_gradients(loss):
                valid_loss = True
                self.valid_step += 1

            should_step = self.valid_step % self.grad_accumulation_factor == 0
            if valid_loss:
                with self.no_sync(not should_step):
                    self.scaler.scale(
                        loss / self.grad_accumulation_factor
                    ).backward()
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.zero_grad()
                    self.optimizer_step += 1
        else:
            if self.bfloat16_mix_prec:
                with torch.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=torch.bfloat16,
                ):
                    outputs = self.compute_forward(batch, Stage.TRAIN)
                    loss = self.compute_objectives(outputs, batch, Stage.TRAIN)
            else:
                outputs = self.compute_forward(batch, Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, Stage.TRAIN)

            if self.check_gradients(loss):
                valid_loss = True
                self.valid_step += 1

            should_step = self.valid_step % self.grad_accumulation_factor == 0
            if valid_loss:
                with self.no_sync(not should_step):
                    (loss / self.grad_accumulation_factor).backward()
                if should_step:
                    self.optimizer.step()
                    self.zero_grad()
                    self.optimizer_step += 1

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after ``fit_batch()``, meant for calculating and logging metrics.

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

    def check_gradients(self, loss):
        """Check if gradients are finite and not too large.
        Automatically clips large gradients.

        Arguments
        ---------
        loss : tensor
            The loss tensor after ``backward()`` has been called but
            before the optimizers ``step()``.

        Returns
        -------
        bool
            Whether or not the optimizer step should be carried out.
        """
        if not torch.isfinite(loss):
            self.nonfinite_count += 1

            # Print helpful debug info
            logger.warning(f"Loss is {loss}.")
            for p in self.modules.parameters():
                if not torch.isfinite(p).all():
                    logger.warning("Parameter is not finite: " + str(p))

            # Check if patience is exhausted
            if self.nonfinite_count > self.nonfinite_patience:
                raise ValueError(
                    "Loss is not finite and patience is exhausted. "
                    "To debug, wrap `fit()` with "
                    "autograd's `detect_anomaly()`, e.g.\n\nwith "
                    "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
                )
            else:
                logger.warning(
                    "Patience not yet exhausted, ignoring this batch."
                )
                return False

        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.modules.parameters()), self.max_grad_norm
            )

        return True

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
        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            for batch in t:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                loss = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )
                t.set_postfix(train_loss=self.avg_train_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

                if (
                    self.checkpointer is not None
                    and self.ckpt_interval_minutes > 0
                    and time.time() - last_ckpt_time
                    >= self.ckpt_interval_minutes * 60.0
                ):
                    # This should not use run_on_main, because that
                    # includes a DDP barrier. That eventually leads to a
                    # crash when the processes'
                    # time.time() - last_ckpt_time differ and some
                    # processes enter this block while others don't,
                    # missing the barrier.
                    if sb.utils.distributed.if_main_process():
                        self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()

        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        self.step = 0
        self.valid_step = 0

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

                    # Profile only if desired (steps allow the profiler to know when all is warmed up)
                    if self.profiler is not None:
                        if self.profiler.record_steps:
                            self.profiler.step()

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                # Only run validation "on_stage_end" on main process
                self.step = 0
                run_on_main(
                    self.on_stage_end,
                    args=[Stage.VALID, avg_valid_loss, epoch],
                )

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
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """
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

    def _compile_jit(self):
        """Compile requested modules with ``torch.jit.script``."""
        if self.jit_module_keys is None:
            return

        for name in self.jit_module_keys:
            if name not in self.modules:
                raise ValueError(
                    "module" + name + " is not defined in your hparams file."
                )
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
                disable=not progressbar,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end, args=[Stage.TEST, avg_test_loss, None]
            )
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
            If set to `False` will still sync gradients, useful to make behaviour togglable.
        """
        if use:
            old_values_list = []
            for module in self.modules.values():
                if not hasattr(module, "require_backward_grad_sync"):
                    # if not using DDP
                    break
                old_values_list.append(module.require_backward_grad_sync)
                module.require_backward_grad_sync = False
            yield
            for module, old_value in zip(
                self.modules.values(), old_values_list
            ):
                if not hasattr(module, "require_backward_grad_sync"):
                    break
                module.require_backward_grad_sync = old_value
        else:
            yield

    @sb.utils.checkpoints.mark_as_saver
    def _save(self, path):
        save_dict = {
            "step": self.step,
            "avg_train_loss": self.avg_train_loss,
            "optimizer_step": self.optimizer_step,
        }
        with open(path, "w") as w:
            w.write(yaml.dump(save_dict))

    @sb.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch, device):
        del end_of_epoch
        del device
        with open(path) as f:
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
