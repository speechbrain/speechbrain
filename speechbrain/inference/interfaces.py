"""Defines interfaces for simple inference with pretrained models

Authors:
 * Aku Rouhe 2021
 * Peter Plantinga 2021
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Titouan Parcollet 2021
 * Abdel Heba 2021
 * Andreas Nautsch 2022, 2023
 * Pooneh Mousavi 2023
 * Sylvain de Langen 2023
 * Adel Moumen 2023
 * Pradnya Kandarkar 2023
"""

import sys
import warnings
from types import SimpleNamespace

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from torch.nn import DataParallel as DP
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP

from speechbrain.dataio.batch import PaddedBatch, PaddedData
from speechbrain.dataio.preprocess import AudioNormalizer
from speechbrain.utils.data_pipeline import DataPipeline
from speechbrain.utils.data_utils import split_path
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.fetching import LocalStrategy, fetch
from speechbrain.utils.logger import get_logger
from speechbrain.utils.superpowers import import_from_path

logger = get_logger(__name__)


def foreign_class(
    source,
    hparams_file="hyperparams.yaml",
    pymodule_file="custom.py",
    classname="CustomInterface",
    overrides={},
    overrides_must_match=True,
    savedir=None,
    use_auth_token=False,
    download_only=False,
    huggingface_cache_dir=None,
    local_strategy: LocalStrategy = LocalStrategy.SYMLINK,
    **kwargs,
):
    """Fetch and load an interface from an outside source

    The source can be a location on the filesystem or online/huggingface

    The pymodule file should contain a class with the given classname. An
    instance of that class is returned. The idea is to have a custom Pretrained
    subclass in the file. The pymodule file is also added to the python path
    before the Hyperparams YAML file is loaded, so it can contain any custom
    implementations that are needed.

    The hyperparams file should contain a "modules" key, which is a
    dictionary of torch modules used for computation.

    The hyperparams file should contain a "pretrainer" key, which is a
    speechbrain.utils.parameter_transfer.Pretrainer

    Arguments
    ---------
    source : str or Path or FetchSource
        The location to use for finding the model. See
        ``speechbrain.utils.fetching.fetch`` for details.
    hparams_file : str
        The name of the hyperparameters file to use for constructing
        the modules necessary for inference. Must contain two keys:
        "modules" and "pretrainer", as described.
    pymodule_file : str
        The name of the Python file that should be fetched.
    classname : str
        The name of the Class, of which an instance is created and returned
    overrides : dict
        Any changes to make to the hparams file when it is loaded.
    overrides_must_match : bool
        Whether an error will be thrown when an override does not match
        a corresponding key in the yaml_stream.
    savedir : str or Path
        Where to put the pretraining material. If not given, just use cache.
    use_auth_token : bool (default: False)
        If true Huggingface's auth_token will be used to load private models from the HuggingFace Hub,
        default is False because the majority of models are public.
    download_only : bool (default: False)
        If true, class and instance creation is skipped.
    huggingface_cache_dir : str
        Path to HuggingFace cache; if None -> "~/.cache/huggingface" (default: None)
    local_strategy : speechbrain.utils.fetching.LocalStrategy
        The fetching strategy to use, which controls the behavior of remote file
        fetching with regards to symlinking and copying.
        See :func:`speechbrain.utils.fetching.fetch` for further details.
    **kwargs : dict
        Arguments to forward to class constructor.

    Returns
    -------
    object
        An instance of a class with the given classname from the given pymodule file.
    """
    hparams_local_path = fetch(
        filename=hparams_file,
        source=source,
        savedir=savedir,
        overwrite=False,
        save_filename=None,
        use_auth_token=use_auth_token,
        revision=None,
        huggingface_cache_dir=huggingface_cache_dir,
        local_strategy=local_strategy,
    )
    pymodule_local_path = fetch(
        filename=pymodule_file,
        source=source,
        savedir=savedir,
        overwrite=False,
        save_filename=None,
        use_auth_token=use_auth_token,
        revision=None,
        huggingface_cache_dir=huggingface_cache_dir,
        local_strategy=local_strategy,
    )
    sys.path.append(str(pymodule_local_path.parent))

    # Load the modules:
    with open(hparams_local_path, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides, overrides_must_match)

    hparams["savedir"] = savedir
    # Pretraining:
    pretrainer = hparams["pretrainer"]
    pretrainer.set_collect_in(savedir)
    # For distributed setups, have this here:
    run_on_main(
        pretrainer.collect_files,
        kwargs={
            "default_source": source,
            "use_auth_token": use_auth_token,
            "local_strategy": local_strategy,
        },
    )
    # Load on the CPU. Later the params can be moved elsewhere by specifying
    if not download_only:
        # run_opts={"device": ...}
        pretrainer.load_collected()

        # Import class and create instance
        module = import_from_path(pymodule_local_path)
        cls = getattr(module, classname)
        return cls(modules=hparams["modules"], hparams=hparams, **kwargs)


class Pretrained(torch.nn.Module):
    """Takes a trained model and makes predictions on new data.

    This is a base class which handles some common boilerplate.
    It intentionally has an interface similar to ``Brain`` - these base
    classes handle similar things.

    Subclasses of Pretrained should implement the actual logic of how
    the pretrained system runs, and add methods with descriptive names
    (e.g. transcribe_file() for ASR).

    Pretrained is a torch.nn.Module so that methods like .to() or .eval() can
    work. Subclasses should provide a suitable forward() implementation: by
    convention, it should be a method that takes a batch of audio signals and
    runs the full model (as applicable).

    Arguments
    ---------
    modules : dict of str:torch.nn.Module pairs
        The Torch modules that make up the learned system. These can be treated
        in special ways (put on the right device, frozen, etc.). These are available
        as attributes under ``self.mods``, like self.mods.model(x)
    hparams : dict
        Each key:value pair should consist of a string key and a hyperparameter
        that is used within the overridden methods. These will
        be accessible via an ``hparams`` attribute, using "dot" notation:
        e.g., self.hparams.model(x).
    run_opts : dict
        Options parsed from command line. See ``speechbrain.parse_arguments()``.
        List that are supported here:
         * device
         * data_parallel_count
         * data_parallel_backend
         * distributed_launch
         * distributed_backend
         * jit
         * jit_module_keys
         * compule
         * compile_module_keys
         * compile_mode
         * compile_using_fullgraph
         * compile_using_dynamic_shape_tracing
    freeze_params : bool
        To freeze (requires_grad=False) parameters or not. Normally in inference
        you want to freeze the params. Also calls .eval() on all modules.
    """

    HPARAMS_NEEDED = []
    MODULES_NEEDED = []

    def __init__(
        self, modules=None, hparams=None, run_opts=None, freeze_params=True
    ):
        super().__init__()
        # Arguments passed via the run opts dictionary. Set a limited
        # number of these, since some don't apply to inference.
        run_opt_defaults = {
            "device": "cpu",
            "data_parallel_count": -1,
            "data_parallel_backend": False,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "jit": False,
            "jit_module_keys": None,
            "compile": False,
            "compile_module_keys": None,
            "compile_mode": "reduce-overhead",
            "compile_using_fullgraph": False,
            "compile_using_dynamic_shape_tracing": False,
        }
        for arg, default in run_opt_defaults.items():
            if run_opts is not None and arg in run_opts:
                setattr(self, arg, run_opts[arg])
            else:
                # If any arg from run_opt_defaults exist in hparams and
                # not in command line args "run_opts"
                if hparams is not None and arg in hparams:
                    setattr(self, arg, hparams[arg])
                else:
                    setattr(self, arg, default)

        # Put modules on the right device, accessible with dot notation
        self.mods = torch.nn.ModuleDict(modules)
        for module in self.mods.values():
            if module is not None:
                module.to(self.device)

        # Check MODULES_NEEDED and HPARAMS_NEEDED and
        # make hyperparams available with dot notation
        if self.HPARAMS_NEEDED and hparams is None:
            raise ValueError("Need to provide hparams dict.")
        if hparams is not None:
            # Also first check that all required params are found:
            for hp in self.HPARAMS_NEEDED:
                if hp not in hparams:
                    raise ValueError(f"Need hparams['{hp}']")
            self.hparams = SimpleNamespace(**hparams)

        # Prepare modules for computation, e.g. jit
        self._prepare_modules(freeze_params)

        # Audio normalization
        self.audio_normalizer = hparams.get(
            "audio_normalizer", AudioNormalizer()
        )

    def _prepare_modules(self, freeze_params):
        """Prepare modules for computation, e.g. jit.

        Arguments
        ---------
        freeze_params : bool
            Whether to freeze the parameters and call ``eval()``.
        """

        # Make jit-able
        self._compile()
        self._wrap_distributed()

        # If we don't want to backprop, freeze the pretrained parameters
        if freeze_params:
            self.mods.eval()
            for p in self.mods.parameters():
                p.requires_grad = False

    def load_audio(self, path, savedir=None):
        """Load an audio file with this model's input spec

        When using a speech model, it is important to use the same type of data,
        as was used to train the model. This means for example using the same
        sampling rate and number of channels. It is, however, possible to
        convert a file from a higher sampling rate to a lower one (downsampling).
        Similarly, it is simple to downmix a stereo file to mono.
        The path can be a local path, a web url, or a link to a huggingface repo.
        """
        source, fl = split_path(path)
        path = fetch(
            fl,
            source=source,
            savedir=savedir,
            local_strategy=LocalStrategy.SYMLINK,
        )
        signal, sr = torchaudio.load(str(path), channels_first=False)
        signal = signal.to(self.device)
        return self.audio_normalizer(signal, sr)

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
                compile_module_keys = set(self.mods)
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
                jit_module_keys = set(self.mods)
            else:
                jit_module_keys = set(self.jit_module_keys)
                logger.warning(
                    "--jit and --jit_module_keys are both specified. "
                    "Only modules specified in --jit_module_keys will be compiled."
                )

        # find missing keys
        for name in compile_module_keys | jit_module_keys:
            if name not in self.mods:
                raise ValueError(
                    f"module {name} is not defined in your hparams file."
                )

        # try 'torch.compile', remove successful compiles from JIT list
        for name in compile_module_keys:
            try:
                module = torch.compile(
                    self.mods[name],
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

            self.mods[name] = module.to(self.device)
            jit_module_keys.discard(name)

        for name in jit_module_keys:
            module = torch.jit.script(self.mods[name])
            self.mods[name] = module.to(self.device)

    def _compile_jit(self):
        warnings.warn("'_compile_jit' is deprecated; use '_compile' instead")
        self._compile()

    def _wrap_distributed(self):
        """Wrap modules with distributed wrapper when requested."""
        if not self.distributed_launch and not self.data_parallel_backend:
            return
        elif self.distributed_launch:
            for name, module in self.mods.items():
                if any(p.requires_grad for p in module.parameters()):
                    # for ddp, all module must run on same GPU
                    module = SyncBatchNorm.convert_sync_batchnorm(module)
                    module = DDP(module, device_ids=[self.device])
                    self.mods[name] = module
        else:
            # data_parallel_backend
            for name, module in self.mods.items():
                if any(p.requires_grad for p in module.parameters()):
                    # if distributed_count = -1 then use all gpus
                    # otherwise, specify the set of gpu to use
                    if self.data_parallel_count == -1:
                        module = DP(module)
                    else:
                        module = DP(
                            module, [i for i in range(self.data_parallel_count)]
                        )
                    self.mods[name] = module

    @classmethod
    def from_hparams(
        cls,
        source,
        hparams_file="hyperparams.yaml",
        pymodule_file="custom.py",
        overrides={},
        savedir=None,
        use_auth_token=False,
        revision=None,
        download_only=False,
        huggingface_cache_dir=None,
        overrides_must_match=True,
        local_strategy: LocalStrategy = LocalStrategy.SYMLINK,
        **kwargs,
    ):
        """Fetch and load based from outside source based on HyperPyYAML file

        The source can be a location on the filesystem or online/huggingface

        You can use the pymodule_file to include any custom implementations
        that are needed: if that file exists, then its location is added to
        sys.path before Hyperparams YAML is loaded, so it can be referenced
        in the YAML.

        The hyperparams file should contain a "modules" key, which is a
        dictionary of torch modules used for computation.

        The hyperparams file should contain a "pretrainer" key, which is a
        speechbrain.utils.parameter_transfer.Pretrainer

        Arguments
        ---------
        source : str
            The location to use for finding the model. See
            ``speechbrain.utils.fetching.fetch`` for details.
        hparams_file : str
            The name of the hyperparameters file to use for constructing
            the modules necessary for inference. Must contain two keys:
            "modules" and "pretrainer", as described.
        pymodule_file : str
            A Python file can be fetched. This allows any custom
            implementations to be included. The file's location is added to
            sys.path before the hyperparams YAML file is loaded, so it can be
            referenced in YAML.
            This is optional, but has a default: "custom.py". If the default
            file is not found, this is simply ignored, but if you give a
            different filename, then this will raise in case the file is not
            found.
        overrides : dict
            Any changes to make to the hparams file when it is loaded.
        savedir : str or Path
            Where to put the pretraining material. If not given, just use cache.
        use_auth_token : bool (default: False)
            If true Huggingface's auth_token will be used to load private models from the HuggingFace Hub,
            default is False because the majority of models are public.
        revision : str
            The model revision corresponding to the HuggingFace Hub model revision.
            This is particularly useful if you wish to pin your code to a particular
            version of a model hosted at HuggingFace.
        download_only : bool (default: False)
            If true, class and instance creation is skipped.
        huggingface_cache_dir : str
            Path to HuggingFace cache; if None -> "~/.cache/huggingface" (default: None)
        overrides_must_match : bool
            Whether the overrides must match the parameters already in the file.
        local_strategy : LocalStrategy, optional
            Which strategy to use to deal with files locally. (default:
            `LocalStrategy.SYMLINK`)
        **kwargs : dict
            Arguments to forward to class constructor.

        Returns
        -------
        Instance of cls
        """
        hparams_local_path = fetch(
            filename=hparams_file,
            source=source,
            savedir=savedir,
            overwrite=False,
            save_filename=None,
            use_auth_token=use_auth_token,
            revision=revision,
            huggingface_cache_dir=huggingface_cache_dir,
            local_strategy=local_strategy,
        )
        try:
            pymodule_local_path = fetch(
                filename=pymodule_file,
                source=source,
                savedir=savedir,
                overwrite=False,
                save_filename=None,
                use_auth_token=use_auth_token,
                revision=revision,
                huggingface_cache_dir=huggingface_cache_dir,
                local_strategy=local_strategy,
            )
            sys.path.append(str(pymodule_local_path.parent))
        except ValueError:
            if pymodule_file == "custom.py":
                # The optional custom Python module file did not exist
                # and had the default name
                pass
            else:
                # Custom Python module file not found, but some other
                # filename than the default was given.
                raise

        # Load the modules:
        with open(hparams_local_path, encoding="utf-8") as fin:
            hparams = load_hyperpyyaml(
                fin, overrides, overrides_must_match=overrides_must_match
            )

        # add savedir to hparams
        hparams["savedir"] = savedir

        # Pretraining:
        pretrainer = hparams.get("pretrainer", None)
        if pretrainer is not None:
            pretrainer.set_collect_in(savedir)
            # For distributed setups, have this here:
            run_on_main(
                pretrainer.collect_files,
                kwargs={
                    "default_source": source,
                    "use_auth_token": use_auth_token,
                    "local_strategy": local_strategy,
                },
            )
            # Load on the CPU. Later the params can be moved elsewhere by specifying
            if not download_only:
                # run_opts={"device": ...}
                pretrainer.load_collected()

                # Now return the system
                return cls(hparams["modules"], hparams, **kwargs)
        else:
            return cls(hparams["modules"], hparams, **kwargs)


class EncodeDecodePipelineMixin:
    """
    A mixin for pretrained models that makes it possible to specify an encoding pipeline and a decoding pipeline
    """

    def create_pipelines(self):
        """
        Initializes the encode and decode pipeline
        """
        self._run_init_steps(self.hparams.encode_pipeline)
        self._run_init_steps(self.hparams.decode_pipeline)
        self.encode_pipeline = DataPipeline(
            static_data_keys=self.INPUT_STATIC_KEYS,
            dynamic_items=self.hparams.encode_pipeline["steps"],
            output_keys=self.hparams.encode_pipeline["output_keys"],
        )
        self.decode_pipeline = DataPipeline(
            static_data_keys=self.hparams.model_output_keys,
            dynamic_items=self.hparams.decode_pipeline["steps"],
            output_keys=self.OUTPUT_KEYS,
        )

    def _run_init_steps(self, pipeline_definition):
        """Encode/decode pipelines may include initialization
        steps, such as filling text encoders with tokens. Calling
        this method will run them, if defined"""
        steps = pipeline_definition.get("init", [])
        for step in steps:
            step_func = step.get("func")
            if not step_func or not callable(step_func):
                raise ValueError("Invalid pipeline init definition")
            step_func()

    def _run_pipeline(self, pipeline, input, batch):
        if batch:
            output = pipeline(input)
        else:
            output = [pipeline(item) for item in input]
        return output

    def _get_encode_pipeline_input(self, input):
        return input if self.batch_inputs else self._itemize(input)

    def _get_decode_pipeline_input(self, model_output):
        model_output_keys = getattr(self.hparams, "model_output_keys", None)
        pipeline_input = model_output
        if len(model_output_keys) == 1:
            pipeline_input = (pipeline_input,)
        # The input to a pipeline is a dictionary. If model_output_keys
        # is provided, the output of the model is assumed to be a collection
        # (e.g. a list or a tuple).
        if model_output_keys:
            pipeline_input = dict(zip(model_output_keys, pipeline_input))

        # By default, the pipeline will be applied to in batch mode
        # to the entire model input
        if not self.batch_outputs:
            pipeline_input = self._itemize(pipeline_input)
        return pipeline_input

    def _itemize(self, pipeline_input):
        first_item = next(iter(pipeline_input.values()))
        keys, values = pipeline_input.keys(), pipeline_input.values()
        batch_length = len(first_item)
        return [
            dict(zip(keys, [value[idx] for value in values]))
            for idx in range(batch_length)
        ]

    def to_dict(self, data):
        """
        Converts padded batches to dictionaries, leaves
        other data types as is

        Arguments
        ---------
        data: object
            a dictionary or a padded batch

        Returns
        -------
        results: dict
            the dictionary
        """
        if isinstance(data, PaddedBatch):
            data = {
                key: self._get_value(data, key)
                for key in self.hparams.encode_pipeline["output_keys"]
            }
        return data

    def _get_value(self, data, key):
        """
        Retrieves the value associated with the specified key, dereferencing
        .data where applicable

        Arguments
        ---------
        data: PaddedBatch
            a padded batch
        key: str
            the key

        Returns
        -------
        result: object
            the result
        """
        value = getattr(data, key)
        if not self.input_use_padded_data and isinstance(value, PaddedData):
            value = value.data
        return value

    @property
    def batch_inputs(self):
        """
        Determines whether the input pipeline
        operates on batches or individual examples
        (true means batched)

        Returns
        -------
        batch_inputs: bool
        """
        return self.hparams.encode_pipeline.get("batch", True)

    @property
    def input_use_padded_data(self):
        """
        If turned on, raw PaddedData instances will be passed to
        the model. If turned off, only .data will be used

        Returns
        -------
        result: bool
            whether padded data is used as is
        """
        return self.hparams.encode_pipeline.get("use_padded_data", False)

    @property
    def batch_outputs(self):
        """
        Determines whether the output pipeline
        operates on batches or individual examples
        (true means batched)

        Returns
        -------
        batch_outputs: bool
        """
        return self.hparams.decode_pipeline.get("batch", True)

    def _collate(self, data):
        if not self.batch_inputs:
            collate_fn = getattr(self.hparams, "collate_fn", PaddedBatch)
            data = collate_fn(data)
        return data

    def encode_input(self, input):
        """
        Encodes the inputs using the pipeline

        Arguments
        ---------
        input: dict
            the raw inputs

        Returns
        -------
        results: object

        """
        pipeline_input = self._get_encode_pipeline_input(input)
        model_input = self._run_pipeline(
            pipeline=self.encode_pipeline,
            input=pipeline_input,
            batch=self.batch_inputs,
        )
        model_input = self._collate(model_input)
        if hasattr(model_input, "to"):
            model_input = model_input.to(self.device)
        return self.to_dict(model_input)

    def decode_output(self, output):
        """
        Decodes the raw model outputs

        Arguments
        ---------
        output: tuple
            raw model outputs

        Returns
        -------
        result: dict or list
            the output of the pipeline
        """
        pipeline_input = self._get_decode_pipeline_input(output)
        return self._run_pipeline(
            pipeline=self.decode_pipeline,
            input=pipeline_input,
            batch=self.batch_outputs,
        )
