"""This module implements a checkpoint saver and loader.

A checkpoint in an experiment usually needs to save the state of many different
things: the model parameters, optimizer parameters, what epoch is this, etc.
The save format for a checkpoint is a directory, where each of these separate
saveable things gets its own file. Additionally, a special file holds meta
information about the checkpoint (by default just time of creation, but you
can specify anything else you may wish, e.g. validation loss)

The interface for the checkpoint system requires you to specify what things to
save. This approach is flexible and agnostic of how your experiment is actually
run.

The interface requires you to specify names for each thing to save. This name
is used to give the right parameter file to the right object when recovering.

Default saving and loading methods are only added for torch.nn.Modules (and
their subclasses). If those methods do not work for your object, you can
specify your own saving and/or loading methods, either for a particular
instance or a for a class.

Example:
    >>> from speechbrain.utils.checkpoints import Checkpointer
    >>> import tempfile
    >>> class Recoverable(torch.nn.Module):
    ...     def __init__(self, param):
    ...         super().__init__()
    ...         self.param = torch.nn.Parameter(torch.tensor([param]))
    ...     def forward(self, x):
    ...         return x * self.param
    >>> model = Recoverable(1.)
    >>>
    >>> with tempfile.TemporaryDirectory() as tempdir:
    ...     # In simple cases, the module aims to have a terse syntax,
    ...     # consisting of three steps.
    ...     # 1. Specifying where to save checkpoints and what is included in a
    ...     # checkpoint:
    ...     checkpointer = Checkpointer(tempdir, {"network": model})
    ...     # 2. Recover from the latest checkpoint, if one is found:
    ...     checkpointer.recover_if_possible()
    ...     # Run your experiment:
    ...     data = [(0.1, 0.9), (0.3, 0.8)]
    ...     for example, target in data:
    ...         loss = (model(example) - target)**2
    ...         # 3. Save checkpoints:
    ...         ckpt = checkpointer.save_checkpoint()

Author:
    Aku Rouhe 2020
"""
import torch
import types
import collections
import collections.abc
import os
import time
import uuid
import yaml
import pathlib
import inspect
import functools

CKPT_PREFIX = "CKPT"
METAFNAME = f"{CKPT_PREFIX}.yaml"  # Important that this is not .ckpt
PARAMFILE_EXT = ".ckpt"  # ...because these files will be


def torch_recovery(obj, path):
    """Loads a torch.nn.Module state_dict from the given path instantly.

    This can be made the default for torch.nn.Modules with:
    >>> DEFAULT_LOAD_HOOKS[torch.nn.Module] = torch_recovery

    Args:
        obj (torch.nn.Module): Instance for which to
            load the parameters
        path (str, pathlib.Path): Path where to load from
    Returns:
        None - given object is modified in place
    Author:
        Aku Rouhe 2020
    """
    obj.load_state_dict(torch.load(path), strict=True)


def torch_lazy_recovery(obj, path, load_method=torch_recovery):
    """Recovers the obj's checkpoint from path at first forward() call.

    This is the default load hook for torch.nn.Modules.

    Loads a torch.nn.Module state_dict from the given path.
    The load is added as a lazy hook: the file is loaded and the parameters
    transferred the next time the Module is called.

    This is especially useful for the model initialization style widely
    used in SpeechBrain, where a model is initialized based on the input,
    as that initialization also happens at the first call.

    Args:
        obj (torch.nn.Module): Instance for which to
            load the parameters
        path (str, pathlib.Path): Path where to load from
        load_method (callable): Callable with signature (instance, path)
            [e.g. def load(self, path)], which actually performs the
            recovery from the given path.
    Returns:
        None - given object is modified in place
    Note:
        The hook is added as the _speechbrain_lazy_recovery_hook attribute,
        which could theoretically conflict with other attributes
    Author:
        Aku Rouhe 2020
    """
    # Use this hook with functools.partial to save objpath properly
    # Otherwise, objpath is searched for dynamically (and has probably changed)
    def _lazy_recovery_hook(path, self, *input):
        load_method(self, path)
        self._speechbrain_lazy_recovery_hook.remove()

    hook = functools.partial(_lazy_recovery_hook, path)
    obj._speechbrain_lazy_recovery_hook = obj.register_forward_pre_hook(hook)


def torch_save(obj, path):
    """Saves the obj's parameters to path.

    Default save hook for torch.nn.Modules
    For saving torch.nn.Module state_dicts.

    Args:
        obj (torch.nn.Module): Instance to save
        path (str, pathlib.Path): Path where to save to
    Returns:
        None - state dict is written to disk.
    Author:
        Aku Rouhe 2020
    """
    torch.save(obj.state_dict(), path)


# These dicts are indexed by class and hold the default checkpoints methods
DEFAULT_LOAD_HOOKS = {
    torch.nn.Module: torch_lazy_recovery,
}
DEFAULT_SAVE_HOOKS = {
    torch.nn.Module: torch_save,
}


def mark_as_saver(method):
    """Method decorator which marks given method as the checkpoint saving hook.

    NOTE: This will not add the hook (not possible via a method decorator),
    you must also decorate the class with @register_checkpoint_hooks
    Only one method can be added as the hook.

    Args:
        method: Method of the class to decorate. Must be callable with
            signature (instance, path) using positional arguments. This is
            satisfied by for example: def saver(self, path):
    Example:
        See register_checkpoint_hooks
    Author:
        Aku Rouhe 2020
    """
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"))
    except TypeError:
        MSG = "Checkpoint saver must match signature (instance, path)"
        raise TypeError(MSG)
    method._speechbrain_saver = True
    return method


def mark_as_loader(method):
    """Method decorator which marks given method as checkpoint loading hook.

    NOTE: This will not add the hook (not possible via a method decorator),
    you must also decorate the class with @register_checkpoint_hooks
    Only one method can be added as the hook.

    Args:
        method: Method of the class to decorate. Must be callable with
            signature (instance, path) using positional arguments. This is
            satisfied by for example: def loader(self, path):
    Example:
        See register_checkpoint_hooks
    Author:
        Aku Rouhe 2020
    """
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"))
    except TypeError:
        MSG = "Checkpoint loader must match signature (instance, path)"
        raise TypeError(MSG)
    method._speechbrain_loader = True
    return method


def register_checkpoint_hooks(cls):
    """Class decorator which registers the recover load and save hooks

    The hooks must have been marked with mark_as_loader and mark_as_saver.
    Args:
        cls: Class to decorate
    Example:
        >>> @register_checkpoint_hooks
        >>> class CustomRecoverable:
        >>>     def __init__(self, param):
        >>>         self.param = int(param)

        >>>     @mark_as_saver
        >>>     def save(self, path):
        >>>         with open(path, "w") as fo:
        >>>             fo.write(str(self.param))

        >>>     @mark_as_loader
        >>>     def load(self, path):
        >>>         with open(path) as fi:
        >>>             self.param = int(fi.read())

    Author:
        Aku Rouhe 2020
    """
    global DEFAULT_LOAD_HOOKS
    global DEFAULT_SAVE_HOOKS
    for name, method in cls.__dict__.items():
        if hasattr(method, "_speechbrain_saver"):
            DEFAULT_SAVE_HOOKS[cls] = method
        if hasattr(method, "_speechbrain_loader"):
            DEFAULT_LOAD_HOOKS[cls] = method
    return cls


def get_default_hook(obj, default_hooks):
    """Finds the default save/load hook to use with the given object.

    Follows the Method Resolution Order, i.e. if no hook is registered for
    the class of the object itself, also searches classes which the object
    inherits from.

    Args:
        obj: Instance of a class
        default_hooks: Mapping from classes to (checkpointing hook) functions
    Returns:
        The correct method or None if no method is registered.
    Example:
        >>> a = torch.nn.Module()
        >>> get_default_hook(a, DEFAULT_SAVE_HOOKS) == torch_save
        True

    Author:
        Aku Rouhe 2020
    """
    mro = inspect.getmro(type(obj))
    for cls in mro:
        if cls in default_hooks:
            return default_hooks[cls]
    # If we got here, no hook found
    return None


Checkpoint = collections.namedtuple(
    "Checkpoint", ["path", "meta", "paramfiles"]
)
Checkpoint.__doc__ = """NamedTuple describing one saved checkpoint

To select a checkpoint to load from many checkpoint,
Checkpoints are first filtered and sorted based on this namedtuple.
Checkpointers put pathlib.Path in path and a dict in meta
You can essentially add any info you want to meta when saving a checkpoint
The only default key in meta is "unixtime"
Checkpoint.parameters is a dict from recoverable name to parameter filepath
"""


# These internal functions also act as examples of how to make checkpoint
# filters and keyfuncs. These are proper functions, but as you can see
# they could be easily implemented as lambdas in a pinch.
def _latest_ckpt_keyfunc(ckpt):
    # Negative of unixtime -> latest first
    return -ckpt.meta["unixtime"]


def _latest_ckpt_filter(ckpt):
    return "unixtime" in ckpt.meta


class Checkpointer:
    """Saves checkpoints and recovers from them.

    Args:
        checkpoints_dir (str, pathlib.Path): 
            Path to directory where to save checkpoints.
        recoverables (mapping, optional): Objects to to recover. They need a 
            (unique) name: this is used to connect the parameters in a 
            checkpoint to the correct recoverable. 
            The name is also used in the filename of the 
            savefile for the objects parameters. These can also be added with 
            add_recoverable or add_recoverables or just modifying 
            checkpointer.recoverables directly.
        custom_load_hooks (mapping, optional): A mapping from name 
            [same as in recoverables] to function or method.
            Sets a custom loading hook for a particular object. The
            function/method must be callable with signature (instance, path)
            using positional arguments. This is satisfied by for example:
            `def loader(self, path)`
        custom_save_hooks (mapping, optional): Mapping from name 
            [same as in recoverables] to function or method.
            Sets a custom saving hook for a particular object. The
            function/method must be callable with
            signature (instance, path) using positional arguments. This is
            satisfied by for example: def saver(self, path):
        allow_partial_load (boolean, optional): default: False 
            If True, allows loading a checkpoint where a savefile is not found 
            for every registered recoverable. In that case, only the found 
            savefiles are loaded. When False, loading such a save will raise 
            RuntimeError.
    Example:
        >>> from speechbrain.utils.checkpoints import Checkpointer
        >>> import torch
        >>> import tempfile
        >>> #SETUP:
        >>> class Recoverable(torch.nn.Module):
        ...     def __init__(self, param):
        ...         super().__init__()
        ...         self.param = torch.nn.Parameter(torch.tensor([param]))
        ...     def forward(self, x):
        ...         return x * self.param
        >>> recoverable = Recoverable(1.)
        >>> recoverables = {'recoverable': recoverable}
        >>> with tempfile.TemporaryDirectory() as tempdir:
        ...     # SETUP DONE.
        ...     checkpointer = Checkpointer(tempdir, recoverables)
        ...     ckpt = checkpointer.save_checkpoint()
        ...     recoverable.param.data = torch.tensor([2.])
        ...     ckpt = checkpointer.recover_if_possible()
        ...     # Parameter hasn't been loaded yet:
        ...     assert recoverable.param.data == torch.tensor([2.])
        ...     result = recoverable(10.)
        ...     # Parameter has been loaded now:
        ...     assert recoverable.param.data == torch.tensor([1.])
        ...     # And parameter was loaded before computation:
        ...     assert result == 10.

    Author:
        Aku Rouhe 2020
    """

    def __init__(
        self,
        checkpoints_dir,
        recoverables=None,
        custom_load_hooks=None,
        custom_save_hooks=None,
        allow_partial_load=False,
    ):
        self.checkpoints_dir = pathlib.Path(checkpoints_dir)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.recoverables = {}
        if recoverables is not None:
            self.add_recoverables(recoverables)
        self.custom_load_hooks = {}
        if custom_load_hooks is not None:
            self.custom_load_hooks.update(custom_io_hooks)
        self.custom_save_hooks = {}
        if custom_save_hooks is not None:
            self.custom_save_hooks.update(custom_io_hooks)
        self.allow_partial_load = allow_partial_load

    def add_recoverable(
        self, name, obj, custom_load_hook=None, custom_save_hook=None
    ):
        """Register a recoverable with possible custom hooks.

        Args:
            name (str): Unique name for recoverable. Used to map savefiles
                to objects.
            obj (instance): The object to recover
            custom_load_hook (callable): Called to load the object's
                savefile. The function/method must be callable with
                signature (instance, path) using positional arguments. This is
                satisfied by for example: def load(self, path):
            custom_save_hook (callable): Called to save the object's
                parameters. The function/method must be callable with
                signature (instance, path) using positional arguments. This is
                satisfied by for example: def saver(self, path):
        """
        self.recoverables[name] = obj
        if custom_load_hook is not None:
            self.custom_load_hooks[name] = custom_load_hook
        if custom_save_hook is not None:
            self.custom_save_hooks[name] = custom_save_hook

    def add_recoverables(self, recoverables):
        """Update the recoverables dict from the given mapping.

        Args:
            recoverables (mapping): Objects to to recover. 
                They need a (unique) name: this is used to
                connect the parameters in a checkpoint to the correct 
                recoverable. The name is also used in the filename of the 
                savefile for the objects parameters.
        """
        if isinstance(recoverables, collections.abc.Mapping):
            self.recoverables.update(recoverables)
        else:
            rec = repr(recoverables)
            MSG = "Checkpointer needs a mapping (e.g. dict), \
                    got {rec} instead."
            raise AttributeError(MSG)

    def save_checkpoint(self, name=None, meta={}):
        """Saves a checkpoint. 
        
        The whole checkpoint becomes a directory.
        Saves each registered object's parameters in a separate file.
        Also a meta file is added. The meta file by default has just the
        unixtime (seconds since unix epoch), but you can add anything
        relevant yourself. The meta information is later used to pick the
        checkpoint to load.

        Args:
            name (str, optional): Specify a custom name for your checkpoint. 
                The name will still have a prefix added. If no name is given, 
                a name is created from a timestamp and a random unique id.
            meta (mapping, optional): A mapping which is added to the meta 
                file in the checkpoint. The key "unixtime" is included by 
                default.
        Returns:
            Checkpoint namedtuple [see above], the saved checkpoint as
        """
        if name is None:
            ckpt_dir = self._new_checkpoint_dirpath()
        else:
            ckpt_dir = self._custom_checkpoint_dirpath(name)
        os.makedirs(ckpt_dir)  # May raise FileExistsError, let it.
        saved_meta = self._save_checkpoint_metafile(ckpt_dir / METAFNAME, meta)
        saved_paramfiles = {}
        for name, obj in self.recoverables.items():
            objfname = f"{name}" + PARAMFILE_EXT
            savepath = ckpt_dir / objfname
            saved_paramfiles[name] = savepath
            # First see if object has custom load hook:
            if name in self.custom_save_hooks:
                self.custom_save_hooks[name](obj, savepath)
                continue
            # Otherwise find the default saver for that type:
            default_hook = get_default_hook(obj, DEFAULT_SAVE_HOOKS)
            if default_hook is not None:
                default_hook(obj, savepath)
                continue
            # If we got here, no custom hook or registered default hook
            MSG = f"Don't know how to load {type(obj)}. Register default hook \
                    or add custom hook for this object."
            raise RuntimeError(MSG)
        return Checkpoint(ckpt_dir, saved_meta, saved_paramfiles)

    def find_checkpoint(
        self,
        ckpt_sort_key=_latest_ckpt_keyfunc,
        ckpt_filter=_latest_ckpt_filter,
    ):
        """Picks a particular checkpoint from all available checkpoints.
        
        Args:
            ckpt_sort_key (callable, optional): The key function used in
                sorting (see the sorted built-in). The first checkpoint in the
                list after sorting is picked. The function is called with
                Checkpoint namedtuples (see above). See also the default
                (_latest_ckpt_keyfunc, above). The default pick "unixtime" from
                the Checkpoint meta.
            ckpt_filter (callable, optional): Before sorting, the list of
                checkpoints is filtered with this. The function is called with
                Checkpoint namedtuples (see above). See also the default
                (_latest_ckpt_filter, above). The default filter out any
                Checkpoints which donot have the "unixtime" key in meta.
        Returns:
            The picked Checkpoint
            OR
            None if no Checkpoints exist/remain after filtering
        """
        ckpts = self.list_checkpoints()
        ckpts = list(filter(ckpt_filter, ckpts))
        if ckpts:
            chosen_ckpt = sorted(ckpts, key=ckpt_sort_key)[0]
            return chosen_ckpt
        else:
            return None  # Be explicit :)

    def recover_if_possible(
        self,
        ckpt_sort_key=_latest_ckpt_keyfunc,
        ckpt_filter=_latest_ckpt_filter,
    ):
        """Picks a checkpoint and recovers from that, if one is found.
        
        If a checkpoint is not found, no recovery is run.

        Args:
            ckpt_sort_key (callable, optional): See find_checkpoint above.
            ckpt_sort_filter (callable, optional): See find_checkpoint above.
        Returns:
            The picked, recovered Checkpoint
            OR
            None if no checkpoints exist/remain after filtering (and in this
                case no recovery is done).
        """
        chosen_ckpt = self.find_checkpoint(ckpt_sort_key, ckpt_filter)
        if chosen_ckpt is not None:
            self.load_checkpoint(chosen_ckpt)
        return chosen_ckpt

    def load_checkpoint(self, checkpoint):
        """Loads the specified checkpoint.

        Args:
            checkpoint (Checkpoint): Checkpoint to load
        """
        self._call_load_hooks(checkpoint)

    def list_checkpoints(self):
        """List all checkpoints in the checkpoints directory.

        Returns:
            List of Checkpoint namedtuple (see above)
        """
        return self._load_checkpoint_extra_data(self._list_checkpoint_dirs())

    def _call_load_hooks(self, checkpoint):
        # This internal function finds the correct hook to call for every
        # recoverable, and calls it.
        for name, obj in self.recoverables.items():
            objfname = f"{name}.ckpt"
            loadpath = checkpoint.path / objfname
            if not loadpath.exists():
                if self.allow_partial_load:
                    continue
                else:
                    MSG = f"Loading checkpoint from {checkpoint.path}, \
                            expected {loadpath} to exist"
                    raise RuntimeError(MSG)
            # First see if object has custom load hook:
            if name in self.custom_load_hooks:
                self.custom_load_hooks[name](obj, loadpath)
                continue
            # Otherwise find the default saver for that type:
            default_hook = get_default_hook(obj, DEFAULT_LOAD_HOOKS)
            if default_hook is not None:
                default_hook(obj, loadpath)
                continue
            # If we got here, no custom hook or registered default hook
            MSG = f"Don't know how to load {type(obj)}. Register default hook \
                    or add custom hook for this object."
            raise RuntimeError(MSG)

    def _list_checkpoint_dirs(self):
        # This internal method returns a list of individual checkpoint
        # directory paths in the top checkpoint directory
        return [
            x
            for x in self.checkpoints_dir.iterdir()
            if Checkpointer._is_checkpoint_dir(x)
        ]

    @staticmethod
    def _load_checkpoint_extra_data(checkpoint_dirs):
        # This internal method takes a list of individual checkpoint
        # directory paths (as produced by _list_checkpoint_dirs)
        checkpoints = []
        for ckpt_dir in checkpoint_dirs:
            with open(ckpt_dir / METAFNAME) as fi:
                meta = yaml.safe_load(fi)
            paramfiles = {}
            for ckptfile in ckpt_dir.iterdir():
                if ckptfile.suffix == PARAMFILE_EXT:
                    paramfiles[ckptfile.stem] = ckptfile
            checkpoints.append(Checkpoint(ckpt_dir, meta, paramfiles))
        return checkpoints

    @staticmethod
    def _is_checkpoint_dir(path):
        # This internal method verifies whether a given path points to a
        # directory that holds a checkpoint.
        path = pathlib.Path(path)
        if not path.is_dir():
            return False
        if not path.name.startswith(CKPT_PREFIX):
            return False
        return (path / METAFNAME).exists()

    def _new_checkpoint_dirpath(self):
        # This internal method creates a checkpoint name and returns a path
        # to that directory (but does not create the directory!)
        t = time.time()
        stamp = time.strftime("%Y-%m-%d+%H-%M-%S", time.localtime(t))
        unique_id = uuid.uuid4().hex[:4]
        return self.checkpoints_dir / f"{CKPT_PREFIX}+{stamp}+{unique_id}"

    def _custom_checkpoint_dirpath(self, name):
        # This internal method creates a checkpoint name based on a given
        # custom name and returns a path to that directory (but does not
        # create the directory!)
        return self.checkpoints_dir / f"{CKPT_PREFIX}+{name}"

    def _save_checkpoint_metafile(self, fpath, meta_to_include={}):
        # This internal method saves the meta information in the given path
        meta = {"unixtime": time.time()}
        meta.update(meta_to_include)
        with open(fpath, "w") as fo:
            fo.write(yaml.dump(meta))
        return meta
