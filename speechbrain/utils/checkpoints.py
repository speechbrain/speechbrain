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

Example
-------
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
>>> tempdir = getfixture('tmpdir')
>>> # In simple cases, the module aims to have a terse syntax,
>>> # consisting of three steps.
>>> # 1. Specifying where to save checkpoints and what is included in a
>>> # checkpoint:
>>> checkpointer = Checkpointer(tempdir, {"network": model})
>>> # 2. Recover from the latest checkpoint, if one is found:
>>> checkpointer.recover_if_possible()
>>> # Run your experiment:
>>> data = [(0.1, 0.9), (0.3, 0.8)]
>>> for example, target in data:
...     loss = (model(example) - target)**2
...     # 3. Save checkpoints, and keep by default just one, the newest:
...     ckpt = checkpointer.save_and_keep_only()

Author
------
Aku Rouhe 2020
"""
import torch
import collections
import collections.abc
import os
import time
import yaml
import pathlib
import inspect
import shutil
import logging

logger = logging.getLogger(__name__)

CKPT_PREFIX = "CKPT"
METAFNAME = f"{CKPT_PREFIX}.yaml"  # Important that this is not .ckpt
PARAMFILE_EXT = ".ckpt"  # ...because these files will be


def torch_recovery(obj, path, end_of_epoch):
    """Loads a torch.nn.Module state_dict from the given path instantly.

    This can be made the default for torch.nn.Modules with:
    >>> DEFAULT_LOAD_HOOKS[torch.nn.Module] = torch_recovery

    Arguments
    ---------
    obj : torch.nn.Module
        Instance for which to load the parameters
    path : str, pathlib.Path
        Path where to load from
    end_of_epoch : bool
        Whether the recovery comes from an end of epoch checkpoint.

    Returns
    -------
    None
        Given object is modified in place

    Author
    ------
    Aku Rouhe 2020
    """
    del end_of_epoch  # Unused
    obj.load_state_dict(torch.load(path), strict=True)


def torch_save(obj, path):
    """Saves the obj's parameters to path.

    Default save hook for torch.nn.Modules
    For saving torch.nn.Module state_dicts.

    Arguments
    ---------
    obj : torch.nn.Module
        Instance to save
    path : str, pathlib.Path
        Path where to save to

    Returns
    -------
    None
        State dict is written to disk.
    """
    state_dict = obj.state_dict()
    if not state_dict:
        logger.warning(f"Saving an empty state_dict for {obj} in {path}.")
    torch.save(state_dict, path)


# These dicts are indexed by class and hold the default checkpoints methods
DEFAULT_LOAD_HOOKS = {
    torch.nn.Module: torch_recovery,
}
DEFAULT_SAVE_HOOKS = {
    torch.nn.Module: torch_save,
}


def mark_as_saver(method):
    """Method decorator which marks given method as the checkpoint saving hook.

    See register_checkpoint_hooks for example

    Arguments
    ---------
    method : callable
        Method of the class to decorate. Must be callable with
        signature (instance, path) using positional arguments. This is
        satisfied by for example: def saver(self, path):

    Note
    ----
    This will not add the hook (not possible via a method decorator),
    you must also decorate the class with @register_checkpoint_hooks
    Only one method can be added as the hook.
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

    Arguments
    ---------
    method : callable
        Method of the class to decorate. Must be callable with
        signature (instance, path, end_of_epoch) using positional
        arguments. This is satisfied by for example:
        `def loader(self, path, end_of_epoch):`

    Note
    ----
    This will not add the hook (not possible via a method decorator),
    you must also decorate the class with @register_checkpoint_hooks
    Only one method can be added as the hook.
    """
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"), True)
    except TypeError:
        MSG = "Checkpoint loader must have signature (self, path, end_of_epoch)"
        raise TypeError(MSG)
    method._speechbrain_loader = True
    return method


def register_checkpoint_hooks(cls):
    """Class decorator which registers the recover load and save hooks

    The hooks must have been marked with mark_as_loader and mark_as_saver.

    Arguments
    ---------
    cls : class
        Class to decorate

    Example
    -------
    >>> @register_checkpoint_hooks
    ... class CustomRecoverable:
    ...     def __init__(self, param):
    ...         self.param = int(param)
    ...
    ...     @mark_as_saver
    ...     def save(self, path):
    ...         with open(path, "w") as fo:
    ...             fo.write(str(self.param))
    ...
    ...     @mark_as_loader
    ...     def load(self, path, end_of_epoch):
    ...         del end_of_epoch  # Unused here
    ...         with open(path) as fi:
    ...             self.param = int(fi.read())
    """
    global DEFAULT_LOAD_HOOKS
    global DEFAULT_SAVE_HOOKS
    for name, method in cls.__dict__.items():
        if hasattr(method, "_speechbrain_saver"):
            DEFAULT_SAVE_HOOKS[cls] = method
            logger.debug(f"Registered checkpoint save hook for {name}")
        if hasattr(method, "_speechbrain_loader"):
            DEFAULT_LOAD_HOOKS[cls] = method
            logger.debug(f"Registered checkpoint load hook for {name}")
    return cls


def get_default_hook(obj, default_hooks):
    """Finds the default save/load hook to use with the given object.

    Follows the Method Resolution Order, i.e. if no hook is registered for
    the class of the object itself, also searches classes which the object
    inherits from.

    Arguments
    ---------
    obj : instance
        Instance of a class
    default_hooks : dict
        Mapping from classes to (checkpointing hook) functions

    Returns
    -------
    The correct method or None if no method is registered.

    Example
    -------
    >>> a = torch.nn.Module()
    >>> get_default_hook(a, DEFAULT_SAVE_HOOKS) == torch_save
    True
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
Checkpointers put pathlib.Path in path and a dict in meta.
You can essentially add any info you want to meta when saving a checkpoint.
The only default key in meta is "unixtime".
Checkpoint.parameters is a dict from recoverable name to parameter filepath.
"""


def ckpt_recency(ckpt):
    """Recency as Checkpoint importance metric

    This function can also act as an example of how to make checkpoint
    importance keyfuncs. This is a named function, but as you can see
    it could be easily implemented as a lambda in a pinch.
    """
    return ckpt.meta["unixtime"]


class Checkpointer:
    """Saves checkpoints and recovers from them.

    Arguments:

    checkpoints_dir : str, pathlib.Path
        Path to directory where to save checkpoints.
    recoverables : mapping, optional
        Objects to to recover. They need a (unique) name: this is used
        to connect the parameters in a checkpoint to the correct recoverable.
        The name is also used in the filename of the
        savefile for the objects parameters. These can also be added with
        add_recoverable or add_recoverables or just modifying
        checkpointer.recoverables directly.
    custom_load_hooks : mapping, optional
        A mapping from name [same as in recoverables] to function or method.
        Sets a custom loading hook for a particular object. The
        function/method must be callable with signature (instance, path)
        using positional arguments. This is satisfied by for example:
        `def loader(self, path)`
    custom_save_hooks : mapping, optional
        Mapping from name [same as in recoverables] to function or method.
        Sets a custom saving hook for a particular object. The
        function/method must be callable with
        signature (instance, path) using positional arguments. This is
        satisfied by for example: def saver(self, path):
    allow_partial_load : bool, optional
        default: False
        If True, allows loading a checkpoint where a savefile is not found
        for every registered recoverable. In that case, only the found
        savefiles are loaded. When False, loading such a save will raise
        RuntimeError.

    Example
    -------
    >>> import torch
    >>> #SETUP:
    >>> tempdir = getfixture('tmpdir')
    >>> class Recoverable(torch.nn.Module):
    ...     def __init__(self, param):
    ...         super().__init__()
    ...         self.param = torch.nn.Parameter(torch.tensor([param]))
    ...     def forward(self, x):
    ...         return x * self.param
    >>> recoverable = Recoverable(1.)
    >>> recoverables = {'recoverable': recoverable}
    >>> # SETUP DONE.
    >>> checkpointer = Checkpointer(tempdir, recoverables)
    >>> first_ckpt = checkpointer.save_checkpoint()
    >>> recoverable.param.data = torch.tensor([2.])
    >>> loaded_ckpt = checkpointer.recover_if_possible()
    >>> # Parameter has been loaded:
    >>> assert recoverable.param.data == torch.tensor([1.])
    >>> # With this call, by default, oldest checkpoints are deleted:
    >>> checkpointer.save_and_keep_only()
    >>> assert first_ckpt not in checkpointer.list_checkpoints()
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
            self.custom_load_hooks.update(custom_load_hooks)
        self.custom_save_hooks = {}
        if custom_save_hooks is not None:
            self.custom_save_hooks.update(custom_save_hooks)
        self.allow_partial_load = allow_partial_load

    def add_recoverable(
        self, name, obj, custom_load_hook=None, custom_save_hook=None
    ):
        """Register a recoverable with possible custom hooks.

        Arguments
        ---------
        name : str
            Unique name for recoverable. Used to map savefiles to objects.
        obj : instance
            The object to recover
        custom_load_hook : callable
            Called to load the object's savefile. The function/method must be
            callable with signature (instance, path) using positional
            arguments. This is satisfied by for example: def load(self, path):
        custom_save_hook : callable
            Called to save the object's parameters. The function/method must
            be callable with signature (instance, path) using positional
            arguments. This is satisfied by for example: def saver(self, path):
        """
        self.recoverables[name] = obj
        if custom_load_hook is not None:
            self.custom_load_hooks[name] = custom_load_hook
        if custom_save_hook is not None:
            self.custom_save_hooks[name] = custom_save_hook

    def add_recoverables(self, recoverables):
        """Update the recoverables dict from the given mapping.

        Arguments
        ---------
        recoverables : mapping
            Objects to to recover.
            They need a (unique) name: this is used to
            connect the parameters in a checkpoint to the correct
            recoverable. The name is also used in the filename of the
            savefile for the objects parameters.
        """
        if isinstance(recoverables, collections.abc.Mapping):
            self.recoverables.update(recoverables)
        else:
            rec = repr(recoverables)  # noqa: F841, rec is used in MSG
            MSG = "Checkpointer needs a mapping (e.g. dict), \
                    got {rec} instead."
            raise AttributeError(MSG)

    def save_checkpoint(self, meta={}, end_of_epoch=True, name=None):
        """Saves a checkpoint.

        The whole checkpoint becomes a directory.
        Saves each registered object's parameters in a separate file.
        Also a meta file is added. The meta file by default has just the
        unixtime (seconds since unix epoch), but you can add anything
        relevant yourself. The meta information is later used to pick the
        checkpoint to load.

        The value of end_of_epoch is saved in the meta. This can affect how
        epoch counters and dataset iterators load their state.

        Arguments
        ---------
        meta : mapping, optional
            A mapping which is added to the meta file in the checkpoint. The
            key "unixtime" is included by default.
        end_of_epoch : bool, optional
            Whether the checkpoint is at the end of an epoch. True by default.
            May affect loading.
        name : str, optional
            Specify a custom name for your checkpoint.
            The name will still have a prefix added. If no name is given,
            a name is created from a timestamp and a random unique id.

        Returns
        -------
        Checkpoint
            namedtuple [see above], the saved checkpoint
        """
        if name is None:
            ckpt_dir = self._new_checkpoint_dirpath()
        else:
            ckpt_dir = self._custom_checkpoint_dirpath(name)
        os.makedirs(ckpt_dir)  # May raise FileExistsError, let it.
        saved_meta = self._save_checkpoint_metafile(
            ckpt_dir / METAFNAME, meta, end_of_epoch
        )
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
            MSG = f"Don't know how to save {type(obj)}. Register default hook \
                    or add custom hook for this object."
            raise RuntimeError(MSG)
        logger.info(f"Saved a checkpoint in {ckpt_dir}")
        return Checkpoint(ckpt_dir, saved_meta, saved_paramfiles)

    def save_and_keep_only(
        self,
        meta={},
        end_of_epoch=True,
        name=None,
        num_to_keep=1,
        importance_keys=[ckpt_recency],
        ckpt_predicate=None,
    ):
        """Saves a checkpoint, then deletes the least important checkpoints

        Essentially this combines save_checkpoint() and delete_checkpoints()
        in one call, only provided for very short syntax in simple cases.

        Arguments
        ---------
        meta : mapping, optional
            A mapping which is added to the meta file in the checkpoint. The
            key "unixtime" is included by default.
        end_of_epoch : bool, optional
            Whether the checkpoint is at the end of an epoch. True by default.
            May affect loading.
        name : str, optional
            Specify a custom name for your checkpoint.
            The name will still have a prefix added. If no name is given,
            a name is created from a timestamp and a random unique id.
        num_to_keep : int, optional
            Number of checkpoints to keep.
            Defaults to 10. You choose to keep 0. This deletes all
            checkpoints remaining after filtering. Must be >=0
        importance_keys : list, optional
            A list of key functions used in sorting (see the sorted built-in).
            Each callable defines a sort order and num_to_keep checkpoints are
            kept for  callable. To be clear, those with the highest key are
            kept.
            The functions are called with Checkpoint namedtuples
            (see above). See also the default (ckpt_recency,
            above). The default deletes all but the latest checkpoint.
        ckpt_predicate : callable, optional
            Use this to exclude some checkpoints from deletion. Before any
            sorting, the list of checkpoints is filtered with this predicate.
            Only the checkpoints for which ckpt_predicate is True can be
            deleted. The function is called with Checkpoint namedtuples
            (see above).

        Returns
        -------
        None
            Unlike save_checkpoint, this does not return anything, since
            we cannot guarantee that the saved checkpoint actually survives
            deletion.
        """
        self.save_checkpoint(meta=meta, end_of_epoch=end_of_epoch, name=name)
        self.delete_checkpoints(
            num_to_keep=num_to_keep,
            importance_keys=importance_keys,
            ckpt_predicate=ckpt_predicate,
        )

    def find_checkpoint(
        self, importance_key=ckpt_recency, ckpt_predicate=None,
    ):
        """Picks a particular checkpoint from all available checkpoints.

        Arguments
        ---------
        importance_key : callable, optional
            The key function used in sorting (see the max built-in).
            The checkpoint with the highest key value is picked. By default,
            the key value is unixtime. The higher the unixtime,
            the newer -> the latest checkpoint is picked.
            The function is called with Checkpoint namedtuples (see above).
            See also the default (ckpt_recency, above).
        ckpt_predicate : callable, optional
            Before sorting, the list of
            checkpoints is filtered with this predicate.
            See the filter builtin.
            The function is called with Checkpoint namedtuples (see above).
            By default, all checkpoints are considered.

        Returns
        -------
        Checkpoint
            if found
        None
            if no Checkpoints exist/remain after filtering
        """
        ckpts = self.list_checkpoints()
        ckpts = list(filter(ckpt_predicate, ckpts))
        if ckpts:
            chosen_ckpt = max(ckpts, key=importance_key)
            return chosen_ckpt
        else:
            return None  # Be explicit :)

    def recover_if_possible(
        self, importance_key=ckpt_recency, ckpt_predicate=None,
    ):
        """Picks a checkpoint and recovers from that, if one is found.

        If a checkpoint is not found, no recovery is run.

        Arguments
        ---------
        importance_key : callable, optional
            The key function used in sorting (see the max built-in).
            The checkpoint with the highest key value is picked. By default,
            the key value is unixtime. The higher the unixtime,
            the newer -> the latest checkpoint is picked.
            The function is called with Checkpoint namedtuples (see above).
            See also the default (ckpt_recency, above).
        ckpt_predicate : callable, optional
            Before sorting, the list of
            checkpoints is filtered with this predicate.
            See the filter builtin.
            The function is called with Checkpoint namedtuples (see above).
            By default, all checkpoints are considered.


        Returns
        -------
        Checkpoint
            if found
        None
            if no Checkpoints exist/remain after filtering
        """
        chosen_ckpt = self.find_checkpoint(importance_key, ckpt_predicate)
        if chosen_ckpt is not None:
            self.load_checkpoint(chosen_ckpt)
        else:
            logger.info("Would load a checkpoint here, but none found yet.")
        return chosen_ckpt

    def load_checkpoint(self, checkpoint):
        """Loads the specified checkpoint.

        Arguments
        ---------
        checkpoint : Checkpoint
            Checkpoint to load
        """
        self._call_load_hooks(checkpoint)

    def list_checkpoints(self):
        """List all checkpoints in the checkpoints directory.

        Returns
        -------
        list
            list of Checkpoint namedtuple (see above)
        """
        return self._load_checkpoint_extra_data(self._list_checkpoint_dirs())

    # NOTE: * in arglist -> keyword only arguments
    def delete_checkpoints(
        self,
        *,
        num_to_keep=1,
        importance_keys=[ckpt_recency],
        ckpt_predicate=None,
    ):
        """Deletes least important checkpoints.

        Since there can be many ways to define importance (e.g. lowest WER,
        lowest loss), the user should provide a list of sort key functions,
        each defining a particular importance order. In essence, each
        importance key function extracts one importance metric (higher is more
        important). For each of these orders, num_to_keep checkpoints are kept.
        However if there is overlap between each orders' preserved checkpoints,
        the additional checkpoints are not preserved, so the total number of
        preserved checkpoints can be less than
            num_to_keep * len(importance_keys)

        Arguments
        ---------
        num_to_keep : int, optional
            Number of checkpoints to keep.
            Defaults to 10. You choose to keep 0. This deletes all
            checkpoints remaining after filtering. Must be >=0
        importance_keys : list, optional
            A list of key functions used in sorting (see the sorted built-in).
            Each callable defines a sort order and num_to_keep checkpoints are
            kept for  callable. To be clear, those with the highest key are
            kept.
            The functions are called with Checkpoint namedtuples
            (see above). See also the default (ckpt_recency,
            above). The default deletes all but the latest checkpoint.
        ckpt_predicate : callable, optional
            Use this to exclude some checkpoints from deletion. Before any
            sorting, the list of checkpoints is filtered with this predicate.
            Only the checkpoints for which ckpt_predicate is True can be
            deleted. The function is called with Checkpoint namedtuples
            (see above).

        Note
        ----
        Must be called with keyword arguments, as a signoff that you
        know what you are doing. Deletion is permanent.
        """
        if num_to_keep < 0:
            raise ValueError("Number of checkpoints to keep must be positive.")
        ckpts = self.list_checkpoints()
        ckpts = list(filter(ckpt_predicate, ckpts))
        protected_checkpoints = []
        for importance_key in importance_keys:
            to_keep = sorted(ckpts, key=importance_key, reverse=True)[
                :num_to_keep
            ]
            protected_checkpoints.extend(to_keep)
        for ckpt in ckpts:
            if ckpt not in protected_checkpoints:
                Checkpointer._delete_checkpoint(ckpt)

    @staticmethod
    def _delete_checkpoint(checkpoint):
        if not Checkpointer._is_checkpoint_dir(checkpoint.path):
            raise RuntimeError("Checkpoint does not appear valid for deletion.")
        shutil.rmtree(checkpoint.path)
        logger.info(f"Deleted checkpoint in {checkpoint.path}")

    def _call_load_hooks(self, checkpoint):
        # This internal function finds the correct hook to call for every
        # recoverable, and calls it.
        logger.info(f"Loading a checkpoint from {checkpoint.path}")
        end_of_epoch = checkpoint.meta["end-of-epoch"]
        for name, obj in self.recoverables.items():
            # NOTE: We want the checkpoint namedtuple to have the paramfile
            # paths for each recoverable.
            # In some rare case, the user can e.g. add a path there manually.
            try:
                loadpath = checkpoint.paramfiles[name]
            except KeyError:
                if self.allow_partial_load:
                    continue
                else:
                    MSG = f"Loading checkpoint from {checkpoint.path}, \
                            but missing a load path for {name}"
                    raise RuntimeError(MSG)
            # First see if object has custom load hook:
            if name in self.custom_load_hooks:
                self.custom_load_hooks[name](obj, loadpath, end_of_epoch)
                continue
            # Otherwise find the default saver for that type:
            default_hook = get_default_hook(obj, DEFAULT_LOAD_HOOKS)
            if default_hook is not None:
                default_hook(obj, loadpath, end_of_epoch)
                continue
            # If we got here, no custom hook or registered default hook exists
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
        suffix_num = 0
        while (
            self.checkpoints_dir / f"{CKPT_PREFIX}+{stamp}+{suffix_num:02d}"
        ).exists():
            suffix_num += 1
        return self.checkpoints_dir / f"{CKPT_PREFIX}+{stamp}+{suffix_num:02d}"

    def _custom_checkpoint_dirpath(self, name):
        # This internal method creates a checkpoint name based on a given
        # custom name and returns a path to that directory (but does not
        # create the directory!)
        return self.checkpoints_dir / f"{CKPT_PREFIX}+{name}"

    def _save_checkpoint_metafile(
        self, fpath, meta_to_include={}, end_of_epoch=True
    ):
        # This internal method saves the meta information in the given path
        meta = {"unixtime": time.time(), "end-of-epoch": end_of_epoch}
        meta.update(meta_to_include)
        with open(fpath, "w") as fo:
            fo.write("# yamllint disable\n")
            fo.write(yaml.dump(meta))
        return meta
