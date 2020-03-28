"""
This module implements a checkpoint saver and loader.
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
    from speechbrain.utils.recovery import Recoverer
    model = speechbrain.nnet.architectures.linear(n_neurons = 512)j
    # In simple cases, the module aims to have a terse syntax, consisting of
    # three steps.
    # 1. Specifying what is included in a checkpoint, and the checkpoint dir:
    recoverer = Recoverer("exp/checkpoint_dir", {"network": model}
    # 2. Recover from the latest checkpoint, if one is found:
    recoverer.recover_if_possible()
    # Run your experiment:
    data = [([0.2, 0.3, 0.4], 0.9), ([0.3, 0.3, 0.2], 0.8)]
    for example, target in data:
        result = model(example)
        # 3. Save checkpoints:
        recoverer.save_checkpoint()
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
METAFNAME = f"{CKPT_PREFIX}.yaml"


def torch_lazy_load(obj, path):
    """
    The default load hook for torch.nn.Modules.
    Description:
        Loads a torch.nn.Module state_dict from the given path. 
        The load is added as a lazy hook: the file is loaded and the parameters
        transferred the next time the Module is called.
        This is especially useful for the model initialization style widely 
        used in SpeechBrain, where a model is initialized based on the input,
        as that initialization also happens at the first call.
    Input:
        obj (instance of torch.nn.Module or derivative) - Instance for which to
            load the parameters
        path (string or path-like) - Path to where to load from
    Output:
        None - Given object is modified in place
    NOTE: The hook is added as the _speechbrain_lazy_recovery_hook attribute,
        which could theoretically conflict with other attributes
    Author:
        Aku Rouhe 2020
    """
    # Use this hook with functools.partial to save objpath properly
    # Otherwise, objpath is searched for dynamically (and has probably changed)
    def _lazy_recovery_hook(path, self, *input):
        self.load_state_dict(torch.load(path))
        self._speechbrain_lazy_recovery_hook.remove()

    hook = functools.partial(_lazy_recovery_hook, path)
    obj._speechbrain_lazy_recovery_hook = obj.register_forward_pre_hook(hook)


def torch_instant_load(obj, path):
    """
    Description:
        Loads a torch.nn.Module state_dict from the given path instantly.
        This can be made the default for torch.nn.Modules with:
        DEFAULT_LOAD_HOOKS[torch.nn.Module] = torch_instant_load
    Input:
        obj (instance of torch.nn.Module or derivative) - Instance for which to
            load the parameters
        path (string or path-like) - Path to where to load from
    Output:
        None - Given object is modified in place
    Author:
        Aku Rouhe 2020
    """
    obj.load_state_dict(torch.load(path))


def torch_save(obj, path):
    """
    Default save hook for torch.nn.Modules
    Description:
        For saving torch.nn.Module state_dicts.
    Input:
        obj (instance of torch.nn.Module or derivative) - Instance to save 
        path (string or path-like) - Path to where to save to
    Output:
        None - State dict is written to disk.
    Author:
        Aku Rouhe 2020
    """
    torch.save(obj.state_dict(), path)


DEFAULT_LOAD_HOOKS = {
    torch.nn.Module: torch_lazy_load,
}
DEFAULT_SAVE_HOOKS = {
    torch.nn.Module: torch_save,
}


def mark_as_saver(method):
    """
    Method decorator which marks the given method as the recovery saving hook.
    NOTE: This will not add the hook (not possible via a method decorator), 
    you must also decorate the class with @register_recovery_hooks

    Only one method can be added as the hook. 
    Input:
        method - Method of the class to decorate. Must be callable with
            signature (instance, path) using positional arguments. This is
            satisfied by for example: def saver(self, path):
    Output:
        The decorated method
    Example:
        See register_recovery_hooks
    Author:
        Aku Rouhe 2020
    """
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"))
    except TypeError:
        MSG = "Recovery saver must match signature (instance, path)"
        raise TypeError(MSG)
    method._speechbrain_saver = True
    return method


def mark_as_loader(method):
    """
    Method decorator which marks the given method as the recovery loading hook.
    NOTE: This will not add the hook (not possible via a method decorator), 
    you must also decorate the class with @register_recovery_hooks

    Only one method can be added as the hook. 
    Input:
        method - Method of the class to decorate. Must be callable with
            signature (instance, path) using positional arguments. This is
            satisfied by for example: def loader(self, path):
    Output:
        The decorated method
    Example:
        See register_recovery_hooks
    Author:
        Aku Rouhe 2020
    """
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"))
    except TypeError:
        MSG = "Recovery loader must match signature (instance, path)"
        raise TypeError(MSG)
    method._speechbrain_loader = True
    return method


def register_recovery_hooks(cls):
    """
    Class decorator which registers the recover load and save hooks marked with
    mark_as_loader and mark_as_saver.
    Input:
        cls - Class to decorate
    Output:
        The decorated class.
    Example:
        @register_recovery_hooks
        class CustomRecoverable:
            def __init__(self, param):
                self.param = int(param)
            
            @mark_as_saver
            def save(self, path):
                with open(path, "w") as fo:
                    fo.write(str(self.param))

            @mark_as_loader
            def load(self, path):
                with open(path) as fi:
                    self.param = int(fi.read())
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
    """
    Description:
        Finds the default recovery hook to use with the given object.
        Follows the Method Resolution Order, i.e. if no hook is registered for
        the class of the object itself, also searches classes which the object
        inherits from.
    Input:
        obj - Instance of a class
        default_hooks - Mapping from classes to (recovery hook) functions
    Output:
        The 
    """
    mro = inspect.getmro(type(obj))
    for cls in mro:
        if cls in default_hooks:
            return default_hooks[cls]
    # If we got here, no hook found
    return None


Checkpoint = collections.namedtuple("Checkpoint", ["path", "meta"])
# To select a checkpoint to load from many checkpoints,
# checkpoints are first filtered and sorted based on this namedtuple.
# Recoverer put pathlib.Path in path and a dict in meta
# You can essentially add any info you want to meta when saving a checkpoint
# The only default key in meta is "unixtime"


# These internal functions also act as examples of how to make checkpoint
# filters and keyfuncs. These are proper functions, but as you can see
# they could be easily implemented as lambdas in a pinch.
def _latest_ckpt_keyfunc(ckpt):
    # Negative of unixtime -> latest first
    return -ckpt.meta["unixtime"]


def _latest_ckpt_filter(ckpt):
    return "unixtime" in ckpt.meta


class Recoverer:
    """
    Description:
        Saves checkpoints and recovers from them.
    Input:
        checkpoints_dir (str or path-like) - Path to directory where to save
            checkpoints.
        recoverables (mapping from name to object instance, optional) - 
            Objects to to recover. They need a (unique) name: this is used to 
            connect the parameters in a checkpoint to the correct recoverable.
            The name is also used in the filename of the savefile for the 
            objects parameters.
            These can also be added with add_recoverable or add_recoverables
            or just modifying recoverer.recoverables directly.
        custom_load_hooks (mapping from name [same as in recoverables] to 
            function or method, optional) - Sets a custom loading hook for a 
            particular object. The function/method must be callable with
            signature (instance, path) using positional arguments. This is
            satisfied by for example: def loader(self, path):
        custom_save_hooks (mapping from name [same as in recoverables] to
            function or method, optional) - Sets a custom saving hook for a 
            particular object. The function/method must be callable with
            signature (instance, path) using positional arguments. This is
            satisfied by for example: def saver(self, path):
        allow_partial_load (boolean, optional, default: False) - If True, 
            allows loading a checkpoint where a savefile is not found for every
            registered recoverable. In that case, only the found savefiles are
            loaded. When False, loading such a save will raise RuntimeError.
    Example:
        from speechbrain.utils.recovery import Recoverer
        import torch

        class Recoverable(torch.nn.Module):
            def __init__(self, param):
                super().__init__()
                self.param = torch.nn.Parameter(torch.tensor([param]))

            def forward(self, x):
                return x * self.param
       
        recoverable = Recoverable(1.)
        recoverables = {"recoverable": recoverable}
        recoverer = Recoverer("recovery_dir", recoverables)
        recoverer.save_checkpoint()
        recoverer.param.data = torch.tensor([2.])
        recoverer.recover_if_possible()
        # Parameter hasn't been loaded yet:
        assert recoverable.param.data == torch.tensor([2.])
        result = recoverable(10.)
        # Parameter has been loaded now:
        assert recoverable.param.data == torch.tensor([1.])
        # And parameter was loaded before computation:
        assert result == 10.
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
        """
        Description:
            Register a recoverable with possible custom hooks.
        Input:
            name (str) - Unique name for recoverable. Used to map savefiles
                to objects.
            obj (instance) - The object to recover
            custom_load_hook (method or function) - Called to load the object's
                savefile. The function/method must be callable with
                signature (instance, path) using positional arguments. This is
                satisfied by for example: def load(self, path):
            custom_save_hook (method or function) - Called to save the object's
                parameters. The function/method must be callable with
                signature (instance, path) using positional arguments. This is
                satisfied by for example: def saver(self, path):
        Output:
            None
        """
        self.recoverables[name] = obj
        if custom_load_hook is not None:
            self.custom_load_hooks[name] = custom_load_hook
        if custom_save_hook is not None:
            self.custom_save_hooks[name] = custom_save_hook

    def add_recoverables(self, recoverables):
        """
        Description:
            Update the recoverables dict from the given mapping.
        Input:
            recoverables (mapping from name to object instance, optional) - 
            Objects to to recover. They need a (unique) name: this is used to 
            connect the parameters in a checkpoint to the correct recoverable.
            The name is also used in the filename of the savefile for the 
            objects parameters.
        Output:
            None
        """
        if isinstance(recoverables, collections.abc.Mapping):
            self.recoverables.update(recoverables)
        else:
            rec = repr(recoverables)
            MSG = "Recoverer needs a mapping (e.g. dict), \
                    got {rec} instead."
            raise AttributeError(MSG)

    def save_checkpoint(self, name=None, meta={}):
        """
        Description:
            Saves a checkpoint. The whole checkpoint becomes a directory.
            Saves each registered object's parameters in a separate file. 
            Also a meta file is added. The meta file by default has just the
            unixtime (seconds since unix epoch), but you can add anything
            relevant yourself. The meta information is later used to pick the 
            checkpoint to load.
        Input:
            name (str, optional) - Specify a custom name for your checkpoint. The name
                will still have a prefix added. If no name is given, a name is
                created from a timestamp and a random unique id.
            meta (mapping, optional) - A mapping which is added to the meta file in the
                checkpoint. The key "unixtime" is included by default.
        Output:
            ckpt (Checkpoint namedtuple [see above]) - The saved checkpoint as
                a Checkpoint.
        """
        if name is None:
            ckpt_dir = self._new_checkpoint_dirpath()
        else:
            ckpt_dir = self._custom_checkpoint_dirpath(name)
        os.makedirs(ckpt_dir)  # May raise FileExistsError, let it.
        saved_meta = self._save_checkpoint_metafile(ckpt_dir / METAFNAME, meta)
        for name, obj in self.recoverables.items():
            objfname = f"{name}.ckpt"
            savepath = ckpt_dir / objfname
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
        return Checkpoint(ckpt_dir, saved_meta)

    def find_checkpoint(
        self,
        ckpt_sort_key=_latest_ckpt_keyfunc,
        ckpt_filter=_latest_ckpt_filter,
    ):
        """
        Description:
            Picks a particular checkpoint from all the checkpoints saved in the
            checkpoint directory.
        Input:
            ckpt_sort_key (callable, optional) - The key function used in 
                sorting (see the sorted built-in). The first checkpoint in the 
                list after sorting is picked. The function is called with 
                Checkpoint namedtuples (see above). See also the default 
                (_latest_ckpt_keyfunc, above). The default pick "unixtime" from
                the Checkpoint meta.
            ckpt_filter (callable, optional) - Before sorting, the list of 
                checkpoints is filtered with this. The function is called with 
                Checkpoint namedtuples (see above). See also the default 
                (_latest_ckpt_filter, above). The default filter out any
                Checkpoints which donot have the "unixtime" key in meta.
        Output:
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
        """
        Description:
            Picks a checkpoint and recovers from that, if one is found.
            If a checkpoint is not found, no recovery is run.
        Input:
            ckpt_sort_key (callable, optional) - See find_checkpoint above.
            ckpt_sort_filter (callable, optional) - See find_checkpoint above.
        Output:
            The picked, recovered Checkpoint
            OR
            None if no Checkpoints exist/remain after filtering (and in this
                case no recovery is done.
        """
        chosen_ckpt = self.find_checkpoint(ckpt_sort_key, ckpt_filter)
        if chosen_ckpt is not None:
            self.load_checkpoint(chosen_ckpt)
        return chosen_ckpt

    def load_checkpoint(self, checkpoint):
        """
        Description:
            Loads the specified checkpoint.
        Input:
            checkpoint (Checkpoint namedtuple [see above])
        """
        self._call_load_hooks(checkpoint)

    def list_checkpoints(self):
        """
        Description:
            List all checkpoints in the checkpoints directory.
        Output:
            List of Checkpoint namedtuple (see above)
        """
        return self._load_checkpoint_meta(self._list_checkpoint_dirs())

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
            if Recoverer._is_checkpoint_dir(x)
        ]

    @staticmethod
    def _load_checkpoint_meta(checkpoint_dirs):
        # This internal method takes a list of individual checkpoint
        # directory paths (as produced by _list_checkpoint_dirs)
        checkpoints = []
        for ckpt_dir in checkpoint_dirs:
            with open(ckpt_dir / METAFNAME) as fi:
                checkpoints.append(Checkpoint(ckpt_dir, yaml.safe_load(fi)))
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
