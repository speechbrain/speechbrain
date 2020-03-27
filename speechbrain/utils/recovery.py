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
    # Use this hook with functools.partial to save objpath properly
    # Otherwise, objpath is searched for dynamically (and has probably changed)
    def _lazy_recovery_hook(path, self, *input):
        self.load_state_dict(torch.load(path))
        self._speechbrain_lazy_recovery_hook.remove()
    hook = functools.partial(_lazy_recovery_hook, path)
    obj._speechbrain_lazy_recovery_hook = obj.register_forward_pre_hook(hook)

def torch_instant_load(obj, path):
    obj.load_state_dict(torch.load(path))

def torch_save(obj, path):
    torch.save(obj.state_dict(), path)


DEFAULT_LOAD_HOOKS = {
        torch.nn.Module: torch_lazy_load,
        }
DEFAULT_SAVE_HOOKS = {
        torch.nn.Module: torch_save,
        }

def mark_as_saver(method):
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"))
    except TypeError:
        MSG = "Recovery saver must match signature (instance, path)"
        raise TypeError(MSG)
    method._speechbrain_saver = True
    return method

def mark_as_loader(method):
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"))
    except TypeError:
        MSG = "Recovery loader must match signature (instance, path)"
        raise TypeError(MSG)
    method._speechbrain_loader = True
    return method

def register_recovery_hooks(cls):
    global DEFAULT_LOAD_HOOKS
    global DEFAULT_SAVE_HOOKS
    for name, method in cls.__dict__.items():
        if hasattr(method, "_speechbrain_saver"):
            DEFAULT_SAVE_HOOKS[cls] = method
        if hasattr(method, "_speechbrain_loader"):
            DEFAULT_LOAD_HOOKS[cls] = method
    return cls
            
def get_default_hook(obj, default_hooks): 
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
    return - ckpt.meta["unixtime"]
def _latest_ckpt_filter(ckpt):
    return "unixtime" in ckpt.meta



class Recoverer:

    def __init__(self, 
            checkpoints_dir, 
            recoverables = None, 
            custom_load_hooks = None,
            custom_save_hooks = None,
            allow_partial_load = False):
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

    def add_recoverable(self, 
            name, 
            obj, 
            custom_load_hook = None, 
            custom_save_hook = None):
        self.recoverables[name] = obj
        if custom_load_hook is not None:
            self.custom_load_hooks[name] = custom_load_hook
        if custom_save_hook is not None:
            self.custom_save_hooks[name] = custom_save_hook
    
    def add_recoverables(self, recoverables):
        if isinstance(recoverables, collections.abc.Mapping):
            self.recoverables.update(recoverables)
        else:
            rec = repr(recoverables)
            MSG = "Recoverer needs a mapping (e.g. dict), \
                    got {rec} instead."
            raise AttributeError(MSG)

    def save_checkpoint(self, name=None, meta = {}):
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

    def find_checkpoint(self, 
            ckpt_sort_key = _latest_ckpt_keyfunc,
            ckpt_filter = _latest_ckpt_filter):
        ckpts = self.list_checkpoints()
        ckpts = list(filter(ckpt_filter, ckpts))
        if ckpts:
            chosen_ckpt = sorted(ckpts, key=ckpt_sort_key)[0]
            return chosen_ckpt
        else:
            return None  # Be explicit :)
    
    def load_checkpoint(self, checkpoint):
        self._call_load_hooks(checkpoint)

    def recover_if_possible(self,
            ckpt_sort_key = _latest_ckpt_keyfunc,
            ckpt_filter = _latest_ckpt_filter):
        chosen_ckpt = self.find_checkpoint(ckpt_sort_key, ckpt_filter)
        if chosen_ckpt is not None:
            self.load_checkpoint(chosen_ckpt)
        return chosen_ckpt

    def list_checkpoints(self):
        return self._load_checkpoint_meta(self._list_checkpoint_dirs())
    
    def _call_load_hooks(self, checkpoint):
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
        return [x for x in self.checkpoints_dir.iterdir()
                if Recoverer._is_checkpoint_dir(x)]
    
    @staticmethod
    def _load_checkpoint_meta(checkpoint_dirs):
        checkpoints = []
        for ckpt_dir in checkpoint_dirs:
            with open(ckpt_dir / METAFNAME) as fi:
                checkpoints.append(Checkpoint(ckpt_dir, yaml.safe_load(fi)))
        return checkpoints

    @staticmethod
    def _is_checkpoint_dir(path):
        path = pathlib.Path(path)
        if not path.is_dir():
            return False
        if not path.name.startswith(CKPT_PREFIX):
            return False
        return (path / METAFNAME).exists()
        
    def _new_checkpoint_dirpath(self):
        t = time.time()
        stamp = time.strftime('%Y-%m-%d+%H-%M-%Z', time.localtime(t))
        unique_id = uuid.uuid4().hex[:4]
        return self.checkpoints_dir / f"{CKPT_PREFIX}+{stamp}+{unique_id}"
    
    def _custom_checkpoint_dirpath(self, name):
        return self.checkpoints_dir / f"{CKPT_PREFIX}+{name}"

    def _save_checkpoint_metafile(self, fpath, meta_to_include = {}):
        meta = {"unixtime": time.time()}
        meta.update(meta_to_include)
        with open(fpath, "w") as fo:
            fo.write(yaml.dump(meta))
        return meta


