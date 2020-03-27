import torch
import types
import collections
import collections.abc
import os
import time
import uuid
import yaml
import pathlib
import functools

CKPT_PREFIX = "CKPT"
METAFNAME = f"{CKPT_PREFIX}.yaml"


Checkpoint = collections.namedtuple("Checkpoint", ["path", "meta"])
# To select a checkpoint to load from many checkpoints,
# checkpoints are first filtered and sorted based on this namedtuple.
# Recoverer put pathlib.Path in path and a dict in meta 
# You can essentially add any info you want to meta when saving a checkpoint
# The only default key in meta is "unixtime" 


# These internal functions also act as examples of how to make checkpoint 
# filters and keyfuncs. These are proper functions, but you can see that 
# they could be easily implemented as lambdas in a pinch. 
def _latest_ckpt_keyfunc(ckpt):
    # Negative of unixtime -> latest first
    return - ckpt.meta["unixtime"]
def _latest_ckpt_filter(ckpt):
    return "unixtime" in ckpt.meta

# Use this hook with functools.partial to save objpath properly
# Otherwise, objpath is searched for dynamically (and has probably changed)
def _lazy_recovery_hook(objpath, self, *input):
    self.load_state_dict(torch.load(objpath))
    self._lazy_recovery_hook.remove()

class Recoverer:

    def __init__(self, 
            checkpoints_dir, 
            recoverables = None, 
            allow_partial_load = False):
        self.checkpoints_dir = pathlib.Path(checkpoints_dir)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.recoverables = {} 
        if recoverables is not None:
            self.add_recoverables(recoverables)
        self.allow_partial_load = allow_partial_load

    def add_recoverable(self, name, obj):
        self.recoverables[name] = obj
    
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
            objfname = f"{name}.pt"
            print(f"Saving: {name}, with state: {obj.state_dict()}")
            torch.save(obj.state_dict(), ckpt_dir / objfname)
        return Checkpoint(ckpt_dir, saved_meta)

    def lazy_recovery_if_possible(self, 
            ckpt_sort_key = _latest_ckpt_keyfunc,
            ckpt_filter = _latest_ckpt_filter):
        ckpts = self.list_checkpoints()
        ckpts = list(filter(ckpt_filter, ckpts))
        if ckpts:
            chosen_ckpt = sorted(ckpts, key=ckpt_sort_key)[0]
            self.add_lazy_load_hooks(chosen_ckpt.path)
            return chosen_ckpt
        else:
            return None  # Be explicit :)

    def add_lazy_load_hooks(self, checkpoint_path):
        for name, obj in self.recoverables.items():
            objfname = f"{name}.pt"
            objpath = checkpoint_path / objfname
            print(f"{name}:{objpath}")
            if not objpath.exists():
                if self.allow_partial_load:
                    continue
                else:
                    MSG = f"Loading checkpoint from {checkpoint_path}, \
                            expected {name}.pt to exist"
                    raise ValueError(MSG)
            # functools.partial to save objpath value
            hook = functools.partial(_lazy_recovery_hook, objpath)
            obj._lazy_recovery_hook = obj.register_forward_pre_hook(hook)

    def list_checkpoints(self):
        return self._load_meta_files(self._list_checkpoint_dirs())

    def _list_checkpoint_dirs(self):
        return [x for x in self.checkpoints_dir.iterdir()
                if Recoverer._is_checkpoint_dir(x)]
    
    @staticmethod
    def _load_meta_files(checkpoint_dirs):
        meta_files = []
        for ckpt_dir in checkpoint_dirs:
            with open(ckpt_dir / METAFNAME) as fi:
                meta_files.append(Checkpoint(ckpt_dir, yaml.safe_load(fi)))
        return meta_files

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


