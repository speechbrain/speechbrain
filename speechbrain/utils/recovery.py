import torch
import types
import collections.abc
import os
import os.path
import time
import uuid
import yaml
import pathlib
# os.path.getmtime

CKPT_PREFIX = "CKPT"
METAFNAME = f"{CKPT_PREFIX}.yaml"

def _latest_ckpt_keyfunc(ckpt_tuple):
    # Negative of unixtime -> latest first
    metadict = ckpt_tuple[1]
    return - metadict["unixtime"]

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
        self._save_checkpoint_metafile(ckpt_dir / METAFNAME, meta)
        for name, obj in self.recoverables.items():
            objfname = f"{name}.pt"
            torch.save(obj.state_dict(), objfname)

    def lazy_recovery_if_possible(self, 
            ckpt_sort_key = _latest_ckpt_keyfunc):
        ckpts = load_meta_files(self.list_checkpoint_dirs())
        if ckpts:
            chosen_ckpt_path, meta = sorted(ckpts, ckpt_sort_key)[0]
            self.add_lazy_load_hooks(chosen_ckpt_path)

    def add_lazy_load_hooks(self, checkpoint_path):
        for name, obj in self.recoverables.items():
            objfname = f"{name}.pt"
            objpath = checkpoint_path / objfname
            if not objpath.exists():
                if self.allow_partial_load:
                    continue
                else:
                    MSG = f"Loading checkpoint from {checkpoint_path}, \
                            expected {name}.pt to exist"
                    raise ValueError(MSG)

            def lazy_load_hook(obj, input):
                obj.load_state_dict(torch.load(objfname))
                obj._lazy_recovery_hook.remove()

            obj._lazy_recovery_hook = \
                    obj.register_forward_pre_hook(lazy_load_hook)

    def list_checkpoint_dirs(self):
        return [x for x in self.checkpoints_dir.iterdir()
                if Recoverer._is_checkpoint_dir(x)]
    
    @staticmethod
    def load_meta_files(checkpoint_dirs):
        return [(ckpt_dir, yaml.safe_load(ckpt_dir / METAFNAME))
                for ckpt_dir in checkpoints_dirs]

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
