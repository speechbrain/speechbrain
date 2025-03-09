from dataclasses import dataclass, field,asdict
from typing import Optional, Dict

@dataclass
class RunOptDefaults:
    test_only: bool = False
    debug: bool = False
    debug_batches: int = 2
    debug_epochs: int = 2
    debug_persistently: bool = False
    device: str = "cpu"
    data_parallel_backend: bool = False
    data_parallel_count: int = -1
    distributed_backend: str = "nccl"
    distributed_launch: bool = False
    find_unused_parameters: bool = False
    jit: bool = False
    jit_module_keys: Optional[None] = None
    compile: bool = False
    compile_module_keys: Optional[None] = None
    compile_mode: str = "default"
    compile_using_fullgraph: bool = False
    compile_using_dynamic_shape_tracing: bool = True
    precision: str = "fp32"
    eval_precision: str = "fp32"
    auto_mix_prec: bool = False
    bfloat16_mix_prec: bool = False
    max_grad_norm: float = 5.0
    skip_nonfinite_grads: bool = False
    nonfinite_patience: int = 3
    noprogressbar: bool = False
    ckpt_interval_minutes: int = 0
    ckpt_interval_steps: int = 0
    grad_accumulation_factor: int = 1
    optimizer_step_limit: Optional[None] = None
    tqdm_colored_bar: bool = False
    tqdm_barcolor: Dict[str, str] = field(default_factory=lambda: {"train": "GREEN", "valid": "MAGENTA", "test": "CYAN"})
    remove_vector_weight_decay: bool = False
    profile_training: bool = False
    profile_warmup: int = 5
    profile_steps: int = 5
    
    def as_dict(self) -> Dict:
        return asdict(self)
