"""Common utilities.

Authors
 * Luca Della Libera 2026
"""

import os

import torch

__all__ = ["ActivationCache", "download_wavlm6"]


class ActivationCache:
    """Register forward hooks on selected submodules and cache inputs and outputs.

    Arguments
    ---------
    model:
        PyTorch module containing the target submodules.
    module_names:
        Name or list of names of the submodules to hook. Nested
        submodules are supported, e.g. ``"feature_proj"`` or
        ``"encoder.layers.3"``.
    detach:
        Whether to detach captured tensors from the computation graph.
    clone:
        Whether to clone captured tensors before storing them.
    cpu:
        Whether to move captured tensors to CPU before storing them.

    Attributes
    ----------
    inputs:
        Dictionary mapping submodule names to their most recently
        cached inputs.
    outputs:
        Dictionary mapping submodule names to their most recently
        cached outputs.
    hooks:
        List of registered forward hook handles.

    """

    def __init__(
        self,
        model,
        module_names,
        detach: "bool" = True,
        clone: "bool" = False,
        cpu: "bool" = False,
    ):
        """Initialize the cache and register forward hooks.

        Arguments
        ---------
        model:
            PyTorch module containing the target submodules.
        module_names:
            Name or list of names of the submodules to hook. Nested
            submodules are supported, e.g. ``"feature_proj"`` or
            ``"encoder.layers.3"``.
        detach:
            Whether to detach captured tensors from the computation graph.
        clone:
            Whether to clone captured tensors before storing them.
        cpu:
            Whether to move captured tensors to CPU before storing them.

        Raises
        ------
        ValueError
            If one of the requested submodules is not found.

        """
        if isinstance(module_names, str):
            module_names = [module_names]

        self.inputs = {}
        self.outputs = {}
        self.hooks = []
        self.detach = detach
        self.clone = clone
        self.cpu = cpu

        named_modules = dict(model.named_modules())

        def process(x):
            """Recursively process cached tensors."""
            if isinstance(x, tuple):
                return tuple(process(xx) for xx in x)
            if isinstance(x, list):
                return [process(xx) for xx in x]
            if isinstance(x, dict):
                return {k: process(v) for k, v in x.items()}

            if self.detach and hasattr(x, "detach"):
                x = x.detach()
            if self.clone and hasattr(x, "clone"):
                x = x.clone()
            if self.cpu and hasattr(x, "cpu"):
                x = x.cpu()

            return x

        for name in module_names:
            if name not in named_modules:
                available = ", ".join(sorted(named_modules.keys()))
                raise ValueError(
                    f"Submodule '{name}' not found. "
                    f"Available modules: {available}"
                )

            def make_hook(module_name):
                def hook(module, inputs, output):
                    self.inputs[module_name] = process(inputs)
                    self.outputs[module_name] = process(output)

                return hook

            handle = named_modules[name].register_forward_hook(make_hook(name))
            self.hooks.append(handle)

    def clear(self):
        """Clear all cached inputs and outputs."""
        self.inputs.clear()
        self.outputs.clear()

    def remove(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def download_wavlm6(
    cache_dir: "str",
    checkpoint_name: "str",
    positional_embedding_checkpoint: "str | None" = None,
    positional_embedding_only: "bool" = False,
) -> "str":
    """Download WavLM6 checkpoint to cache and return the path.

    Arguments
    ---------
    cache_dir:
        Cache directory where the checkpoint will be saved.
    checkpoint_name:
        Name of the checkpoint file to save.
    positional_embedding_checkpoint:
        Optional path to a positional embedding checkpoint whose weights
        will overwrite the current positional embedding weights in the
        model before saving.
    positional_embedding_only:
        Whether to save only the positional embedding state dict from
        ``codec.encoder.encoder.positional_embedding``.

    Returns
    -------
        Path to the saved checkpoint.

    """
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, checkpoint_name)

    # If already cached, return immediately
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # Load FocalCodec model
    codec = torch.hub.load(
        repo_or_dir="lucadellalib/focalcodec",
        model="focalcodec",
        config="lucadellalib/focalcodec_50hz",
    )

    # Optionally overwrite positional embedding weights
    if positional_embedding_checkpoint is not None:
        positional_embedding_state_dict = torch.load(
            positional_embedding_checkpoint,
            map_location="cpu",
        )
        codec.encoder.encoder.positional_embedding.load_state_dict(
            positional_embedding_state_dict
        )

    # Get state dict
    if positional_embedding_only:
        state_dict = codec.encoder.encoder.positional_embedding.state_dict()
    else:
        state_dict = codec.encoder.state_dict()

    # Save checkpoint
    torch.save(state_dict, checkpoint_path)

    return checkpoint_path
