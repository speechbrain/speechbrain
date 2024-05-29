"""The SpeechBrain implementation of various pre-trained model adapters e.g.
LoRA, Houlsby

Authors
 * Titouan Parcollet 2024
"""

import torch
import torch.nn as nn

from speechbrain.nnet.activations import Swish


def add_adapters_to_model(
    model: nn.Module,
    adapter_class: nn.Module,
    target_layers=["all-linear"],
    unfrozen_layers=[],
    **kwargs,
):
    """Given any torch model, e.g. asr_brain.modules.Transformer, and an adapter
    class, e.g. HoulsbyAdapter, this method will replace the target layers
    with this new adapter class (while preserving the parameters).

    Arguments
    ---------
    model: nn.Module
        The base PyTorch model.
    adapter_class: nn.Module
        A Module corresponding to one of the adapter of this (not initialized)
        SpeechBrain library.
    target_layers: list of str
        A list of module names in the given model that should be replaced.
        If the list includes "all-linear" then all linear layers will be
        replaced, and similarly for "all-conv" for convolution layers.
    unfrozen_layers: list of str
        List of layers to be unfrozen during training.
    **kwargs: dict
        Ensemble of parameters that should be given to the adapter.
    """

    # Collect and freeze layers
    replace_layers = []
    for name, module in model.named_modules():
        if is_layer_adaptable(name, module, target_layers):
            replace_layers.append(name)
        elif name not in unfrozen_layers:
            module.requires_grad = False

    # Replace the collected layer names
    for name in replace_layers:
        module = model.get_submodule(name)
        new_module = adapter_class(module, **kwargs)
        replace_module(model, name, new_module)


def is_layer_adaptable(name, module, target_layers):
    """Check if layer is among list of layers to be adapted.

    Arguments
    ---------
    name: str
        The name of the module to check.
    module: torch.nn.Module
        The module to check.
    target_layers: str or list of str
        See `add_adapters_to_model`

    Returns
    -------
    bool
        Whether the layer is to be adapted or not.
    """
    if "all-linear" in target_layers and isinstance(module, nn.Linear):
        return True
    if "all-conv" in target_layers and isinstance(
        module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    ):
        return True
    if name and name in target_layers:
        return True
    return False


def replace_module(model: nn.Module, name: str, new_module: nn.Module):
    """Replace layer with a new module based on a parent assignation.
    This is used to replace layers with an Adapter layer wrapped around
    the original layer. Hence, old parameters are preserved and new ones are
    added.

    Arguments
    ---------
    model: nn.Module
        Model containing the module to be replaced.
    name: str
        Name of the target module to replace.
    new_module: nn.Module
        New module made of the old plus the new parameters.
    """

    parent_name, target_name = name.rsplit(".", 1)
    parent_module = model.get_submodule(parent_name)

    setattr(parent_module, target_name, new_module)


class HoulsbyAdapterLinear(nn.Module):
    """This class implements the Houlsby Adapter as described in:
    'Parameter-Efficient Transfer Learning for NLP'
    https://arxiv.org/abs/1902.00751

    Arguments
    ---------
    target_linear: nn.Module
        Module corresponding to the pretrained Linear that will be wrapped with
        this adapter.
    projection_size: int
        Size of the projection layer (usually smaller).
    activation: nn.Module
        The activation function. Default is Swish.
    bias: bool
        Whether to use biases in the linear projections.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 64))
    >>> base_linear = nn.Linear(64, 64)
    >>> adapt = HoulsbyAdapterLinear(base_linear, 8)
    >>> output = adapt(x)
    >>> output.shape
    torch.Size([8, 60, 64])
    """

    def __init__(
        self,
        target_linear,
        projection_size,
        activation=Swish,
        bias=True,
    ):
        super().__init__()

        if not isinstance(target_linear, nn.linear):
            raise ValueError(
                "HoulsbyLinear currently only supports linear layers, "
                f"but instead got {type(target_linear)}."
            )

        output_size = target_linear.weight.data.shape[0]
        device = target_linear.weight.device

        self.pretrained_linear = target_linear
        self.pretrained_linear.requires_grad = False
        self.adapter_down_proj = nn.Linear(
            output_size, projection_size, bias=bias, device=device
        )
        self.adapter_up_proj = nn.Linear(
            projection_size, output_size, bias=bias, device=device
        )
        self.activation = activation()

        if bias:
            self.adapter_down_proj.bias.data.fill_(0.0)
            self.adapter_up_proj.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor):
        """Applies the HoulsbyAdapter to an input tensor `x`.

        Arguments
        ---------
        x: torch.Tensor
            Input tensor to the adapter module. Shape: [B, Time, X]

        Returns
        -------
        The linear outputs
        """

        x_pretrained = self.pretrained_linear(x)

        return (
            self.adapter_up_proj(
                self.activation(self.adapter_down_proj(x_pretrained))
            )
            + x_pretrained
        )


class LoRA(nn.Module):
    """This class implements the LoRA Adapter as described in:
    'LoRA: Low-Rank Adaptation of Large Language Models'
    https://arxiv.org/abs/2106.09685

    Arguments
    ---------
    target_module: nn.Module
        Module corresponding to the pretrained layer that will be wrapped with
        this adapter. Works with nn.Linear and nn.Conv
    rank: int
        Size of the projection layer or rank (usually smaller).
    alpha : float
        Value used to control the scaling in LoRA. Default is one.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 64))
    >>> base_linear = nn.Linear(64, 64)
    >>> adapt = LoRA(base_linear, 64, 4)
    >>> output = adapt(x)
    >>> output.shape
    torch.Size([8, 60, 64])
    """

    def __init__(self, target_module, rank=16, alpha=1.0):
        super().__init__()

        input_size = target_module.weight.data.shape[1]
        output_size = target_module.weight.data.shape[0]

        self.pretrained_module = target_module
        self.pretrained_module.requires_grad = False
        device = target_module.weight.device

        self.adapter_down_proj = nn.Linear(
            input_size, rank, bias=False, device=device
        )
        self.adapter_up_proj = nn.Linear(
            rank, output_size, bias=False, device=device
        )
        self.adapter_up_proj.weight.data.fill_(0.0)

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor):
        """Applies the LoRA Adapter.

        Arguments
        ---------
        x: torch.Tensor
            Input tensor to the adapter module.

        Returns
        -------
        The linear outputs
        """
        x_pretrained = self.pretrained_module(x)
        x_lora = self.adapter_up_proj(self.adapter_down_proj(x)) * self.scaling

        return x_pretrained + x_lora
