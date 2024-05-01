"""The SpeechBrain implementation of various pre-trained model adapters e.g.
LoRA, Houlsby

Authors
 * Titouan Parcollet 2024
"""

import torch
import torch.nn as nn

from speechbrain.nnet.activations import Swish


def add_adapters_to_linear_in_model(
    model: torch.nn.Module,
    adapter_class: torch.nn.Module,
    **kwargs,
):
    """Given any torch model, e.g. asr_brain.modules.Transformer, and an adapter
    class, e.g. HoulsbyAdapter, this method will change all the linear layers
    with this new adapter class (while preserving the parameters).

    Arguments
    ---------
    model: torch.nn.Module
        The base PyTorch model.
    adapter_class: torch.nn.Module
        A Module corresponding to one of the adapter of this (not initialized)
        SpeechBrain library.
    kwargs: dict,
        Ensemble of parameters that should be given to the adapter.
    """

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_module, target_name, target_module = get_submodules(
                model, name
            )
            new_module = adapter_class(target_module, **kwargs)
            replace_linear(
                parent_module, target_name, target_module, new_module
            )


class HoulsbyAdapterLinear(nn.Module):
    """This class implements the Houlsby Adapter as described in:
    'Parameter-Efficient Transfer Learning for NLP'
    https://arxiv.org/abs/1902.00751

    Arguments
    ---------

    target_linear: torch.nn.Module
        Module corresponding to the pretrained Linear that will be wrapped with
        this adapter.
    input_size : int
        Size of the incoming feature vector (previous layer). Output size is the
        same.
    projection_size : int
        Size of the projection layer (usually smaller).
    activation : torch.nn.Module
        The activation function. Default is Swish.
    bias : bool
        Whether to use biases in the linear projections.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 64))
    >>> base_linear = torch.nn.Linear(64,64)
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

        output_size = target_linear.weight.data.shape[0]

        self.pretrained_linear = target_linear
        self.adapter_down_proj = nn.Linear(
            output_size, projection_size, bias=bias
        )
        self.adapter_up_proj = nn.Linear(
            projection_size, output_size, bias=bias
        )
        self.activation = activation()

        if bias:
            self.adapter_down_proj.bias.data.fill_(0.0)
            self.adapter_up_proj.bias.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,
    ):
        """Applies the HoulsbyAdapter to an input tensor `x`.

        Arguments
        ---------
        x: torch.Tensor
            Input tensor to the adapter module. Shape: [B, Time, X]
        """

        x_pretrained = self.pretrained_linear(x)

        return (
            self.adapter_up_proj(
                self.activation(self.adapter_down_proj(x_pretrained))
            )
            + x_pretrained
        )


class LoRALinear(nn.Module):
    """This class implements the LoRA Adapter as described in:
    'LoRA: Low-Rank Adaptation of Large Language Models'
    https://arxiv.org/abs/2106.09685

    Arguments
    ---------

    target_linear: torch.nn.Module
        Module corresponding to the pretrained Linear that will be wrapped with
        this adapter.
    input_size : int
        Size of the incoming feature vector (previous layer). Output size is the
        same.
    rank : int
        Size of the projection layer or rank (usually smaller).
    alpha : float
        Value used to control the scaling in LoRA. Default is one.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 64))
    >>> base_linear = torch.nn.Linear(64,64)
    >>> adapt = LoRALinear(base_linear, 64, 4)
    >>> output = adapt(x)
    >>> output.shape
    torch.Size([8, 60, 64])
    """

    def __init__(
        self,
        target_linear,
        rank=16,
        alpha=1.0,
    ):
        super().__init__()

        input_size = target_linear.weight.data.shape[1]
        output_size = target_linear.weight.data.shape[0]

        self.pretrained_linear = target_linear

        self.adapter_down_proj = nn.Linear(input_size, rank, bias=False)
        self.adapter_up_proj = nn.Linear(rank, output_size, bias=False)

        self.scaling = alpha / rank
        self.adapter_up_proj.weight.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,
    ):
        """Applies the LoRA Adapter.

        Arguments
        ---------
        x: torch.Tensor
            Input tensor to the adapter module. Shape: [B, Time, X]
        """
        x_pretrained = self.pretrained_linear(x)
        x_lora = self.adapter_up_proj(self.adapter_down_proj(x)) * self.scaling

        return x_pretrained + x_lora


def replace_linear(
    parent_module: torch.nn.Module,
    name: str,
    old_linear: torch.nn.Module,
    new_module: torch.nn.Module,
):
    """Replace linear layers with a new module based on a parent assignation.
    This is used to replace Linear layers with an Adapter layer wrapped around
    the original layer. Hence, old parameters are preserved and new ones are
    added.

    Arguments
    ---------
    parent_module: torch.nn.Module
        Parent module for the old module.
    name: str
        Name of the child module.
    old_linear: torch.nn.Module
        Module corresponding to the old linear layer.
    new_module: torch.nn.Module
        New module made of the old linear plus the new parameters.
    """

    device = old_linear.weight.device
    setattr(parent_module, name, new_module)

    new_module.weight = old_linear.weight
    if hasattr(old_linear, "bias") and old_linear.bias is not None:
        new_module.bias = old_linear.bias

    new_module.to(device)


def get_submodules(model: torch.nn.Module, name: str):
    """Get the parent module, the target name as well as the target module
    given a torch.nn.Module and a name (obtained from .named_modules()). We use
    this function to get the parent node of a given module that we want to
    replace with something else (e.g. an adapter).

    Arguments
    ---------
    model: torch.nn.Module
        The base PyTorch model.
    name: str
        Name of the child module to look for in the model.
    """
    parent_module = model.get_submodule(".".join(name.split(".")[:-1]))
    target_name = name.split(".")[-1]
    target_module = model.get_submodule(name)
    return parent_module, target_name, target_module
