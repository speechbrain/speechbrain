import torch
import torch.nn as nn


def get_EF(input_size, dim, method="convolution", head_dim=None, bias=True):
    """
    The methodology is inspired from the following GitHub repo:
    https://github.com/tatp22/linformer-pytorch/blob/master/linformer_pytorch/linformer_pytorch.py#L26

    For more info about the E and F matrices, refer to the following:
    Linformer: Self-Attention with Linear Complexity: https://arxiv.org/abs/2006.04768

    The function returns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    assert method in [
        "learnable",
        "convolution",
        "maxpool",
        "avgpool",
        "no_params",
    ], "The method flag needs to be either 'learnable', 'convolution', or 'no_params'."
    if method == "convolution":
        conv = nn.Conv1d(head_dim, head_dim, kernel_size=3, stride=3, padding=3)
        return conv
    if method == "maxpool":
        pool = nn.MaxPool1d(
            kernel_size=int(input_size / dim), stride=int(input_size / dim)
        )
        return pool
    if method == "avgpool":
        pool = nn.MaxPool1d(
            kernel_size=int(input_size / dim), stride=int(input_size / dim)
        )
        return pool
    if method == "no_params":
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1 / dim)
        return mat

    # The below is the case when using "learnable"
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin
