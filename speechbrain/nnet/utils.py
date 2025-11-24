"""
Assorted reusable neural network modules.

Authors
 * Artem Ploujnikov 2023
"""

from torch import nn

from speechbrain.dataio.dataio import length_to_mask


class DoneDetector(nn.Module):
    """A wrapper for the done detector using a model (e.g. a CRDNN) and
    an output layer.

    The goal of using a wrapper is to apply masking before the output layer
    (e.g. Softmax) so that the model can't "cheat" by outputting probabilities
    in the masked area

    Arguments
    ---------
    model: torch.nn.Module
        the model used to make the prediction
    out: torch.nn.Module
        the output function

    Example
    -------
    >>> import torch
    >>> from torch import nn
    >>> from speechbrain.nnet.activations import Softmax
    >>> from speechbrain.nnet.containers import Sequential
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.lobes.models.CRDNN import CRDNN
    >>> crdnn = CRDNN(
    ...     input_size=80,
    ...     cnn_blocks=1,
    ...     cnn_kernelsize=3,
    ...     rnn_layers=1,
    ...     rnn_neurons=16,
    ...     dnn_blocks=1,
    ...     dnn_neurons=16,
    ... )
    >>> model_out = Linear(n_neurons=1, input_size=16)
    >>> model_act = nn.Sigmoid()
    >>> model = Sequential(crdnn, model_out, model_act)
    >>> out = Softmax(
    ...     apply_log=False,
    ... )
    >>> done_detector = DoneDetector(
    ...     model=model,
    ...     out=out,
    ... )
    >>> preds = torch.randn(4, 10, 80)  # Batch x Length x Feats
    >>> length = torch.tensor([1.0, 0.8, 0.5, 1.0])
    >>> preds_len = done_detector(preds, length)
    >>> preds_len.shape
    torch.Size([4, 10, 1])
    """

    def __init__(self, model, out):
        super().__init__()
        self.model = model
        self.out = out

    def forward(self, feats, length=None):
        """Computes the forward pass

        Arguments
        ---------
        feats: torch.Tensor
            the features used for the model (e.g. spectrograms)
        length: torch.Tensor
            a tensor of relative lengths

        Returns
        -------
        preds: torch.Tensor
            predictions
        """
        out = self.model(feats)
        if length is not None:
            max_len = feats.size(1)
            mask = length_to_mask(length=length * max_len, max_len=max_len)
            out = out * mask.unsqueeze(-1)
        out = self.out(out)
        return out
