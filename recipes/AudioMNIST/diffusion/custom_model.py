"""Custom models for diffusion

Authors
 * Artem Ploujnikov 2022
"""
from torch import nn
from speechbrain.dataio.dataio import length_to_mask


class DoneDetector(nn.Module):
    """A wrapper for the done detector using a model (e.g. a CRDNN) and
    an output layer.

    The goal of using a wrapper is to apply masking before the output layer
    (e.g. Softmax) so that the model can't "cheat" by outputting probabilities
    in the masked area
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
