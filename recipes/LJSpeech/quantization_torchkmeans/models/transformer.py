"""Transformer models.

Authors
 * Luca Della Libera 2024
"""

from typing import Optional

import torch
from torch import Tensor, nn


__all__ = ["TransformerDecoder"]


class TransformerDecoder(nn.Module):
    """Transformer decoder model.

    Parameters
    ----------
    frontend:
        The frontend.
    backbone:
        The backbone.
    head:
        The head.

    Examples
    --------
    >>> from speechbrain.lobes.models.convolution import ConvolutionFrontEnd
    >>> from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
    >>> from torch import nn
    >>>
    >>> input_size = 256
    >>> d_model = 128
    >>> out_channels = (28, 28, 28)
    >>> strides = [1, 2, 2]
    >>> frontend = ConvolutionFrontEnd([None, None, input_size], out_channels=out_channels, strides=strides)
    >>> backbone = TransformerASR(
    ...     input_size=input_size // torch.Size(strides).numel() * out_channels[-1],
    ...     tgt_vocab=1,
    ...     num_decoder_layers=0,
    ...     d_model=d_model,
    ... )
    >>> head = nn.Linear(d_model, input_size)
    >>>
    >>> decoder = TransformerDecoder(frontend=frontend, backbone=backbone, head=head)
    >>>
    >>> input = torch.rand([10, 200, input_size])
    >>> output = decoder(input)

    """

    def __init__(
        self,
        backbone: "nn.Module",
        frontend: "nn.Module" = nn.Identity(),
        head: "nn.Module" = nn.Identity(),
    ) -> "None":
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone
        self.head = head

    def forward(
        self, src: "Tensor", length: "Optional[Tensor]" = None
    ) -> "Tensor":
        output = self.frontend(src)
        if hasattr(self.backbone, "encode"):
            # Transformer ASR
            output = self.backbone.encode(output, length)
        else:
            output = self.backbone(output, length)
        output = self.head(output)
        return output


# Test
if __name__ == "__main__":
    from speechbrain.lobes.models.convolution import ConvolutionFrontEnd
    from speechbrain.lobes.models.transformer.TransformerASR import (
        TransformerASR,
    )

    input_size = 256
    d_model = 128
    out_channels = (28, 28, 28)
    strides = [1, 2, 2]
    frontend = ConvolutionFrontEnd(
        [None, None, input_size], out_channels=out_channels, strides=strides
    )
    backbone = TransformerASR(
        input_size=input_size // torch.Size(strides).numel() * out_channels[-1],
        tgt_vocab=1,
        num_decoder_layers=0,
        d_model=d_model,
    )
    head = nn.Linear(d_model, input_size)

    decoder = TransformerDecoder(
        frontend=frontend, backbone=backbone, head=head
    )

    input = torch.rand([10, 200, input_size])
    output = decoder(input)
    print(output.shape)
