"""CNN Transformer model for SE in the SpeechBrain style

Authors
* Chien-Feng Liao 2020
"""
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np
import torch  # noqa E402
from torch import nn
from speechbrain.nnet.linear import Linear
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerInterface,
    get_lookahead_mask,
)


class CNNTransformerSE(TransformerInterface):
    """This is an implementation of transformer model with CNN pre-encoder for SE

    Arguements
    ----------
    output_size: int
        the number of expected neurons in the output layer.
    output_activation: torch class
        the activation function of output layer, (default=ReLU).
    nhead: int
        the number of heads in the multiheadattention models (default=8).
    num_layers: int
        the number of sub-layers in the transformer (default=8).
    d_ffn: int
        the number of expected features in the encoder layers (default=512).
    dropout: int
        the dropout value (default=0.1).
    activation: torch class
        the activation function of intermediate layers (default=LeakyReLU).
    causal: bool
        True for causal setting, the model is forbidden to see future frames. (default=True)
    custom_emb_module: torch class
        module that process the input features before the transformer model.
    Example
    -------
    >>> src = torch.rand([8, 120, 256])
    >>> net = CNNTransformerSE(257)
    >>> out = net.forward(src, init_params=True)
    >>> print(out.shape)
    torch.Size([8, 120, 257])
    """

    def __init__(
        self,
        output_size,
        output_activation=nn.ReLU,
        nhead=8,
        num_layers=8,
        d_ffn=512,
        dropout=0.1,
        activation=nn.LeakyReLU,
        causal=True,
        custom_emb_module=None,
        num_modules=1,
        use_group_comm=False,
    ):
        super().__init__(
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            return_attention=True,
            positional_encoding=False,
            num_modules=num_modules,
            use_group_comm=use_group_comm,
        )

        self.custom_emb_module = custom_emb_module
        self.causal = causal
        self.output_layer = Linear(output_size, bias=False)
        self.output_activation = output_activation()

    def forward(self, x, src_key_padding_mask=None, init_params=False):
        if self.causal:
            self.attn_mask = get_lookahead_mask(x)
        else:
            self.attn_mask = None

        if self.custom_emb_module is not None:
            src = self.custom_emb_module(x, init_params)

        encoder_output, attn_list = self.encoder(
            src=src,
            src_mask=self.attn_mask,
            src_key_padding_mask=src_key_padding_mask,
            init_params=init_params,
        )

        output = self.output_layer(encoder_output, init_params)
        output = self.output_activation(output)
        # if not init_params:
        #     self.plot(attn_list, x)

        return output

    # def plot(self, attn_list, spec):
    #     clean_attn_list = []
    #     for attn in attn_list:
    #         if attn is not None:
    #             clean_attn_list.append(attn.cpu().detach().numpy())

    #     # number of batch
    #     for bs in range(clean_attn_list[0].shape[0]):
    #         plt.figure(figsize=[10, 2 * len(clean_attn_list)])

    #         plt.subplot(len(clean_attn_list) + 1, 1, 1)
    #         spec_tmp = (
    #             torch.log(torch.expm1(spec[bs, :, :])).cpu().detach().numpy()
    #         )
    #         plt.imshow(
    #             spec_tmp[:, ::-1].T, interpolation="nearest", aspect="auto"
    #         )

    #         # number of layers
    #         for i, attn in enumerate(clean_attn_list):
    #             plt.subplot(len(clean_attn_list) + 1, 1, i + 2)
    #             plt.xlim([0, attn.shape[1]])

    #             # number of competitions
    #             for n in range(attn.shape[-1]):
    #                 if n == 0:
    #                     bottom = 0
    #                 elif n == 1:
    #                     bottom = attn[bs, :, 0]
    #                 elif n == 2:
    #                     bottom = attn[bs, :, 0] + attn[bs, :, 1]
    #                 elif n == 3:
    #                     bottom = (
    #                         attn[bs, :, 0] + attn[bs, :, 1] + attn[bs, :, 2]
    #                     )
    #                 plt.bar(
    #                     np.arange(attn.shape[1]),
    #                     attn[bs, :, n],
    #                     width=1,
    #                     bottom=bottom,
    #                 )

    #         plt.tight_layout()
    #         plt.savefig("test{}.png".format(bs))
