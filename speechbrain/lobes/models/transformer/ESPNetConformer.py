"""
Conformer speech translation model (pytorch).
It is a fusion of `e2e_st_transformer.py`
Refer to: https://arxiv.org/abs/2005.08100
"""

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from speechbrain.lobes.models.transformer.ESPNetTransformer import (
    E2E as E2ETransformer,
)


class E2E(E2ETransformer):
    """E2E module.
    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        adim: int,
        aheads: int,
        wshare: int,
        ldconv_encoder_kernel_length: int,
        ldconv_usebias: bool,
        eunits: int,
        elayers: int,
        transformer_input_layer: str,
        transformer_encoder_selfattn_layer_type: str,
        transformer_decoder_selfattn_layer_type: str,
        ldconv_decoder_kernel_length: int,
        dunits: int,
        dlayers: int,
        transformer_encoder_pos_enc_layer_type: str,
        transformer_encoder_activation_type: str,
        macaron_style: bool = True,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        dropout_rate: float = 0.1,
        transformer_attn_dropout_rate: float = 0,
        sos: int = 1,
        eos: int = 2,
        ignore_id: int = -1,
    ):
        """Construct an E2E object.
        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super().__init__(
            idim,
            odim,
            adim,
            aheads,
            wshare,
            ldconv_encoder_kernel_length,
            ldconv_usebias,
            eunits,
            elayers,
            transformer_input_layer,
            transformer_encoder_selfattn_layer_type,
            transformer_decoder_selfattn_layer_type,
            ldconv_decoder_kernel_length,
            dunits,
            dlayers,
            dropout_rate,
            transformer_attn_dropout_rate,
            sos,
            eos,
            ignore_id,
        )

        self.encoder = Encoder(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=transformer_input_layer,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            attention_dropout_rate=transformer_attn_dropout_rate,
            pos_enc_layer_type=transformer_encoder_pos_enc_layer_type,
            selfattention_layer_type=transformer_encoder_selfattn_layer_type,
            activation_type=transformer_encoder_activation_type,
            macaron_style=macaron_style,
            use_cnn_module=use_cnn_module,
            cnn_module_kernel=cnn_module_kernel,
        )
