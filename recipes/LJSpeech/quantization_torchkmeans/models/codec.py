"""Neural codec based on semantic discrete audio representations.

Authors
 * Luca Della Libera 2024
"""

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


__all__ = ["Codec"]


class Codec(nn.Module):
    """Neural codec based on semantic discrete audio representations.

    Parameters
    ----------
    encoder:
        The underlying continuous model whose representations are to be quantized, i.e.
        a module that receives as an input a waveform and returns the corresponding
        continuous representations.
    quantizer:
        The quantizer, i.e. a module that receives as an input continuous representations
        and returns the corresponding tokens and quantized representations.
    dequantizer:
        The dequantizer, i.e. a module that receives as an input quantized representations
        and returns the corresponding continuous representations.
    decoder:
        The decoder, i.e. a module that receives as an input continuous representations
        and returns the corresponding waveform.
    layer_ids:
        The encoder layer IDs from which continuous representations are extracted.

    Examples
    --------
    >>> from speechbrain.lobes.models.HifiGAN import HifiganGenerator
    >>> from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
    >>> from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
    >>>
    >>> from kmeans import MultiKMeans
    >>> from transformer import TransformerDecoder
    >>>
    >>> layer_ids = [6, 7]
    >>> num_features = 768
    >>> num_clusters = [300, 300]
    >>>
    >>> encoder = WavLM(
    ...     source="microsoft/wavlm-base",
    ...     save_path="savedir",
    ...     output_all_hiddens=True,
    ... )
    >>> quantizer = MultiKMeans(num_features, num_clusters)
    >>> dequantizer = TransformerDecoder(
    ...     TransformerASR(
    ...         input_size=num_features,
    ...         tgt_vocab=1,
    ...         d_model=128,
    ...         nhead=4,
    ...         num_encoder_layers=6,
    ...         num_decoder_layers=0,
    ...         d_ffn=512,
    ...     ),
    ... )
    >>> decoder = HifiganGenerator(
    ...     in_channels=128 * len(num_clusters),
    ...     out_channels=1,
    ...     resblock_type="1",
    ...     resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ...     resblock_kernel_sizes=[3, 7, 11],
    ...     upsample_kernel_sizes=[16, 16, 4, 4],
    ...     upsample_initial_channel=512,
    ...     upsample_factors=[8, 8, 2, 2],
    ... )
    >>>
    >>> codec = Codec(encoder, quantizer, dequantizer, decoder, layer_ids)
    >>>
    >>> wavs = torch.rand([10, 16000])
    >>> wavs_pred = codec(wavs)

    """

    def __init__(
        self,
        encoder: "nn.Module",
        quantizer: "nn.Module",
        dequantizer: "nn.Module",
        decoder: "nn.Module",
        layer_ids: "Sequence[int]",
    ) -> "None":
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.dequantizer = dequantizer
        self.decoder = decoder
        self.layer_ids = layer_ids
        assert hasattr(self.encoder, "model")
        assert hasattr(self.encoder.model, "encoder")
        assert hasattr(self.encoder.model.encoder, "layers")

    def forward(
        self, wav: "Tensor", length: "Optional[Tensor]" = None
    ) -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        wav:
            The input waveform, shape (B, T).
        length:
            The relative length, shape (B,).

        Returns
        -------
            The reconstructed waveform, shape (B, T).

        """
        _, discrete_feats = self.encode_discrete(wav, length)
        wav_pred = self.decode_discrete(discrete_feats, length)
        return wav_pred

    def encode_discrete(
        self, wav: "Tensor", length: "Optional[Tensor]" = None
    ) -> "Tuple[Tensor, Tensor]":
        feats = self.encode(wav, length)
        tokens, discrete_feats = self.quantize(feats)
        return tokens, discrete_feats

    def decode_discrete(
        self, discrete_feats: "Tensor", length: "Optional[Tensor]" = None
    ) -> "Tensor":
        feats = self.dequantize(discrete_feats, length)
        wav = self.decode(feats)
        return wav

    def encode(
        self, wav: "Tensor", length: "Optional[Tensor]" = None
    ) -> "Tensor":
        # Workaround for early exiting to avoid the computational overhead of forwarding through the whole model
        out = self.encoder(wav, length)
        layers_backup = self.encoder.model.encoder.layers
        # NOTE: + 1 to account for layer norm applied to the last hidden states in WavLM:
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/wavlm/modeling_wavlm.py#L816
        self.encoder.model.encoder.layers = layers_backup[
            : max(self.layer_ids) + 1
        ]
        feats = self.encoder(wav, length)  # (K, B, N, H)
        self.encoder.model.encoder.layers = layers_backup
        return feats[self.layer_ids].movedim(0, -1)

    def quantize(self, feats: "Tensor") -> "Tuple[Tensor, Tensor]":
        tokens, discrete_feats = self.quantizer(feats)
        return tokens, discrete_feats

    def dequantize(
        self, discrete_feats: "Tensor", length: "Optional[Tensor]" = None
    ) -> "Tensor":
        B, N, H, K = discrete_feats.shape
        discrete_feats = discrete_feats.movedim(-1, 0).flatten(end_dim=1)
        if length is not None:
            length = length.expand(K, -1).flatten()
        feats = self.dequantizer(discrete_feats, length)
        feats = feats.reshape(K, B, N, -1).movedim(0, -1)
        return feats

    def decode(self, feats: "Tensor") -> "Tensor":
        feats = feats.flatten(start_dim=-2).movedim(-1, -2)
        wav = self.decoder(feats)
        return wav[:, 0]

    @classmethod
    def from_hparams(
        cls, hparams_file: "str", savedir: "str" = "savedir"
    ) -> "Codec":
        import pathlib
        from hyperpyyaml import load_hyperpyyaml

        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin)

        codec = hparams["codec"]
        checkpointer = hparams["checkpointer"]
        decoder_path = str(
            checkpointer.find_checkpoint(min_key="loss").paramfiles["decoder"]
        )
        pretrainer = hparams["pretrainer"]
        pretrainer.add_loadables({"decoder": codec.decoder})
        pretrainer.add_paths({"decoder": decoder_path})
        pretrainer.collect_in = pathlib.Path(savedir)
        pretrainer.collect_files()
        pretrainer.load_collected()

        return cls(
            codec.encoder,
            codec.quantizer,
            codec.dequantizer,
            codec.decoder,
            codec.layer_ids,
        )


# Test
if __name__ == "__main__":
    from speechbrain.lobes.models.HifiGAN import HifiganGenerator
    from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
    from speechbrain.lobes.models.transformer.TransformerASR import (
        TransformerASR,
    )

    from kmeans import MultiKMeans
    from transformer import TransformerDecoder

    layer_ids = [6, 7]
    num_features = 768
    num_clusters = [300, 300]

    encoder = WavLM(
        source="microsoft/wavlm-base",
        save_path="savedir",
        output_all_hiddens=True,
    )
    quantizer = MultiKMeans(num_features, num_clusters)
    dequantizer = TransformerDecoder(
        TransformerASR(
            input_size=num_features,
            tgt_vocab=1,
            d_model=128,
            nhead=4,
            num_encoder_layers=6,
            num_decoder_layers=0,
            d_ffn=512,
        ),
    )
    decoder = HifiganGenerator(
        in_channels=128 * len(num_clusters),
        out_channels=1,
        resblock_type="1",
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes=[3, 7, 11],
        upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=512,
        upsample_factors=[8, 8, 2, 2],
    )

    codec = Codec(encoder, quantizer, dequantizer, decoder, layer_ids)

    wavs = torch.rand([10, 16000])
    wavs_pred = codec(wavs)
    print(wavs_pred.shape)
