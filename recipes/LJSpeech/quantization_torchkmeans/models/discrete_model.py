"""Discrete representation model.

Authors
 * Luca Della Libera 2024
"""

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


__all__ = ["DiscreteModel"]


class DiscreteModel(nn.Module):
    """Discrete representation model.

    Parameters
    ----------
    continuous_model:
        The underlying continuous model whose representations are to be quantized.
    layer_ids:
        The layer IDs from which representations are extracted.
    quantizers:
        The quantizers (one for each layer ID).
        A quantizer is a module that receives as an input continuous representations
        and returns the corresponding tokens and quantized representations.

    Examples
    --------
    >>> import torch
    >>> from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
    >>>
    >>> from kmeans import MultiKMeans
    >>>
    >>> layer_ids = [6, 7]
    >>> num_features = [768, 768]
    >>> num_clusters = [300, 300]
    >>>
    >>> continuous_wavlm = WavLM(
    ...     source="microsoft/wavlm-base", save_path="savedir", output_all_hiddens=True
    ... )
    >>> quantizers = MultiKMeans(num_features, num_clusters)
    >>> discrete_wavlm = DiscreteModel(continuous_wavlm, layer_ids, quantizers)
    >>>
    >>> wavs = torch.rand([10, 600])
    >>> tokens, discrete_feats = discrete_wavlm(wavs)

    """

    def __init__(
        self,
        continuous_model: "nn.Module",
        layer_ids: "Sequence[int]",
        quantizers: "Sequence[nn.Module]",
    ) -> "None":
        super().__init__()
        self.continuous_model = continuous_model
        self.layer_ids = layer_ids
        self.quantizers = torch.nn.ModuleList(quantizers)
        assert hasattr(self.continuous_model, "model")
        assert hasattr(self.continuous_model.model, "encoder")
        assert hasattr(self.continuous_model.model.encoder, "layers")

    def forward(
        self, wav: "Tensor", wav_lens: "Optional[Tensor]" = None
    ) -> "Tuple[Tensor, Tensor]":
        """Forward pass.

        Parameters
        ----------
        wav:
            The input waveform, shape (B, T).
        wav_lens:
            The input waveform relative lengths, shape (B,).

        Returns
        -------
            - The tokens, shape (B, N, K).
            - The corresponding quantized representations, shape (B, N, H, K).

        """
        feats = self.forward_continuous(wav, wav_lens)
        tokens, discrete_feats = self.forward_discrete(feats)
        return tokens, discrete_feats

    def forward_continuous(
        self, wav: "Tensor", wav_lens: "Optional[Tensor]" = None
    ) -> "Tensor":
        # Workaround for early exiting to avoid the computational
        # overhead of forwarding through the whole model
        layers_backup = self.continuous_model.model.encoder.layers
        self.continuous_model.model.encoder.layers = layers_backup[
            : max(self.layer_ids)
        ]
        feats = self.continuous_model(wav, wav_lens)  # (K, B, N, H)
        self.continuous_model.model.encoder.layers = layers_backup
        return feats[self.layer_ids]

    def forward_discrete(self, feats: "Tensor") -> "Tuple[Tensor, Tensor]":
        assert len(feats) == len(self.quantizers)

        if len(self.quantizers) == 1:
            # Fast path
            tokens, discrete_feats = self.quantizers[0](feats[0])
            return tokens[..., None], discrete_feats[..., None]

        tokens_list, discrete_feats_list = [], []
        for i, quantizer in enumerate(self.quantizers):
            tokens, discrete_feats = quantizer(feats[i])
            tokens_list.append(tokens)
            discrete_feats_list.append(discrete_feats)
        tokens = torch.stack(tokens_list).movedim(0, -1)
        discrete_feats = torch.stack(discrete_feats_list).movedim(0, -1)

        return tokens, discrete_feats


# Test
if __name__ == "__main__":
    from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM

    from kmeans import MultiKMeans

    layer_ids = [6, 7]
    num_features = [1024, 1024]
    num_clusters = [300, 300]

    continuous_wavlm = WavLM(
        source="microsoft/wavlm-large",
        save_path="savedir",
        output_all_hiddens=True,
    )
    quantizers = MultiKMeans(num_features, num_clusters)
    discrete_wavlm = DiscreteModel(continuous_wavlm, layer_ids, quantizers)

    # Set checkpoint path to load a checkpoint (e.g. ../results/kmeans_wavlm/0/save/CKPT+2024-02-14+17-14-04+00/discrete_wavlm.ckpt)
    checkpoint_path = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        discrete_wavlm.load_state_dict(checkpoint)

    wavs = torch.rand([10, 16000])
    tokens, discrete_feats = discrete_wavlm(wavs)
    print(tokens.shape)
    print(discrete_feats.shape)
