"""This lobe enables the integration of huggingface pretrained Mimi.

Mimi codec is a state-of-the-art audio neural codec, developed by Kyutai.
It combines semantic and acoustic information into audio tokens running at 12Hz and a bitrate of 1.1kbps.

Note that you need to install `transformers>=4.45.1` to use this module.

Repository: https://huggingface.co/kyutai/mimi
Paper: https://kyutai.org/Moshi.pdf

Authors
 * Pooneh Mousavi 2024
"""

import torch

from speechbrain.dataio.dataio import clean_padding_, length_to_mask
from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class Mimi(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace pretrained Mimi model.
    Mimi codec is a state-of-the-art audio neural codec, developed by Kyutai.
    It combines semantic and acoustic information into audio tokens running at 12Hz and a bitrate of 1.1kbps.

    Source paper:
       https://kyutai.org/Moshi.pdf

    Transformers>=4.45.1 from HuggingFace needs to be installed:
        https://huggingface.co/transformers/installation.html

    The code is adapted from the official HF Kyutai repository:
        https://huggingface.co/kyutai/mimi

    Arguments
    ---------
    source : str
        A HuggingFace repository identifier or a path
    save_path : str
        The location where the pretrained model will be saved
    sample_rate : int (default: 24000)
        The audio sampling rate
    freeze : bool
        whether the model will be frozen (e.g. not trainable if used as part of training another model)
    num_codebooks : int (default: 8)
        Number of codebooks. It could be [2,3,4,5,6,7,8]

    Example
    -------
    >>> model_hub = "kyutai/mimi"
    >>> save_path = "savedir"
    >>> model = Mimi(model_hub, save_path)
    >>> audio = torch.randn(4, 48000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens, emb = model.encode(audio, length)
    >>> tokens.shape
    torch.Size([4, 8, 25])
    >>> emb.shape
    torch.Size([4, 8, 25, 256])
    >>> rec = model.decode(tokens, length)
    >>> rec.shape
    torch.Size([4, 1, 48000])
    """

    def __init__(
        self,
        source,
        save_path=None,
        sample_rate=24000,
        freeze=True,
        num_codebooks=8,
    ):

        super().__init__(source=source, save_path=save_path, freeze=freeze)
        self.num_codebooks = num_codebooks
        self.sampling_rate = sample_rate
        self.embeddings = self._compute_embedding()

    @torch.no_grad()
    def _compute_embedding(self):
        semantic_layers = (
            self.model.quantizer.semantic_residual_vector_quantizer.layers
        )
        acoustic_layers = (
            self.model.quantizer.acoustic_residual_vector_quantizer.layers
        )
        layers = (semantic_layers + acoustic_layers)[: self.num_codebooks]
        embs = [layer.codebook.embed for layer in layers]
        embs = torch.stack(embs)  # [K, C, H]
        return embs

    def forward(self, inputs, length):
        """Encodes the input audio as tokens and embeddings and  decodes audio from tokens

        Arguments
        ---------
        inputs : torch.Tensor
            A (Batch x Samples) or (Batch x Channel x Samples)
            tensor of audio
        length : torch.Tensor
            A tensor of relative lengths

        Returns
        -------
        tokens : torch.Tensor
            A (Batch x Tokens x Heads) tensor of audio tokens
        emb : torch.Tensor
            Raw vector embeddings from the model's
            quantizers
        audio : torch.Tensor
            the reconstructed audio
        """
        tokens, embedding = self.encode(inputs, length)
        audio = self.decode(tokens, length)

        return tokens, embedding, audio

    def encode(self, inputs, length):
        """Encodes the input audio as tokens and embeddings

        Arguments
        ---------
        inputs : torch.Tensor
            A (Batch x Samples) or (Batch x Channel x Samples)
            tensor of audio
        length : torch.Tensor
            A tensor of relative lengths

        Returns
        -------
        tokens : torch.Tensor
            A (Batch x num_codebooks x Length) tensor of audio tokens
        emb : torch.Tensor
            Raw vector embeddings from the model's
            quantizers
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        max_len = inputs.size(-1)
        padding_mask = length_to_mask(
            length * max_len, max_len, device=inputs.device
        ).unsqueeze(1)

        tokens = self.model.encode(
            inputs, padding_mask, num_quantizers=self.num_codebooks
        )[0]

        # Reshape input_tensor for broadcasting
        input_tensor = tokens.unsqueeze(-1).expand(
            -1, -1, -1, self.embeddings.shape[-1]
        )  # [B, N, T, D]
        # Gather embeddings for each token
        embeddings = torch.gather(
            self.embeddings.unsqueeze(0).expand(tokens.shape[0], -1, -1, -1),
            2,
            input_tensor,
        )

        return tokens, embeddings

    def decode(self, tokens, length=None):
        """Decodes audio from tokens

        Arguments
        ---------
        tokens : torch.Tensor
            A (Batch x num_codebooks x Length) tensor of audio tokens
        length : torch.Tensor
            A 1-D tensor of relative lengths

        Returns
        -------
        audio : torch.Tensor
            the reconstructed audio
        """
        result = self.model.decode(tokens)
        audio = result.audio_values
        if length is not None:
            clean_padding_(audio, length)
        return audio
