"""This lobe enables the integration of huggingface pretrained EnCodec.

EnCodec makes it possible to compress audio into a sequence of discrete tokens
at different bandwidths - and to reconstruct audio from such sequences, with
some loss of quality depending on the bandwidth.

Note that while encodec can be used to reconstruct speech data, for a
high-quality reconstruction, it is recommended to use a specially trained
vocoder, such as Vocos (speechbrain.lobes.models.huggingface_transformers.vocos)

Repository: https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/encodec
Paper: https://arxiv.org/abs/2210.13438

Authors
 * Artem Ploujnikov 2023
"""

import torch
import logging
from torch.nn import functional as F
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)

DEFAULT_SAMPLE_RATE = 24000

logger = logging.getLogger(__name__)


class Encodec(HFTransformersInterface):
    """An wrapper for the HuggingFace encodec model

    Arguments
    ---------
    source : str
        a HuggingFace repository identifier or a path
    save_path : str
        the location where the pretrained model will be saved
    sample_rate : int
        the audio sampling rate
    bandwidth : float
        the encoding bandwidth, in kbps (optional)
        Supported bandwidths:
        1.5, 3.0, 6.0, 12.0, 24.0
    freeze : bool
        whether the model will be frozen (e.g. not trainable if used
        as part of training another model)

    Example
    -------
    >>> model_hub = "facebook/encodec_24khz"
    >>> save_path = "savedir"
    >>> model = Encodec(model_hub, save_path)
    >>> audio = torch.randn(4, 1000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens = model.encode(audio, length)
    >>> tokens.shape
    torch.Size([4, 4, 2])
    >>> rec = model.decode(tokens, length)
    >>>
    """

    def __init__(
        self,
        source,
        save_path=None,
        sample_rate=None,
        freeze=True,
        bandwidth=1.5,
    ):
        super().__init__(source=source, save_path=save_path, freeze=freeze)
        if not sample_rate:
            sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.num_heads = self.model.quantizer.get_num_quantizers_for_bandwidth(
            bandwidth
        )
        quantizer_layers = self.model.quantizer.layers[: self.num_heads]
        self.vocabulary = torch.stack(
            [layer.codebook.embed for layer in quantizer_layers]
        )
        _, self.num_tokens, self.emb_dim = self.vocabulary.shape
        self.vocabulary_flat = self.vocabulary.reshape(
            self.num_heads * self.num_tokens, self.emb_dim
        )
        self.token_index_offsets = (
            torch.arange(self.num_heads)[None, None, :] * self.num_tokens
        )
        if self.freeze:
            logger.warning("huggingface_Encodec - Encodec is frozen.")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, inputs, length):
        """Encodes the input audio as tokens

        Arguments
        ---------
        inputs : torch.Tensor
            a (Batch x Samples) or (Batch x Channel x Samples)
            tensor of audio
        length : torch.Tensor
            a tensor of relative lengths

        Returns
        -------
        tokens : torch.Tensor
            a (Batch X Tokens) tensor of audio tokens
        """
        return self.encode(inputs, length)

    def encode(self, inputs, length):
        """Encodes the input audio as tokens

        Arguments
        ---------
        inputs : torch.Tensor
            a (Batch x Samples) or (Batch x Channel x Samples)
            tensor of audio
        length : torch.Tensor
            a tensor of relative lengths

        Returns
        -------
        tokens : torch.Tensor
            a (Batch x Tokens x Heads) tensor of audio tokens
        emb : torch.Tensor
            raw vector embeddings from the model's
            quantizers
        """
        with torch.set_grad_enabled(not self.freeze):
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)
            max_len = inputs.size(-1)
            mask = length_to_mask(
                length * max_len, max_len, device=inputs.device
            ).unsqueeze(1)
            result = self.model.encode(inputs, mask, bandwidth=self.bandwidth)
            tokens = result.audio_codes.squeeze(0).transpose(-1, -2)
            emb = self.embeddings(tokens)
            return tokens, emb

    def embeddings(self, tokens):
        """Converts token indexes to vector embeddings

        Arguments
        ---------
        tokens : torch.Tensor
            a (Batch x Length x Heads) tensor of token indexes

        Returns
        -------
        emb : torch.Tensor
            a (Batch x Length x Heads x Embedding) tensor
            of raw vector embeddings from the model's
            quantizer codebooks
        """
        idx = tokens + self.token_index_offsets
        emb = F.embedding(idx, self.vocabulary_flat)
        return emb

    def decode(self, tokens, length=None):
        """Decodes audio from tokens

        Arguments
        ---------
        tokens : torch.Tensor
            a (Batch x Length x Heads) tensor of audio tokens
        length : torch.Tensor
            a 1-D tensor of relative lengths

        Returns
        -------
        audio : torch.Tensor
            the reconstructed audio
        """
        with torch.set_grad_enabled(not self.freeze):
            result = self.model.decode(
                tokens.unsqueeze(0).transpose(-1, -2), [None]
            )
            audio = result.audio_values
            if length is not None:
                max_len = audio.size(-1)
                mask = length_to_mask(
                    length * max_len, max_len, device=tokens.device
                ).unsqueeze(1)
                audio = audio * mask
            return audio

    def decode_emb(self, emb, length):
        """Decodes raw vector embeddings into audio

        Arguments
        ---------
        emb : torch.Tensor
            A (Batch x Length x Heads x Embedding) tensor of
            raw vector embeddings

        Returns
        -------
        audio : torch.Tensor
            the reconstructed audio
        """
        with torch.set_grad_enabled(not self.freeze):
            scaled_states = emb.pow(2).sum(-1, keepdim=True)
            vocab = self.vocabulary.transpose(-1, -2).unsqueeze(0)
            emb_perm = emb.permute(0, 2, 1, 3)
            emb_vocab_prod = (emb_perm @ vocab).moveaxis(1, 2)
            vocab_sum = vocab.pow(2).sum(-2, keepdim=True).moveaxis(1, 2)
            dist = -(scaled_states - 2 * emb_vocab_prod + vocab_sum)
            tokens = dist.max(dim=-1).indices
            return self.decode(tokens, length)
