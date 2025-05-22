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
from torch.nn import functional as F

from speechbrain.dataio.dataio import clean_padding_, length_to_mask
from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.logger import get_logger

DEFAULT_SAMPLE_RATE = 24000

logger = get_logger(__name__)


class Encodec(HFTransformersInterface):
    """An wrapper for the HuggingFace encodec model

    Arguments
    ---------
    source : str
        A HuggingFace repository identifier or a path
    save_path : str
        The location where the pretrained model will be saved
    sample_rate : int
        The audio sampling rate
    bandwidth : float
        The encoding bandwidth, in kbps (optional)
        Supported bandwidths:
        1.5, 3.0, 6.0, 12.0, 24.0
    flat_embeddings : bool
        If set to True, embeddings will be flattened into
        (Batch x Length x (Heads * Embedding))
    freeze : bool
        whether the model will be frozen (e.g. not trainable if used
        as part of training another model)
    renorm_embeddings : bool
        whether embeddings should be renormalized. In the original
        model.

    Example
    -------
    >>> model_hub = "facebook/encodec_24khz"
    >>> save_path = "savedir"
    >>> model = Encodec(model_hub, save_path)
    >>> audio = torch.randn(4, 1000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens, emb = model.encode(audio, length)
    >>> tokens.shape
    torch.Size([4, 4, 2])
    >>> emb.shape
    torch.Size([4, 4, 2, 128])
    >>> rec = model.decode(tokens, length)
    >>> rec.shape
    torch.Size([4, 1, 1280])
    >>> rec_emb = model.decode_emb(emb, length)
    >>> rec_emb.shape
    torch.Size([4, 1, 1280])
    >>> rec_tokens = model.tokens(emb, length)
    >>> rec_tokens.shape
    torch.Size([4, 4, 2])
    >>> model = Encodec(model_hub, save_path, flat_embeddings=True)
    >>> _, emb = model.encode(audio, length)
    >>> emb.shape
    torch.Size([4, 4, 256])
    """

    def __init__(
        self,
        source,
        save_path=None,
        sample_rate=None,
        bandwidth=1.5,
        flat_embeddings=False,
        freeze=True,
        renorm_embeddings=True,
    ):
        super().__init__(source=source, save_path=save_path, freeze=freeze)
        if not sample_rate:
            sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.flat_embeddings = flat_embeddings
        self.num_heads = self.model.quantizer.get_num_quantizers_for_bandwidth(
            bandwidth
        )
        self.num_tokens = self.model.config.codebook_size
        quantizer_layers = self.model.quantizer.layers[: self.num_heads]
        vocabulary = torch.stack(
            [layer.codebook.embed for layer in quantizer_layers]
        )
        self.register_buffer("vocabulary", vocabulary)
        _, self.num_tokens, self.emb_dim = self.vocabulary.shape
        vocabulary_flat = self.vocabulary.reshape(
            self.num_heads * self.num_tokens, self.emb_dim
        )
        self.register_buffer("vocabulary_flat", vocabulary_flat)
        token_index_offsets = (
            torch.arange(self.num_heads)[None, None, :] * self.num_tokens
        )
        self.register_buffer("token_index_offsets", token_index_offsets)
        self.renorm_embeddings = renorm_embeddings
        if self.renorm_embeddings:
            emb_mean, emb_std = self._precalibrate()
            self.register_buffer("emb_mean", emb_mean)
            self.register_buffer("emb_std", emb_std)
        if self.freeze:
            logger.warning("huggingface_Encodec - Encodec is frozen.")
            for param in self.model.parameters():
                param.requires_grad = False

    def _precalibrate(self):
        """Compute parameters required to renormalize embeddings"""
        sample = torch.arange(self.num_tokens)[None, :, None].expand(
            1, self.num_tokens, self.num_heads
        )
        return self._compute_embedding_norm(sample)

    def _compute_embedding_norm(self, sample, length=None):
        """Computes the normalization for embeddings based on
        a sample.

        Arguments
        ---------
        sample : torch.Tensor
            A (Batch x Samples) or (Batch x Channel x Samples)
            audio sample
        length : torch.Tensor
            A tensor of relative lengths

        Returns
        -------
        emb_mean : torch.Tensor
        emb_std : torch.Tensor
            Norm stats for embeddings.
        """
        if length is None:
            length = torch.ones(len(sample), device=sample.device)
        max_len = sample.size(1)
        emb = self._raw_embeddings(sample)
        mask = length_to_mask(length * max_len, max_len)[
            :, :, None, None
        ].expand_as(emb)
        emb_mean = (emb.mean(-1).sum(1) / mask.mean(-1).sum(1)).mean(0)[
            None, None, :, None
        ]
        emb_diff_sq = ((emb - emb_mean) * mask) ** 2
        emb_std = (
            emb_diff_sq.sum(dim=[0, 1, 3])
            / (mask.expand_as(emb_diff_sq).sum(dim=[0, 1, 3]) - 1)
        ).sqrt()[None, None, :, None]
        return emb_mean, emb_std

    def calibrate(self, sample, length):
        """Calibrates the normalization on a sound sample

        Arguments
        ---------
        sample : torch.Tensor
            A (Batch x Samples) or (Batch x Channel x Samples)
            audio sample

        length : torch.Tensor
            A tensor of relative lengths

        Returns
        -------
        emb_mean : torch.Tensor
            The embedding mean

        emb_std : torch.Tensor
            The embedding standard deviation
        """
        if not self.renorm_embeddings:
            raise ValueError("Not supported when renorm_embeddings is disabled")
        sample_tokens = self._encode_tokens(sample, length)
        self.emb_mean, self.emb_std = self._compute_embedding_norm(
            sample_tokens, length
        )
        return self.emb_mean.squeeze(), self.emb_std.squeeze()

    def forward(self, inputs, length):
        """Encodes the input audio as tokens

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
            A (Batch X Tokens) tensor of audio tokens
        """
        return self.encode(inputs, length)

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
            A (Batch x Tokens x Heads) tensor of audio tokens
        emb : torch.Tensor
            Raw vector embeddings from the model's
            quantizers
        """
        with torch.set_grad_enabled(not self.freeze):
            tokens = self._encode_tokens(inputs, length)
            emb = self.embeddings(tokens)
            return tokens, emb

    def _encode_tokens(self, inputs, length):
        """Encodes audio as tokens only

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
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        max_len = inputs.size(-1)
        mask = length_to_mask(
            length * max_len, max_len, device=inputs.device
        ).unsqueeze(1)
        result = self.model.encode(inputs, mask, bandwidth=self.bandwidth)
        tokens = result.audio_codes.squeeze(0).transpose(-1, -2)
        return tokens

    def _raw_embeddings(self, tokens):
        """Converts token indexes to vector embeddings, for
        each quantizer

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
        emb = self._raw_embeddings(tokens)
        if self.renorm_embeddings:
            emb = (emb - self.emb_mean) / self.emb_std
        if self.flat_embeddings:
            batch_size, max_len, num_heads, emb_dim = emb.shape
            emb = emb.reshape(batch_size, max_len, num_heads * emb_dim)
        return emb

    def decode(self, tokens, length=None):
        """Decodes audio from tokens

        Arguments
        ---------
        tokens : torch.Tensor
            A (Batch x Length x Heads) tensor of audio tokens
        length : torch.Tensor
            A 1-D tensor of relative lengths

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
                clean_padding_(audio, length)
            return audio

    def tokens(self, emb, length=None):
        """Comberts embeddings to raw tokens

        Arguments
        ---------
        emb : torch.Tensor
            Raw embeddings
        length : torch.Tensor
            A 1-D tensor of relative lengths. If supplied,
            padded positions will be zeroed out

        Returns
        -------
        tokens : torch.Tensor
            A (Batch x Length) tensor of token indices"""
        with torch.set_grad_enabled(not self.freeze):
            if self.flat_embeddings:
                batch_size, max_len, _ = emb.shape
                emb = emb.reshape(
                    batch_size, max_len, self.num_heads, self.emb_dim
                )
            if self.renorm_embeddings:
                emb = emb * self.emb_std + self.emb_mean
            scaled_states = emb.pow(2).sum(-1, keepdim=True)
            vocab = self.vocabulary.transpose(-1, -2).unsqueeze(0)
            emb_perm = emb.permute(0, 2, 1, 3)
            emb_vocab_prod = (emb_perm @ vocab).moveaxis(1, 2)
            vocab_sum = vocab.pow(2).sum(-2, keepdim=True).moveaxis(1, 2)
            dist = -(scaled_states - 2 * emb_vocab_prod + vocab_sum)
            tokens = dist.max(dim=-1).indices
            if length is not None:
                clean_padding_(tokens, length)
            return tokens

    def decode_emb(self, emb, length):
        """Decodes raw vector embeddings into audio

        Arguments
        ---------
        emb : torch.Tensor
            A (Batch x Length x Heads x Embedding) tensor of
            raw vector embeddings
        length : torch.Tensor
            The corresponding lengths of the inputs.

        Returns
        -------
        audio : torch.Tensor
            the reconstructed audio
        """
        with torch.set_grad_enabled(not self.freeze):
            tokens = self.tokens(emb)
            return self.decode(tokens, length)
