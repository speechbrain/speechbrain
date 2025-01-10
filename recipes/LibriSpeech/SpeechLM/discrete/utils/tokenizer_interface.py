"""
Unified interface for tokenizers, standardizing the output shape of encode and decode functions.

This class reshapes the outputs of various tokenizers to ensure consistency, simplifying integration with recipes and workflows.

Authors
---------
* Pooneh Mousavi, 2024
"""

import torch
from abc import ABC, abstractmethod
from speechbrain.lobes.models.huggingface_transformers.encodec import Encodec
from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import (
    DiscreteSSL,
)
from speechbrain.lobes.models.discrete.dac import DAC
from speechbrain.lobes.models.discrete.speechtokenizer_interface import (
    SpeechTokenizer_interface,
)
from speechbrain.lobes.models.discrete.hubert import FairseqHuBERT


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers that encode signals into discrete tokens
    and decode tokens back into signals.

    This class defines the essential methods that any tokenizer must implement,
    including encoding, decoding, and retrieving pretrained embeddings.

    Naming Convenstion
    ------------------
    B : int
        Batch size.
    T : int
        Sequence length in the time domain.
    N : int
        Sequence length in the token domain.
    C : int
        Vocabulary size, assuming each codebook has the same number of tokens.
    K : int
        Number of codebooks.
    """

    def __init__(self):
        """
        Initialize the BaseTokenizer.

        This is a base constructor that other tokenizers can extend.
        """
        super().__init__()

    @abstractmethod
    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, num_codebooks=None, **kwargs):
        """
        Encode a signal into discrete tokens.

        Arguments
        ---------
        signal : torch.Tensor
            Input signal with shape [B, T].
        lengths : torch.Tensor
            Lengths of each sequence in the batch, with shape [B].
        num_codebooks : int, optional
            Number of codebooks to use for encoding. If None, all codebooks are used (default: None).
            If specified as an int, the tokens will be truncated to include only the first `num_codebooks` codebooks. If specified as a list,
            the tokens will include only the codebooks at the specified indices.
        **kwargs : dict
            Additional arguments for the tokenizer.

        Returns
        -------
        tokens : torch.Tensor
            Discretized tokens with shape [B, N, K].
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        """
        Decode discrete tokens back into a signal.

        Arguments
        ---------
        tokens : torch.Tensor
            Input tokens with shape [B, N, K].
        **kwargs : dict
            Additional arguments for the tokenizer.

        Returns
        -------
        signal : torch.Tensor
            Reconstructed signal with shape [B, T].
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def get_pretrained_embeddings(self, vocab_size, num_codebooks, **kwargs):
        """
        Retrieve pretrained embeddings for the tokenizer.

        Arguments
        ---------
        vocab_size : int
            Number of tokens in each codebook.
        num_codebooks : int
            Number of codebooks.
        **kwargs : dict
            Additional arguments for embedding retrieval.

        Returns
        -------
        embeddings : torch.Tensor
            Pretrained embedding weights with shape [K, C, H], where H is the embedding dimension.
        """
        pass


class EncodecTokenizer(Encodec, BaseTokenizer):
    def __init__(self, *args, **kwargs):
        Encodec.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, num_codebooks=None, **kwargs):
        self.eval()
        tokens, _ = self.encode(signal, lengths)
        if num_codebooks:
            if tokens.shape[-1] < num_codebooks:
                raise ValueError(
                    f"Model only outputs {tokens.shape[-1]} codebooks, but {num_codebooks} requested"
                )
            tokens = tokens[..., :num_codebooks]
        return tokens

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        signal = self.decode(tokens)[:, 0]
        return signal

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        embeddings = self.vocabulary
        return embeddings.reshape(-1, embeddings.shape[-1])


class DACTokenizer(DAC, BaseTokenizer):
    def __init__(self, *args, **kwargs):
        DAC.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, num_codebooks=None, **kwargs):
        self.eval()
        tokens, _ = self(signal[:, None], n_quantizers=num_codebooks)
        return tokens.movedim(-1, -2)

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        quantized_feats, _, _ = self.quantizer.from_codes(
            tokens.movedim(-1, -2)
        )
        return self.decode(quantized_feats)[:, 0]

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        toks = torch.arange(vocab_size).to(next(self.parameters()).device)
        toks = toks[:, None, None].expand(-1, num_codebooks, -1).clone()
        self.eval()
        z_q, z_p, _ = self.quantizer.from_codes(toks)
        z_ps = z_p.split(z_p.shape[1] // toks.shape[1], dim=1)
        z_qs = [
            self.quantizer.quantizers[i].out_proj(z_p_i)
            for i, z_p_i in enumerate(z_ps)
        ]
        return torch.cat(z_qs)[:, :, 0]


class SpeechTokenizer(SpeechTokenizer_interface, BaseTokenizer):
    def __init__(self, *args, **kwargs):
        SpeechTokenizer_interface.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)
        self.sample_rate = 16000

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, num_codebooks=None, **kwargs):
        self.eval()
        tokens = self(signal)
        if num_codebooks:
            if len(tokens) < num_codebooks:
                raise ValueError(
                    f"Model only outputs {len(tokens)} codebooks, but {num_codebooks} requested"
                )
            tokens = tokens[:num_codebooks]
        return tokens.movedim(-3, -1)

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        return self.decode(tokens.movedim(-1, -3))

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        toks = torch.arange(vocab_size).to(next(self.parameters()).device)
        toks = toks[None, :, None].expand(num_codebooks, -1, -1).clone()
        self.eval()
        embs = [
            self.model.quantizer.vq.layers[i].decode(indices)
            for i, indices in enumerate(toks)
        ]
        return torch.cat(embs)[:, :, 0]


class FairseqHuBERTTokenizer(FairseqHuBERT, BaseTokenizer):
    def __init__(self, *args, **kwargs):
        FairseqHuBERT.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)
    
    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, num_codebooks=None, **kwargs):
        self.eval()
        tokens = self.encode(signal)
        return tokens.unsqueeze(0).permute(0, 2, 1)

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        raise NotImplementedError("Fairseq HuBERT does not support decoding")

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        raise NotImplementedError("Fairseq HuBERT does not support embeddings")

class DiscreteSSLTokenizer(DiscreteSSL, BaseTokenizer):
    def __init__(self, *args, **kwargs):
        DiscreteSSL.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, num_codebooks=None, **kwargs):
        self.eval()
        tokens, _, _ = self.encode(
            signal, lengths, SSL_layers=num_codebooks, **kwargs
        )
        return tokens

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        return self.decode(tokens, **kwargs)

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        embs = []
        for layer_num, vocabulary in zip(
            self.ssl_layer_ids, self.vocabularies,
        ):
            if layer_num not in num_codebooks:
                continue
            embs.append(torch.as_tensor(vocabulary, dtype=torch.float32))
        embs = torch.cat(embs)
        return embs