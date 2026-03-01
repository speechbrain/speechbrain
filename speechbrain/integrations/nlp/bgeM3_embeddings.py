"""Wrappers for BGE-M3 sentence embeddings.

Reference: https://arxiv.org/abs/2402.03216

Authors
* Salima Mdhaffar 2025
* Maryem Bouziane 2025
"""

from typing import List

import torch

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError as e:
    raise ImportError(
        f"Failed to import FlagEmbedding: {e}\n"
        f"Please install FlagEmbedding e.g. using "
        f"`conda install -c conda-forge flagembedding`."
    ) from e


class BGEM3SentenceEmbeddings(torch.nn.Module):
    """
    Simple wrapper for BGE-M3 sentence embeddings.

    The wrapper exposes a callable interface that returns PyTorch tensors
    from ``BGEM3FlagModel.encode`` outputs.

    Arguments
    ---------
    source : str (default: 'BAAI/bge-m3')
        HuggingFace repo name or local path for the BGE-M3 model.
    use_fp16 : bool (default: False)
        If True, loads the internal model in fp16 when possible.
    return_dense : bool (default: True)
        If True, returns dense embeddings (``dense_vecs``).
    return_sparse : bool (default: False)
        If True, returns sparse embeddings (``sparse_vecs``).
    return_colbert_vecs : bool (default: False)
        If True, returns ColBERT-style token embeddings (``colbert_vecs``).
    max_length : int (default: 8192)
        Maximum sequence length (in tokens) used by the encoder.
    batch_size : int (default: 12)
        Internal batch size used by ``BGEM3FlagModel.encode``.
    **kwargs
        Extra keyword arguments passed to ``BGEM3FlagModel``.

    Example
    -------
    >>> embedder = BGEM3SentenceEmbeddings(source="BAAI/bge-m3")
    >>> sentences = ["hello world", "speechbrain integration"]
    >>> embeddings = embedder(sentences)
    """

    def __init__(
        self,
        source: str = "BAAI/bge-m3",
        use_fp16: bool = False,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
        max_length: int = 8192,
        batch_size: int = 12,
        **kwargs,
    ) -> None:
        super().__init__()

        self.return_dense = bool(return_dense)
        self.return_sparse = bool(return_sparse)
        self.return_colbert_vecs = bool(return_colbert_vecs)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)

        # Buffer used to track device / dtype when the module is moved
        self.register_buffer("_device_indicator", torch.empty(0))

        # Internal BGE-M3 model (FlagEmbedding)
        self.model = BGEM3FlagModel(
            source,
            use_fp16=use_fp16,
            **kwargs,
        )

        logger.info(
            "BGEM3SentenceEmbeddings initialized with source='%s', "
            "use_fp16=%s, return_dense=%s, return_sparse=%s, "
            "return_colbert_vecs=%s, max_length=%d, batch_size=%d",
            source,
            use_fp16,
            self.return_dense,
            self.return_sparse,
            self.return_colbert_vecs,
            self.max_length,
            self.batch_size,
        )

    def forward(self, inputs: List[str]):
        """Extract BGE-M3 embeddings for a batch of sentences.

        Arguments
        ---------
        inputs : list of str
            Sentences to embed.

        Returns
        -------
        torch.Tensor or dict
            If only ``return_dense=True`` is set, returns a tensor of
            dense embeddings of shape ``[batch, dim]``.
            Otherwise, returns a dict containing the requested fields
            (e.g. ``"dense_vecs"``, ``"sparse_vecs"``, ``"colbert_vecs"``).
        """
        if isinstance(inputs, str):
            raise ValueError("Expected a list of sentences, not a single str.")

        if not isinstance(inputs, list) or len(inputs) == 0:
            raise ValueError("Input must be a non-empty list of sentences.")

        device = self._device_indicator.device
        dtype = self._device_indicator.dtype or torch.float32

        raw = self.model.encode(
            inputs,
            return_dense=self.return_dense,
            return_sparse=self.return_sparse,
            return_colbert_vecs=self.return_colbert_vecs,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )

        # Dense only -> directly return a tensor
        if self.return_dense and not (
            self.return_sparse or self.return_colbert_vecs
        ):
            dense = torch.from_numpy(raw["dense_vecs"]).to(
                device=device, dtype=dtype
            )
            return dense

        # Multiple outputs -> return a dict
        outputs = {}

        if self.return_dense and "dense_vecs" in raw:
            outputs["dense_vecs"] = torch.from_numpy(raw["dense_vecs"]).to(
                device=device, dtype=dtype
            )

        if self.return_sparse and "sparse_vecs" in raw:
            outputs["sparse_vecs"] = raw["sparse_vecs"]

        if self.return_colbert_vecs and "colbert_vecs" in raw:
            outputs["colbert_vecs"] = torch.from_numpy(raw["colbert_vecs"]).to(
                device=device, dtype=dtype
            )

        return outputs

    def embed_sentence(self, sentence: str) -> torch.Tensor:
        """Embeds a single sentence and returns a dense vector.

        Arguments
        ---------
        sentence : str
            Sentence to embed.

        Returns
        -------
        torch.Tensor
            Dense embedding of shape ``[embedding_dim]``.
        """
        out = self([sentence])
        if isinstance(out, dict):
            return out["dense_vecs"][0]
        return out[0]
