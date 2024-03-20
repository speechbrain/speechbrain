"""Provides a metrics class for the SemDist metric.

Authors
* Sylvain de Langen 2024
"""

from speechbrain.utils.metric_stats import MetricStats
from typing import Literal, Callable, Union, List
import torch


class BaseSemDistStats(MetricStats):
    """Base class to implement the SemDist metric, for the variants that
    estimate a single cosine similarity per pair of target and predicted texts.
    The SemDist metrics are described by the paper
    `Evaluating User Perception of Speech Recognition System Quality with
Semantic Distance Metric <https://arxiv.org/abs/2110.05376>`_.

    Arguments
    ---------
    model : transformers.AutoModelForTextEncoding
        Transformers text encoder. May live on a non-CPU device.
    tokenizer : transformers.AutoTokenizer
        Transformers tokenizer.
    embed_function : Callable[[List[str]], torch.Tensor]
        Given a list of sentences, return their summarized embedding using the
        method of your choice (e.g. mean pooling)
    scale : float, optional
        The `α` scale applied to the cosine similarity result for clarity. The
        default is `1000`, in order to match the authors' recommendation.
    """

    def __init__(self, embed_function : Callable[[List[str]], torch.Tensor], scale: float = 1000.0, batch_size: int = 64):
        self.clear()
        self.embed_function = embed_function
        self.scale = scale
        self.batch_size = batch_size

    def clear(self):
        """Clears the collected metrics"""
        self.ids = []
        self.predictions = []
        self.targets = []
        self.summary = {}

    def append(self, ids, predictions, targets):
        """
        Appends inputs, predictions and targets to internal
        lists

        Arguments
        ---------
        ids: list
            the string IDs for the samples
        predictions: list
            the model's predictions in tokenizable format
        targets: list
            the ground truths in tokenizable format
        """
        self.ids.extend(ids)
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def summarize(self, field=None):
        """Summarize the SemDist metric scores. Performs the actual embedding
        function call and SemDist calculation."""

        with torch.no_grad():
            self._update_summary()

        if field is not None:
            return self.summary[field]

        return self.summary

    def _update_summary(self):
        semdist_sum = 0.0

        for chunk_idx in range(0, len(self.predictions), self.batch_size):
            ref_text = self.targets[chunk_idx : chunk_idx + self.batch_size]
            hyp_text = self.predictions[chunk_idx : chunk_idx + self.batch_size]

            ref_emb = self.embed_function(ref_text)
            hyp_emb = self.embed_function(hyp_text)

            print(ref_emb.shape, hyp_emb.shape)

            similarity = torch.nn.functional.cosine_similarity(ref_emb, hyp_emb, dim=-1)
            print(similarity)
            semdist_sum += (1.0 - similarity).sum() * self.scale

        semdist = semdist_sum / len(self.predictions)
        self.summary["semdist"] = semdist


class SemDistStats(BaseSemDistStats):
    """Computes the SemDist metric with a provided HuggingFace Transformers
    tokenizer and LM.
    
    Arguments
    ---------
    model : transformers.AutoModelForTextEncoding
        Transformers text encoder. May live on a non-CPU device.
    tokenizer : transformers.AutoTokenizer
        Transformers tokenizer.
    method : "meanpool" or "cls"
        - `"meanpool"` (default): Computes the mean of all contextualized
          embeddings, excluding padding tokens.
        - `"cls"`: Exclusively uses the first contextualized embedding, which
          with BERT-like tokenizers is the `[CLS]` token, which is typically
          intended to capture classification information.
    *args
        Extra positional arguments passed to the base constructor.
    **kwargs
        Extra keyword arguments passed to the base constructor."""

    def __init__(self, lm, tokenizer, method : Literal["meanpool", "cls"] = "meanpool", *args, **kwargs):
        super().__init__(embed_function=self._embed, *args, **kwargs)
        self.lm = lm
        self.tokenizer = tokenizer
        self.method = method

    def _embed(self, sentences: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            sentences, return_tensors="pt", padding=True
        ).to(self.lm.device)

        mask = tokens["attention_mask"]

        hidden = self.lm(**tokens).last_hidden_state.cpu()

        if self.method == "meanpool":
            masked_hidden = (hidden * mask.unsqueeze(-1))
            nonmasked_counts = torch.sum(mask, dim=-1)  # shape: [batch_size]
            return torch.sum(masked_hidden, dim=-2) / nonmasked_counts.unsqueeze(-1)
        elif self.method == "cls":
            return hidden[:, 0, :]  # the first token
