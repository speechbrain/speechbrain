"""Provides a metrics class for the SemDist metric.

Authors
* Sylvain de Langen 2024
"""

from typing import Callable, List, Literal

import torch

from speechbrain.utils.metric_stats import MetricStats


class BaseSemDistStats(MetricStats):
    """
    Base class to implement the SemDist metric, for the variants that estimate a
    single cosine similarity per pair of target and predicted texts.
    The SemDist metrics are described by the paper
    `Evaluating User Perception of Speech Recognition System Quality with Semantic Distance Metric <https://arxiv.org/abs/2110.05376>`_.

    Arguments
    ---------
    embed_function : Callable[[List[str]], torch.Tensor]
        Given a list of sentences, return their summarized embedding using the
        method of your choice (e.g. mean pooling)
    scale : float, optional
        The `α` scale applied to the cosine similarity result for clarity. The
        default is `1000`, in order to match the authors' recommendation.
    batch_size : int, optional
        How many pairs of utterances should be considered at once. Higher is
        faster but may result in OOM.
    """

    def __init__(
        self,
        embed_function: Callable[[List[str]], torch.Tensor],
        scale: float = 1000.0,
        batch_size: int = 64,
    ):
        self.clear()
        self.embed_function = embed_function
        self.scale = scale
        self.batch_size = batch_size

    def clear(self):
        """Clears the collected metrics"""
        self.ids = []
        self.predictions = []
        self.targets = []
        self.scores = []
        self.summary = {}

    def append(self, ids, predict, target):
        """
        Appends inputs, predictions and targets to internal
        lists

        Arguments
        ---------
        ids: list
            the string IDs for the samples
        predict: list
            the model's predictions in tokenizable format
        target: list
            the ground truths in tokenizable format
        """
        self.ids.extend(ids)
        self.predictions.extend(predict)
        self.targets.extend(target)

    def summarize(self, field=None):
        """Summarize the SemDist metric scores. Performs the actual embedding
        function call and SemDist calculation.

        Full set of fields:
        - `semdist`: The average SemDist over all utterances, multiplied by
          the scale optionally specified at initialization.

        Additionally, a `scores` list is populated by this function for each
        pair of sentences. Each entry of that list is a dict, with the fields:
        - `key`: the ID of the utterance.
        - `semdist`: The SemDist of the utterance, multiplied by the scale.

        Arguments
        ---------
        field : str, optional
            The field to return, if you are only interested in one of them.
            If specified, a single `float` is returned, otherwise, a dict is.

        Returns
        -------
        dict from str to float, if `field is None`
            A dictionary of the fields documented above.
        float, if `field is not None`
            The single field selected by `field`.
        """

        with torch.no_grad():
            self._update_summary()

        if field is not None:
            return self.summary[field]

        return self.summary

    def _update_summary(self):
        """Performs the actual inference and SemDist estimation, updating the
        `summary` field. Automatically called by `summarize`."""

        semdist_sum = 0.0

        for chunk_idx in range(0, len(self.predictions), self.batch_size):
            ids = self.ids[chunk_idx : chunk_idx + self.batch_size]
            ref_text = self.targets[chunk_idx : chunk_idx + self.batch_size]
            hyp_text = self.predictions[chunk_idx : chunk_idx + self.batch_size]

            ref_emb = self.embed_function(ref_text).cpu()
            hyp_emb = self.embed_function(hyp_text).cpu()

            similarity = torch.nn.functional.cosine_similarity(
                ref_emb, hyp_emb, dim=-1
            )
            chunk_semdist = (1.0 - similarity) * self.scale

            for i, utt_id in enumerate(ids):
                self.scores.append(
                    {"key": utt_id, "semdist": chunk_semdist[i].item()}
                )

            semdist_sum += chunk_semdist.sum()

        semdist = (semdist_sum / len(self.predictions)).item()
        self.summary["semdist"] = semdist


class SemDistStats(BaseSemDistStats):
    """Computes the SemDist metric with a provided HuggingFace Transformers text
    encoder.

    Arguments
    ---------
    lm : speechbrain.lobes.models.huggingface_transformers.TextEncoder
        HF Transformers tokenizer and text encoder wrapper to use as a LM.
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

    def __init__(
        self,
        lm,
        method: Literal["meanpool", "cls"] = "meanpool",
        *args,
        **kwargs,
    ):
        super().__init__(embed_function=self._embed, *args, **kwargs)
        self.lm = lm
        self.method = method

    def _embed(self, sentences: List[str]) -> torch.Tensor:
        """Computes the LM embedding of a batch of independent sentences,
        according to the pooling method chosen at initialization.

        Arguments
        ---------
        sentences : list of str
            List of unprocessed sentences to tokenize and encode.

        Returns
        -------
        torch.Tensor
            Embedding of the LM encoder.
        """

        sentences = [" ".join(sent) for sent in sentences]

        tokens, hidden = self.lm(sentences, return_tokens=True)
        mask = tokens["attention_mask"].cpu()

        if self.method == "meanpool":
            masked_hidden = hidden.cpu() * mask.unsqueeze(-1)
            nonmasked_counts = torch.sum(mask, dim=-1)  # shape: [batch_size]
            return torch.sum(
                masked_hidden, dim=-2
            ) / nonmasked_counts.unsqueeze(-1)
        elif self.method == "cls":
            return hidden[:, 0, :].cpu()  # the first token
        else:
            raise ValueError(
                f"Specified SemDist method {self.method} is invalid"
            )
