"""Provides a metrics class for the BERTscore metric.

Authors
* Sylvain de Langen 2024
"""

import math
from collections import defaultdict
from typing import Iterable, Optional

import torch

from speechbrain.lobes.models.huggingface_transformers import TextEncoder
from speechbrain.utils.distances import cosine_similarity_matrix
from speechbrain.utils.logger import get_logger
from speechbrain.utils.metric_stats import MetricStats

logger = get_logger(__name__)


class BERTScoreStats(MetricStats):
    """Computes BERTScore with a provided HuggingFace Transformers text encoder,
    using the method described in the paper
    `BERTScore: Evaluating Text Generation with BERT <https://arxiv.org/abs/1904.09675>`_.

    BERTScore operates over contextualized tokens (e.g. the output of BERT, but
    many other models would work). Since cosine similarities are used, the
    output range would be between `-1` and `1`.
    See the linked resources for more details.

    Special tokens (as queried from the tokenizer) are entirely ignored.

    Authors' reference implementation of the metric can be found
    `here <https://github.com/Tiiiger/bert_score>`_. The linked page extensively
    describes the approach and compares how the BERTScore relates to human
    evaluation with many different models.

    .. warning::
        Out of the box, this implementation may not strictly match the results
        of the reference implementation. Please read the argument documentation
        to understand the differences.

    Arguments
    ---------
    lm : speechbrain.lobes.models.huggingface_transformers.TextEncoder
        HF Transformers tokenizer and text encoder wrapper to use as a LM.
    batch_size : int, optional
        How many pairs of utterances should be considered at once. Higher is
        faster but may result in OOM.
    use_idf : bool, optional
        If enabled (default), tokens in the reference are weighted by
        Inverse Document Frequency, which allows to weight down the impact of
        common words that may carry less information. Every sentence appended
        is considered a document in the IDF calculation.
    sentence_level_averaging : bool, optional
        When `True`, the final recall/precision metrics will be the average of
        recall/precision for each tested sentence, rather of each tested token,
        e.g. a very long sentence will weigh as much as a very short sentence in
        the final metrics. The default is `True`, which matches the reference
        implementation.
    allow_matching_special_tokens : bool, optional
        When `True`, non-special tokens may match against special tokens during
        greedy matching (e.g. `[CLS]`/`[SEP]`). Batch size must be 1 due to
        padding handling.
        The default is `False`, which is different behavior from the reference
        implementation (see
        `bert_score#180 <https://github.com/Tiiiger/bert_score/issues/180>`_).
    """

    def __init__(
        self,
        lm: TextEncoder,
        batch_size: int = 64,
        use_idf: bool = True,
        sentence_level_averaging: bool = True,
        allow_matching_special_tokens: bool = False,
    ):
        self.clear()
        self.lm = lm
        self.batch_size = batch_size
        self.use_idf = use_idf
        self.sentence_level_averaging = sentence_level_averaging
        self.allow_matching_special_tokens = allow_matching_special_tokens

    def clear(self):
        """Clears the collected statistics"""
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
        """Summarize the classification metric scores. Performs the actual LM
        inference and BERTScore estimation.

        Full set of fields:
         - `bertscore-recall`, optionally weighted by idf of ref tokens
         - `bertscore-precision`, optionally weighted by idf of hyp tokens
         - `bertscore-f1`

        Arguments
        ---------
        field : str
            If provided, only returns selected statistic. If not,
            returns all computed statistics.

        Returns
        -------
        float or dict
            Returns a float if ``field`` is provided, otherwise
            returns a dictionary containing all computed stats.
        """

        with torch.no_grad():
            self._update_summary()

        if field is not None:
            return self.summary[field]

        return self.summary

    def _update_summary(self):
        """Performs the actual LM inference and BERTscore estimation, updating
        the `summary` field. Automatically called by `summarize`."""

        if self.allow_matching_special_tokens:
            assert self.batch_size == 1, (
                "Batch size must be 1 when passing "
                "`allow_matching_special_tokens` due to padding handling."
            )

        token_masks = get_bert_token_mask(self.lm.tokenizer)
        token_weights = self._make_weights(self.targets)

        recall_sum = recall_weight = 0.0
        precision_sum = precision_weight = 0.0

        for chunk_idx in range(0, len(self.predictions), self.batch_size):
            ids = self.ids[chunk_idx : chunk_idx + self.batch_size]
            ref_text = self.targets[chunk_idx : chunk_idx + self.batch_size]
            hyp_text = self.predictions[chunk_idx : chunk_idx + self.batch_size]

            ref_text = [" ".join(ref) for ref in ref_text]
            hyp_text = [" ".join(hyp) for hyp in hyp_text]

            ref_toks, ref_hidden = self.lm(ref_text, return_tokens=True)
            hyp_toks, hyp_hidden = self.lm(hyp_text, return_tokens=True)

            ref_hidden = ref_hidden.cpu()
            hyp_hidden = hyp_hidden.cpu()
            ref_toks = ref_toks["input_ids"].cpu()
            hyp_toks = hyp_toks["input_ids"].cpu()

            # shape [batch, ref dim, hyp dim]
            similarity_matrix = cosine_similarity_matrix(ref_hidden, hyp_hidden)

            ref_mask = self._select_by_tokens(token_masks, ref_toks)
            hyp_mask = self._select_by_tokens(token_masks, hyp_toks)

            # mask rows according to ref_mask and columns according to hyp_mask
            if not self.allow_matching_special_tokens:
                similarity_matrix[~ref_mask, :] = 0.0
                similarity_matrix.transpose(1, 2)[~hyp_mask, :] = 0.0

            # for recall, greedily select the "closest" hyp token for every ref
            # token, thus of shape [batch, ref dim]
            recall_values, _ = similarity_matrix.max(dim=-1)
            # for precision, same thing but with the closest ref for every hyp
            precision_values, _ = similarity_matrix.max(dim=-2)

            # for each token, load the matching token weight
            # the result is a weight tensor with the same shape as the inputs
            recall_weights = self._select_by_tokens(
                token_weights, ref_toks.cpu()
            )
            precision_weights = self._select_by_tokens(
                token_weights, hyp_toks.cpu()
            )

            # mask off weights
            recall_weights[~ref_mask] = 0.0
            precision_weights[~hyp_mask] = 0.0

            batch_recall = recall_values * recall_weights
            batch_precision = precision_values * precision_weights

            for i, utt_id in enumerate(ids):
                # TODO: optionally provide a token->token map
                self.scores.append(
                    {
                        "key": utt_id,
                        "recall": (
                            batch_recall[i].sum() / recall_weights[i].sum()
                        ).item(),
                        "precision": (
                            batch_precision[i].sum()
                            / precision_weights[i].sum()
                        ).item(),
                    }
                )

            if self.sentence_level_averaging:
                recall_sum += batch_recall.sum() / recall_weights.sum()
                recall_weight += 1.0

                precision_sum += batch_precision.sum() / precision_weights.sum()
                precision_weight += 1.0
            else:
                recall_sum += batch_recall.sum()
                recall_weight += recall_weights.sum()

                precision_sum += batch_precision.sum()
                precision_weight += precision_weights.sum()

        recall = recall_sum / recall_weight
        precision = precision_sum / precision_weight
        f1 = 2.0 * (recall * precision) / (recall + precision)

        self.summary.update(
            {
                "bertscore-recall": recall,
                "bertscore-precision": precision,
                "bertscore-f1": f1,
            }
        )

    def _make_weights(self, corpus):
        """Makes a token weight tensor, optionally including IDF. If not using
        IDF, currently simply returns a tensor full of ones."""
        if self.use_idf:
            if len(self.predictions) == 1:
                raise ValueError(
                    "Token IDF weighting was enabled, but 1 text is not "
                    "enough. Compute the summary over more texts or disable "
                    "IDF weighting."
                )

            return get_bertscore_token_weights(self.lm.tokenizer, corpus)

        return get_bertscore_token_weights(self.lm.tokenizer)

    def _select_by_tokens(self, token_weight, input_tokens):
        """From a batch of tokenized texts `input_tokens`, returns an
        identically shaped tensor where each item `token_id` becomes
        `token_weight[token_id]`."""
        return token_weight.index_select(
            dim=0, index=input_tokens.flatten()
        ).reshape(input_tokens.shape)


def get_bert_token_mask(tokenizer) -> torch.BoolTensor:
    """Returns a token mask with special tokens masked.

    Arguments
    ---------
    tokenizer : transformers.PreTrainedTokenizer
        HuggingFace tokenizer for the BERT model.

    Returns
    -------
    torch.BoolTensor
        A mask tensor that can be indexed by token ID (of shape `[vocab_size]`).
    """

    vocab = tokenizer.get_vocab()
    max_idx = max(vocab.values())

    weights = torch.ones((max_idx + 1,), dtype=torch.bool)

    special_tokens = []

    for tok_entry in tokenizer.special_tokens_map.values():
        if isinstance(tok_entry, str):
            special_tokens.append(vocab[tok_entry])
        else:
            for tok in tok_entry:
                special_tokens.append(vocab[tok])

    weights[special_tokens] = False

    return weights


def get_bertscore_token_weights(
    tokenizer, corpus: Optional[Iterable[str]] = None
) -> torch.Tensor:
    """Returns token weights for use with the BERTScore metric.
    When specifying `corpus`, the weights are the Inverse Document Frequency
    (IDF) of each token, extracted from the `corpus`.

    The IDF formula is adapted from the BERTScore paper, where words missing
    from the reference corpus are weighted with `+1` smoothing.

    Arguments
    ---------
    tokenizer : transformers.PreTrainedTokenizer
        HuggingFace tokenizer for the BERT model.
    corpus : Iterable[str], optional
        Iterable corpus to compute the IDF from. Each iterated value is
        considered a document in the corpus in the IDF calculation.
        If omitted, no IDF weighting is done.

    Returns
    -------
    torch.Tensor
        A floating-point tensor that can be indexed by token ID, of shape
        `[vocab_size]`, where each entry is by how much the impact of a given
        token should be multiplied.
    """

    max_idx = max(tokenizer.get_vocab().values())

    if corpus is None:
        return torch.ones((max_idx,))

    freq_dict = defaultdict(lambda: 0)

    for document_idx, document in enumerate(corpus):
        tokens = tokenizer(" ".join(document))["input_ids"]
        unique_words = set(tokens)

        for unique_word in unique_words:
            freq_dict[unique_word] += 1

    document_count = document_idx + 1

    weights = [
        math.log((document_count + 1) / (freq_dict[token_id] + 1))
        for token_id in range(max_idx + 1)
    ]

    return torch.tensor(weights)
