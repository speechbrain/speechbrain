"""Provides a function to compute the BERTscore metric.

Authors
* Sylvain de Langen 2024
"""

from collections import defaultdict
from typing import Iterable, Optional
import torch
import logging
import math

from speechbrain.utils.distances import cosine_similarity_matrix
from speechbrain.utils.metric_stats import MetricStats

logger = logging.getLogger(__name__)


def get_bert_token_mask(tokenizer) -> torch.BoolTensor:
    """Returns a token mask with special tokens masked.

    Arguments
    ---------
    tokenizer
        HuggingFace tokenizer for the BERT model.

    Returns
    -------
    torch.BoolTensor
        A mask tensor that can be indexed by token ID (of shape `[vocab_size]`).
    """

    vocab = tokenizer.get_vocab()
    max_idx = max(vocab.values())

    weights = torch.ones((max_idx + 1,), dtype=torch.bool)

    special_tokens = [
        vocab[token] for token in tokenizer.special_tokens_map.values()
    ]

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
    tokenizer
        HuggingFace tokenizer for the BERT model.
    corpus : Iterable[str], optional
        Iterable corpus to compute the IDF from. Each iterated value is
        considered a document in the corpus in the IDF calculation.
        If omitted, no IDF weighting is done.
    """

    max_idx = max(tokenizer.get_vocab().values())
    weights = torch.ones((max_idx + 1,))

    if corpus is not None:
        freq_dict = defaultdict(lambda: 0)

        for document_idx, document in enumerate(corpus):
            tokens = tokenizer(document)["input_ids"]
            unique_words = set(tokens)

            for unique_word in unique_words:
                freq_dict[unique_word] += 1

        document_count = document_idx + 1

        for token_id in range(weights.size(0)):
            weights[token_id] *= math.log(
                document_count / freq_dict.get(token_id, 1)
            )

    return weights


class BERTScoreStats(MetricStats):
    """Computes BERTScore with a provided HuggingFace Transformers tokenizer and
    LM, using the method described in the paper
    `BERTScore: Evaluating Text Generation with BERT <https://arxiv.org/abs/1904.09675>`_.

    BERTScore operates over contextualized tokens (e.g. the output of BERT, but
    many other models would work). See the linked resources for more details.

    Special tokens (as queried from the tokenizer) are entirely ignored.

    Authors' own implementation of the metric can be found
    `here <https://github.com/Tiiiger/bert_score>`_. The linked page extensively
    describes the approach and compares how the BERTScore relates to human
    evaluation with many different models.

    Arguments
    ---------
    model : transformers.AutoModelForTextEncoding
        Transformers text encoder. May live on a non-CPU device.
    tokenizer : transformers.AutoTokenizer
        Transformers tokenizer.
    batch_size : int
        How many pairs of utterances should be considered at once. Higher is
        faster but may result in OOM.
    uses_idf : bool
        If enabled (default), tokens in the reference are weighted by
        Inverse Document Frequency, which allows to weight down the impact of
        common words that may carry less information.
    """

    def __init__(
        self,
        lm,
        tokenizer,
        batch_size: int,
        uses_idf: bool = True,
        mask_special_tokens: bool = True,
    ):
        self.clear()
        self.lm = lm
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.uses_idf = uses_idf
        self.mask_special_tokens = mask_special_tokens

    def clear(self):
        """Clears the collected statistics"""
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

        token_masks = get_bert_token_mask(self.tokenizer)
        ref_token_weights = self._make_weights(self.targets)
        hyp_token_weights = self._make_weights(self.predictions)

        recall_sum = 0.0
        recall_weight = 0.0

        precision_sum = 0.0
        precision_weight = 0.0

        for chunk_idx in range(0, len(self.predictions), self.batch_size):
            ref_text = self.targets[chunk_idx : chunk_idx + self.batch_size]
            hyp_text = self.predictions[chunk_idx : chunk_idx + self.batch_size]

            refs = self.tokenizer(ref_text, return_tensors="pt", padding=True)
            hyps = self.tokenizer(hyp_text, return_tensors="pt", padding=True)

            ref_tokens = refs["input_ids"]
            hyp_tokens = hyps["input_ids"]

            ref_hidden = self.lm(**refs).last_hidden_state.cpu()
            hyp_hidden = self.lm(**hyps).last_hidden_state.cpu()

            # shape [batch, ref dim, hyp dim]
            similarity_matrix = cosine_similarity_matrix(ref_hidden, hyp_hidden)

            ref_mask = self._select_by_tokens(token_masks, ref_tokens)
            hyp_mask = self._select_by_tokens(token_masks, hyp_tokens)

            # mask rows according to ref_mask and columns according to hyp_mask
            # reminder: this is the mask used to mask off special tokens
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
                ref_token_weights, ref_tokens
            )
            precision_weights = self._select_by_tokens(
                hyp_token_weights, hyp_tokens
            )

            # mask off weights
            recall_weights[~ref_mask] = 0.0
            precision_weights[~hyp_mask] = 0.0

            print()
            print(similarity_matrix)
            print(recall_values)
            print(recall_weights)
            print(ref_text)
            print(hyp_text)
            print((recall_values * recall_weights).sum() / recall_weights.sum())
            print((precision_values * precision_weights).sum() / precision_weights.sum())

            recall_sum += (recall_values * recall_weights).sum()
            recall_weight += recall_weights.sum()

            precision_sum += (precision_values * precision_weights).sum()
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

        if field is not None:
            return self.summary[field]

        return self.summary

    def _make_weights(self, corpus):
        """Makes a token weight tensor, optionally including IDF. If not using
        IDF, currently simply returns a tensor full of ones."""
        if self.uses_idf:
            if len(self.predictions) == 1:
                raise ValueError(
                    "Token IDF weighting was enabled, but 1 text is not "
                    "enough. Compute the summary over more texts or disable "
                    "IDF weighting."
                )

            return get_bertscore_token_weights(self.tokenizer, corpus)

        return get_bertscore_token_weights(self.tokenizer)

    def _select_by_tokens(self, token_weight, input_tokens):
        """From a batch of tokenized texts `input_tokens`, returns an
        identically shaped tensor where each item `token_id` becomes
        `token_weight[token_id]`."""
        return token_weight.index_select(
            dim=0, index=input_tokens.flatten()
        ).reshape(input_tokens.shape)
