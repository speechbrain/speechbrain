"""The ``metric_stats`` module provides an abstract class for storing
statistics produced over the course of an experiment and summarizing them.

Authors:
 * Peter Plantinga 2020
 * Mirco Ravanelli 2020
 * Gaëlle Laperrière 2021
 * Sahar Ghannay 2021
"""

from typing import Callable, List, Optional

import torch
from joblib import Parallel, delayed

from speechbrain.dataio.dataio import (
    extract_concepts_values,
    merge_char,
    split_word,
)
from speechbrain.dataio.wer import print_alignments, print_wer_summary
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.edit_distance import (
    EDIT_SYMBOLS,
    _str_equals,
    wer_details_for_batch,
    wer_summary,
)


class MetricStats:
    """A default class for storing and summarizing arbitrary metrics.

    More complex metrics can be created by sub-classing this class.

    Arguments
    ---------
    metric : function
        The function to use to compute the relevant metric. Should take
        at least two arguments (predictions and targets) and can
        optionally take the relative lengths of either or both arguments.
        Not usually used in sub-classes.
    n_jobs : int
        The number of jobs to use for computing the metric. If this is
        more than one, every sample is processed individually, otherwise
        the whole batch is passed at once.
    batch_eval : bool
        When True it feeds the evaluation metric with the batched input.
        When False and n_jobs=1, it performs metric evaluation one-by-one
        in a sequential way. When False and n_jobs>1, the evaluation
        runs in parallel over the different inputs using joblib.

    Example
    -------
    >>> from speechbrain.nnet.losses import l1_loss
    >>> loss_stats = MetricStats(metric=l1_loss)
    >>> loss_stats.append(
    ...      ids=["utterance1", "utterance2"],
    ...      predictions=torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
    ...      targets=torch.tensor([[0.1, 0.2], [0.1, 0.2]]),
    ...      reduction="batch",
    ... )
    >>> stats = loss_stats.summarize()
    >>> stats['average']
    0.050...
    >>> stats['max_score']
    0.100...
    >>> stats['max_id']
    'utterance2'
    """

    def __init__(self, metric, n_jobs=1, batch_eval=True):
        self.metric = metric
        self.n_jobs = n_jobs
        self.batch_eval = batch_eval
        self.clear()

    def clear(self):
        """Creates empty container for storage, removing existing stats."""
        self.scores = []
        self.ids = []
        self.summary = {}

    def append(self, ids, *args, **kwargs):
        """Store a particular set of metric scores.

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        *args : tuple
            Arguments to pass to the metric function.
        **kwargs : dict
            Arguments to pass to the metric function.
        """
        self.ids.extend(ids)

        # Batch evaluation
        if self.batch_eval:
            scores = self.metric(*args, **kwargs).detach()

        else:
            if "predict" not in kwargs or "target" not in kwargs:
                raise ValueError(
                    "Must pass 'predict' and 'target' as kwargs if batch_eval=False"
                )
            if self.n_jobs == 1:
                # Sequence evaluation (loop over inputs)
                scores = sequence_evaluation(metric=self.metric, **kwargs)
            else:
                # Multiprocess evaluation
                scores = multiprocess_evaluation(
                    metric=self.metric, n_jobs=self.n_jobs, **kwargs
                )

        self.scores.extend(scores)

    def summarize(self, field=None):
        """Summarize the metric scores, returning relevant stats.

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
        min_index = torch.argmin(torch.tensor(self.scores))
        max_index = torch.argmax(torch.tensor(self.scores))
        self.summary = {
            "average": float(sum(self.scores) / len(self.scores)),
            "min_score": float(self.scores[min_index]),
            "min_id": self.ids[min_index],
            "max_score": float(self.scores[max_index]),
            "max_id": self.ids[max_index],
        }

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream, verbose=False):
        """Write all relevant statistics to file.

        Arguments
        ---------
        filestream : file-like object
            A stream for the stats to be written to.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        if not self.summary:
            self.summarize()

        message = f"Average score: {self.summary['average']}\n"
        message += f"Min error: {self.summary['min_score']} "
        message += f"id: {self.summary['min_id']}\n"
        message += f"Max error: {self.summary['max_score']} "
        message += f"id: {self.summary['max_id']}\n"

        filestream.write(message)
        if verbose:
            print(message)


def multiprocess_evaluation(metric, predict, target, lengths=None, n_jobs=8):
    """Runs metric evaluation if parallel over multiple jobs."""
    if lengths is not None:
        lengths = (lengths * predict.size(1)).round().int().cpu()
        predict = [p[:length].cpu() for p, length in zip(predict, lengths)]
        target = [t[:length].cpu() for t, length in zip(target, lengths)]

    while True:
        try:
            scores = Parallel(n_jobs=n_jobs, timeout=30)(
                delayed(metric)(p, t) for p, t in zip(predict, target)
            )
            break
        except Exception as e:
            print(e)
            print("Evaluation timeout...... (will try again)")

    return scores


def sequence_evaluation(metric, predict, target, lengths=None):
    """Runs metric evaluation sequentially over the inputs."""
    if lengths is not None:
        lengths = (lengths * predict.size(1)).round().int().cpu()
        predict = [p[:length].cpu() for p, length in zip(predict, lengths)]
        target = [t[:length].cpu() for t, length in zip(target, lengths)]

    scores = []
    for p, t in zip(predict, target):
        score = metric(p, t)
        scores.append(score)
    return scores


class ErrorRateStats(MetricStats):
    """A class for tracking error rates (e.g., WER, PER).

    Arguments
    ---------
    merge_tokens : bool
        Whether to merge the successive tokens (used for e.g.,
        creating words out of character tokens).
        See ``speechbrain.dataio.dataio.merge_char``.
    split_tokens : bool
        Whether to split tokens (used for e.g. creating
        characters out of word tokens).
        See ``speechbrain.dataio.dataio.split_word``.
    space_token : str
        The character to use for boundaries. Used with ``merge_tokens``
        this represents character to split on after merge.
        Used with ``split_tokens`` the sequence is joined with
        this token in between, and then the whole sequence is split.
    keep_values : bool
        Whether to keep the values of the concepts or not.
    extract_concepts_values : bool
        Process the predict and target to keep only concepts and values.
    tag_in : str
        Start of the concept ('<' for example).
    tag_out : str
        End of the concept ('>' for example).
    equality_comparator : Callable[[str, str], bool]
        The function used to check whether two words are equal.

    Example
    -------
    >>> cer_stats = ErrorRateStats()
    >>> i2l = {0: 'a', 1: 'b'}
    >>> cer_stats.append(
    ...     ids=['utterance1'],
    ...     predict=torch.tensor([[0, 1, 1]]),
    ...     target=torch.tensor([[0, 1, 0]]),
    ...     target_len=torch.ones(1),
    ...     ind2lab=lambda batch: [[i2l[int(x)] for x in seq] for seq in batch],
    ... )
    >>> stats = cer_stats.summarize()
    >>> stats['WER']
    33.33...
    >>> stats['insertions']
    0
    >>> stats['deletions']
    0
    >>> stats['substitutions']
    1
    """

    def __init__(
        self,
        merge_tokens=False,
        split_tokens=False,
        space_token="_",
        keep_values=True,
        extract_concepts_values=False,
        tag_in="",
        tag_out="",
        equality_comparator: Callable[[str, str], bool] = _str_equals,
    ):
        self.clear()
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token
        self.extract_concepts_values = extract_concepts_values
        self.keep_values = keep_values
        self.tag_in = tag_in
        self.tag_out = tag_out
        self.equality_comparator = equality_comparator

    def append(
        self,
        ids,
        predict,
        target,
        predict_len=None,
        target_len=None,
        ind2lab=None,
    ):
        """Add stats to the relevant containers.

        * See MetricStats.append()

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : torch.tensor
            A predicted output, for comparison with the target output
        target : torch.tensor
            The correct reference output, for comparison with the prediction.
        predict_len : torch.tensor
            The predictions relative lengths, used to undo padding if
            there is padding present in the predictions.
        target_len : torch.tensor
            The target outputs' relative lengths, used to undo padding if
            there is padding present in the target.
        ind2lab : callable
            Callable that maps from indices to labels, operating on batches,
            for writing alignments.
        """
        self.ids.extend(ids)

        if predict_len is not None:
            predict = undo_padding(predict, predict_len)

        if target_len is not None:
            target = undo_padding(target, target_len)

        if ind2lab is not None:
            predict = ind2lab(predict)
            target = ind2lab(target)

        if self.merge_tokens:
            predict = merge_char(predict, space=self.space_token)
            target = merge_char(target, space=self.space_token)

        if self.split_tokens:
            predict = split_word(predict, space=self.space_token)
            target = split_word(target, space=self.space_token)

        if self.extract_concepts_values:
            predict = extract_concepts_values(
                predict,
                self.keep_values,
                self.tag_in,
                self.tag_out,
                space=self.space_token,
            )
            target = extract_concepts_values(
                target,
                self.keep_values,
                self.tag_in,
                self.tag_out,
                space=self.space_token,
            )

        scores = wer_details_for_batch(
            ids,
            target,
            predict,
            compute_alignments=True,
            equality_comparator=self.equality_comparator,
        )

        self.scores.extend(scores)

    def summarize(self, field=None):
        """Summarize the error_rate and return relevant statistics.

        * See MetricStats.summarize()
        """
        self.summary = wer_summary(self.scores)

        # Add additional, more generic key
        self.summary["error_rate"] = self.summary["WER"]

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        if not self.summary:
            self.summarize()

        print_wer_summary(self.summary, filestream)
        print_alignments(self.scores, filestream)


class WeightedErrorRateStats(MetricStats):
    """Metric that reweighs the WER from :class:`~ErrorRateStats` with any
    chosen method. This does not edit the sequence of found edits
    (insertion/deletion/substitution) but multiplies their impact on the metric
    by a value between 0 and 1 as returned by the cost function.

    Arguments
    ---------
    base_stats : ErrorRateStats
        The base WER calculator to use.
    cost_function : Callable[[str, Optional[str], Optional[str]], float]
        Cost function of signature `fn(edit_symbol, a, b) -> float`, where the
        returned value, between 0 and 1, is the weight that should be assigned
        to a particular edit in the weighted WER calculation.
        In the case of insertions and deletions, either of `a` or `b` may be
        `None`. In the case of substitutions, `a` and `b` will never be `None`.
    weight_name : str
        Prefix to be prepended to each metric name (e.g. `xxx_wer`)
    """

    def __init__(
        self,
        base_stats: ErrorRateStats,
        cost_function: Callable[[str, Optional[str], Optional[str]], float],
        weight_name: str = "weighted",
    ):
        self.clear()
        self.base_stats = base_stats
        self.cost_function = cost_function
        self.weight_name = weight_name

    def append(self, *args, **kwargs):
        """Append function, which should **NOT** be used for the weighted error
        rate stats. Please append to the specified `base_stats` instead.

        `WeightedErrorRateStats` reuses the scores from the base
        :class:`~ErrorRateStats` class.

        Arguments
        ---------
        *args : tuple
            Ignored.
        **kwargs : dict
            Ignored.
        """

        raise ValueError(
            "Cannot append to a WeightedErrorRateStats. "
            "You should only append to the base ErrorRateStats."
        )

    def summarize(self, field=None):
        """Returns a dict containing some detailed WER statistics after
        weighting every edit with a weight determined by `cost_function`
        (returning `0.0` for no error, `1.0` for the default error behavior, and
        anything in between).

        Does not require :meth:`~ErrorRateStats.summarize` to have been called.

        Full set of fields, **each of which are prepended with
        `<weight_name_specified_at_init>_`**:
        - `wer`: Weighted WER (ratio `*100`)
        - `insertions`: Weighted insertions
        - `substitutions`: Weighted substitutions
        - `deletions`: Weighted deletions
        - `num_edits`: Sum of weighted insertions/substitutions/deletions

        Additionally, a `scores` list is populated by this function for each
        pair of sentences. Each entry of that list is a dict, with the fields:
        - `key`: the ID of the utterance.
        - `WER`, `insertions`, `substitutions`, `deletions`, `num_edits` with
          the same semantics as described above, but at sentence level rather
          than global.

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

        weighted_insertions = 0.0
        weighted_substitutions = 0.0
        weighted_deletions = 0.0
        total = 0.0

        for i, utterance in enumerate(self.base_stats.scores):
            utt_weighted_insertions = 0.0
            utt_weighted_substitutions = 0.0
            utt_weighted_deletions = 0.0
            utt_total = 0.0

            for edit_symbol, a_idx, b_idx in utterance["alignment"]:
                a = (
                    utterance["ref_tokens"][a_idx]
                    if a_idx is not None
                    else None
                )
                b = (
                    utterance["hyp_tokens"][b_idx]
                    if b_idx is not None
                    else None
                )

                if edit_symbol != EDIT_SYMBOLS["eq"]:
                    pair_score = self.cost_function(edit_symbol, a, b)

                    if edit_symbol == EDIT_SYMBOLS["ins"]:
                        utt_weighted_insertions += pair_score
                    elif edit_symbol == EDIT_SYMBOLS["del"]:
                        utt_weighted_deletions += pair_score
                    elif edit_symbol == EDIT_SYMBOLS["sub"]:
                        utt_weighted_substitutions += pair_score

                utt_total += 1.0

            utt_weighted_edits = (
                utt_weighted_insertions
                + utt_weighted_substitutions
                + utt_weighted_deletions
            )
            utt_weighted_wer_ratio = utt_weighted_edits / utt_total
            self.scores.append(
                {
                    "key": self.base_stats.ids[i],
                    "WER": utt_weighted_wer_ratio * 100.0,
                    "insertions": utt_weighted_insertions,
                    "substitutions": utt_weighted_substitutions,
                    "deletions": utt_weighted_deletions,
                    "num_edits": utt_weighted_edits,
                }
            )

            weighted_insertions += utt_weighted_insertions
            weighted_substitutions += utt_weighted_substitutions
            weighted_deletions += utt_weighted_deletions
            total += utt_total

        weighted_edits = (
            weighted_insertions + weighted_substitutions + weighted_deletions
        )
        weighted_wer_ratio = weighted_edits / total

        self.summary = {
            f"{self.weight_name}_wer": weighted_wer_ratio * 100.0,
            f"{self.weight_name}_insertions": weighted_insertions,
            f"{self.weight_name}_substitutions": weighted_substitutions,
            f"{self.weight_name}_deletions": weighted_deletions,
            f"{self.weight_name}_num_edits": weighted_edits,
        }

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream):
        """Write all relevant info to file; here, only the weighted info as
        returned by `summarize`.
        See :meth:`~ErrorRateStats.write_stats`.
        """
        if not self.summary:
            self.summarize()

        print(f"Weighted WER metrics ({self.weight_name}):", file=filestream)

        for k, v in self.summary.items():
            print(f"{k}: {v}", file=filestream)


class EmbeddingErrorRateSimilarity:
    """Implements the similarity function from the EmbER metric as defined by
    https://www.isca-archive.org/interspeech_2022/roux22_interspeech.pdf

    This metric involves a dictionary to map a token to a single word embedding.
    Substitutions in the WER get weighted down when the embeddings are similar
    enough. The goal is to reduce the impact of substitution errors with small
    semantic impact. Only substitution errors get weighted.

    This is done by computing the cosine similarity between the two embeddings,
    then weighing the substitution with `low_similarity_weight` if
    `similarity >= threshold` or with `high_similarity_weight` otherwise (e.g.
    a substitution with high similarity could be weighted down to matter 10% as
    much as a substitution with low similarity).

    .. note ::
        The cited paper recommended `(1.0, 0.1, 0.4)` as defaults for fastTexst
        French embeddings, chosen empirically. When using different embeddings,
        you might want to test other values; thus we don't provide defaults.

    Arguments
    ---------
    embedding_function : Callable[[str], Optional[torch.Tensor]]
        Function that returns an embedding (as a :class:`torch.Tensor`) from a
        word. If no corresponding embedding could be found for the word, should
        return `None`. In that case, `low_similarity_weight` will be chosen.
    low_similarity_weight : float
        Weight applied to the substitution if `cosine_similarity < threshold`.
    high_similarity_weight : float
        Weight applied to the substitution if `cosine_similarity >= threshold`.
    threshold : float
        Cosine similarity threshold used to select by how much a substitution
        error should be weighed for this word.
    """

    def __init__(
        self,
        embedding_function: Callable[[str], Optional[torch.Tensor]],
        low_similarity_weight: float,
        high_similarity_weight: float,
        threshold: float,
    ):
        self.embedding_function = embedding_function
        self.low_similarity_weight = low_similarity_weight
        self.high_similarity_weight = high_similarity_weight
        self.threshold = threshold

    def __call__(
        self, edit_symbol: str, a: Optional[str], b: Optional[str]
    ) -> float:
        """Returns the weight that should be associated with a specific edit
        in the WER calculation.

        Compatible candidate for the cost function of
        :class:`~WeightedErrorRateStats` so an instance of this class can be
        passed as a `cost_function`.

        Arguments
        ---------
        edit_symbol: str
            Edit symbol as assigned by the WER functions, see `EDIT_SYMBOLS`.
        a: str, optional
            First word to compare (if present)
        b: str, optional
            Second word to compare (if present)

        Returns
        -------
        float
            Weight to assign to the edit.
            For actual edits, either `low_similarity_weight` or
            `high_similarity_weight` depending on the embedding distance and
            threshold.
        """
        if edit_symbol in (EDIT_SYMBOLS["ins"], EDIT_SYMBOLS["del"]):
            return 1.0

        if edit_symbol == EDIT_SYMBOLS["sub"]:
            if a is None or a == "":
                return self.low_similarity_weight

            if b is None or b == "":
                return self.low_similarity_weight

            a_emb = self.embedding_function(a)
            if a_emb is None:
                return self.low_similarity_weight

            b_emb = self.embedding_function(b)
            if b_emb is None:
                return self.low_similarity_weight

            similarity = torch.nn.functional.cosine_similarity(
                a_emb, b_emb, dim=0
            ).item()

            if similarity >= self.threshold:
                return self.high_similarity_weight

            return self.low_similarity_weight

        # eq
        return 0.0


class BinaryMetricStats(MetricStats):
    """Tracks binary metrics, such as precision, recall, F1, EER, etc."""

    def __init__(self, positive_label=1):
        self.clear()
        self.positive_label = positive_label

    def clear(self):
        """Clears the stored metrics."""
        self.ids = []
        self.scores = []
        self.labels = []
        self.summary = {}

    def append(self, ids, scores, labels):
        """Appends scores and labels to internal lists.

        Does not compute metrics until time of summary, since
        automatic thresholds (e.g., EER) need full set of scores.

        Arguments
        ---------
        ids : list
            The string ids for the samples.
        scores : list
            The scores corresponding to the ids.
        labels : list
            The labels corresponding to the ids.
        """
        self.ids.extend(ids)
        self.scores.extend(scores.detach())
        self.labels.extend(labels.detach())

    def summarize(
        self, field=None, threshold=None, max_samples=None, beta=1, eps=1e-8
    ):
        """Compute statistics using a full set of scores.

        Full set of fields:
         - TP - True Positive
         - TN - True Negative
         - FP - False Positive
         - FN - False Negative
         - FAR - False Acceptance Rate
         - FRR - False Rejection Rate
         - DER - Detection Error Rate (EER if no threshold passed)
         - threshold - threshold (EER threshold if no threshold passed)
         - precision - Precision (positive predictive value)
         - recall - Recall (sensitivity)
         - F-score - Balance of precision and recall (equal if beta=1)
         - MCC - Matthews Correlation Coefficient

        Arguments
        ---------
        field : str
            A key for selecting a single statistic. If not provided,
            a dict with all statistics is returned.
        threshold : float
            If no threshold is provided, equal error rate is used.
        max_samples: float
            How many samples to keep for positive/negative scores.
            If no max_samples is provided, all scores are kept.
            Only effective when threshold is None.
        beta : float
            How much to weight precision vs recall in F-score. Default
            of 1. is equal weight, while higher values weight recall
            higher, and lower values weight precision higher.
        eps : float
            A small value to avoid dividing by zero.

        Returns
        -------
        summary
            if field is specified, only returns the score for that field.
            if field is None, returns the full set of fields.
        """
        if isinstance(self.scores, list):
            self.scores = torch.stack(self.scores)
            self.labels = torch.stack(self.labels)

        if threshold is None:
            positive_scores = self.scores[
                (self.labels == self.positive_label).nonzero(as_tuple=True)
            ]
            negative_scores = self.scores[
                (self.labels != self.positive_label).nonzero(as_tuple=True)
            ]
            if max_samples is not None:
                if len(positive_scores) > max_samples:
                    positive_scores, _ = torch.sort(positive_scores)
                    positive_scores = positive_scores[
                        [
                            i
                            for i in range(
                                0,
                                len(positive_scores),
                                int(len(positive_scores) / max_samples),
                            )
                        ]
                    ]
                if len(negative_scores) > max_samples:
                    negative_scores, _ = torch.sort(negative_scores)
                    negative_scores = negative_scores[
                        [
                            i
                            for i in range(
                                0,
                                len(negative_scores),
                                int(len(negative_scores) / max_samples),
                            )
                        ]
                    ]

            eer, threshold = EER(positive_scores, negative_scores)

        pred = (self.scores > threshold).float()
        true = self.labels

        TP = self.summary["TP"] = float(pred.mul(true).sum())
        TN = self.summary["TN"] = float((1.0 - pred).mul(1.0 - true).sum())
        FP = self.summary["FP"] = float(pred.mul(1.0 - true).sum())
        FN = self.summary["FN"] = float((1.0 - pred).mul(true).sum())

        self.summary["FAR"] = FP / (FP + TN + eps)
        self.summary["FRR"] = FN / (TP + FN + eps)
        self.summary["DER"] = (FP + FN) / (TP + TN + eps)
        self.summary["threshold"] = threshold

        self.summary["precision"] = TP / (TP + FP + eps)
        self.summary["recall"] = TP / (TP + FN + eps)
        self.summary["F-score"] = (
            (1.0 + beta**2.0)
            * TP
            / ((1.0 + beta**2.0) * TP + beta**2.0 * FN + FP)
        )

        self.summary["MCC"] = (TP * TN - FP * FN) / (
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps
        ) ** 0.5

        if field is not None:
            return self.summary[field]
        else:
            return self.summary


def EER(positive_scores, negative_scores):
    """Computes the EER (and its threshold).

    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.

    Returns
    -------
    EER : float
        The EER score.
    threshold : float
        The corresponding threshold for the EER score.

    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_eer, threshold = EER(positive_scores, negative_scores)
    >>> val_eer
    0.0
    """
    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    intermediate_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, intermediate_thresholds]))

    # Variable to store the min FRR, min FAR and their corresponding index
    min_index = 0
    final_FRR = 0
    final_FAR = 0

    for i, cur_thresh in enumerate(thresholds):
        pos_scores_threshold = positive_scores <= cur_thresh
        FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[0]
        del pos_scores_threshold

        neg_scores_threshold = negative_scores > cur_thresh
        FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[0]
        del neg_scores_threshold

        # Finding the threshold for EER
        if (FAR - FRR).abs().item() < abs(final_FAR - final_FRR) or i == 0:
            min_index = i
            final_FRR = FRR.item()
            final_FAR = FAR.item()

    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    EER = (final_FAR + final_FRR) / 2

    return float(EER), float(thresholds[min_index])


def minDCF(
    positive_scores, negative_scores, c_miss=1.0, c_fa=1.0, p_target=0.01
):
    """Computes the minDCF metric normally used to evaluate speaker verification
    systems. The min_DCF is the minimum of the following C_det function computed
    within the defined threshold range:

    C_det =  c_miss * p_miss * p_target + c_fa * p_fa * (1 -p_target)

    where p_miss is the missing probability and p_fa is the probability of having
    a false alarm.

    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.
    c_miss : float
         Cost assigned to a missing error (default 1.0).
    c_fa : float
        Cost assigned to a false alarm (default 1.0).
    p_target: float
        Prior probability of having a target (default 0.01).

    Returns
    -------
    minDCF : float
        The minDCF score.
    threshold : float
        The corresponding threshold for the minDCF score.

    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_minDCF, threshold = minDCF(positive_scores, negative_scores)
    >>> val_minDCF
    0.0
    """
    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    intermediate_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, intermediate_thresholds]))

    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    p_miss = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    # Computing False Acceptance Rate (false alarm)
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    p_fa = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    c_min, min_index = torch.min(c_det, dim=0)

    return float(c_min), float(thresholds[min_index])


class ClassificationStats(MetricStats):
    """Computes statistics pertaining to multi-label classification tasks, as
    well as tasks that can be loosely interpreted as such for the purpose of evaluations.

    Example
    -------
    >>> import sys
    >>> from speechbrain.utils.metric_stats import ClassificationStats
    >>> cs = ClassificationStats()
    >>> cs.append(
    ...     ids=["ITEM1", "ITEM2", "ITEM3", "ITEM4"],
    ...     predictions=[
    ...         "M EY K AH",
    ...         "T EY K",
    ...         "B AE D",
    ...         "M EY K",
    ...     ],
    ...     targets=[
    ...         "M EY K",
    ...         "T EY K",
    ...         "B AE D",
    ...         "M EY K",
    ...     ],
    ...     categories=[
    ...         "make",
    ...         "take",
    ...         "bad",
    ...         "make"
    ...     ]
    ... )
    >>> cs.write_stats(sys.stdout)
    Overall Accuracy: 75%
    <BLANKLINE>
    Class-Wise Accuracy
    -------------------
    bad -> B AE D : 1 / 1 (100.00%)
    make -> M EY K: 1 / 2 (50.00%)
    take -> T EY K: 1 / 1 (100.00%)
    <BLANKLINE>
    Confusion
    ---------
    Target: bad -> B AE D
      -> B AE D   : 1 / 1 (100.00%)
    Target: make -> M EY K
      -> M EY K   : 1 / 2 (50.00%)
      -> M EY K AH: 1 / 2 (50.00%)
    Target: take -> T EY K
      -> T EY K   : 1 / 1 (100.00%)
    >>> summary = cs.summarize()
    >>> summary['accuracy']
    0.75
    >>> summary['classwise_stats'][('bad', 'B AE D')]
    {'total': 1.0, 'correct': 1.0, 'accuracy': 1.0}
    >>> summary['classwise_stats'][('make', 'M EY K')]
    {'total': 2.0, 'correct': 1.0, 'accuracy': 0.5}
    >>> summary['keys']
    [('bad', 'B AE D'), ('make', 'M EY K'), ('take', 'T EY K')]
    >>> summary['predictions']
    ['B AE D', 'M EY K', 'M EY K AH', 'T EY K']
    >>> summary['classwise_total']
    {('bad', 'B AE D'): 1.0, ('make', 'M EY K'): 2.0, ('take', 'T EY K'): 1.0}
    >>> summary['classwise_correct']
    {('bad', 'B AE D'): 1.0, ('make', 'M EY K'): 1.0, ('take', 'T EY K'): 1.0}
    >>> summary['classwise_accuracy']
    {('bad', 'B AE D'): 1.0, ('make', 'M EY K'): 0.5, ('take', 'T EY K'): 1.0}
    """

    def __init__(self):
        super()
        self.clear()
        self.summary = None

    def append(self, ids, predictions, targets, categories=None):
        """
        Appends inputs, predictions and targets to internal
        lists

        Arguments
        ---------
        ids: list
            the string IDs for the samples
        predictions: list
            the model's predictions (human-interpretable,
            preferably strings)
        targets: list
            the ground truths (human-interpretable, preferably strings)
        categories: list
            an additional way to classify training
            samples. If available, the categories will
            be combined with targets
        """
        self.ids.extend(ids)
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        if categories is not None:
            self.categories.extend(categories)

    def summarize(self, field=None):
        """Summarize the classification metric scores

        The following statistics are computed:

        accuracy: the overall accuracy (# correct / # total)
        confusion_matrix: a dictionary of type
            {(target, prediction): num_entries} representing
            the confusion matrix
        classwise_stats: computes the total number of samples,
            the number of correct classifications and accuracy
            for each class
        keys: all available class keys, which can be either target classes
            or (category, target) tuples
        predictions: all available predictions all predictions the model
            has made

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
        self._build_lookups()
        confusion_matrix = self._compute_confusion_matrix()
        self.summary = {
            "accuracy": self._compute_accuracy(),
            "confusion_matrix": confusion_matrix,
            "classwise_stats": self._compute_classwise_stats(confusion_matrix),
            "keys": self._available_keys,
            "predictions": self._available_predictions,
        }
        for stat in ["total", "correct", "accuracy"]:
            self.summary[f"classwise_{stat}"] = {
                key: key_stats[stat]
                for key, key_stats in self.summary["classwise_stats"].items()
            }
        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def _compute_accuracy(self):
        return sum(
            prediction == target
            for prediction, target in zip(self.predictions, self.targets)
        ) / len(self.ids)

    def _build_lookups(self):
        self._available_keys = self._get_keys()
        self._available_predictions = list(
            sorted(set(prediction for prediction in self.predictions))
        )
        self._keys_lookup = self._index_lookup(self._available_keys)
        self._predictions_lookup = self._index_lookup(
            self._available_predictions
        )

    def _compute_confusion_matrix(self):
        confusion_matrix = torch.zeros(
            len(self._available_keys), len(self._available_predictions)
        )
        for key, prediction in self._get_confusion_entries():
            key_idx = self._keys_lookup[key]
            prediction_idx = self._predictions_lookup[prediction]
            confusion_matrix[key_idx, prediction_idx] += 1
        return confusion_matrix

    def _compute_classwise_stats(self, confusion_matrix):
        total = confusion_matrix.sum(dim=-1)

        # This can be used with "classes" that are not
        # statically determined; for example, they could
        # be constructed from seq2seq predictions. As a
        # result, one cannot use the diagonal
        key_targets = (
            self._available_keys
            if not self.categories
            else [target for _, target in self._available_keys]
        )
        correct = torch.tensor(
            [
                (
                    confusion_matrix[idx, self._predictions_lookup[target]]
                    if target in self._predictions_lookup
                    else 0
                )
                for idx, target in enumerate(key_targets)
            ]
        )
        accuracy = correct / total
        return {
            key: {
                "total": item_total.item(),
                "correct": item_correct.item(),
                "accuracy": item_accuracy.item(),
            }
            for key, item_total, item_correct, item_accuracy in zip(
                self._available_keys, total, correct, accuracy
            )
        }

    def _get_keys(self):
        if self.categories:
            keys = zip(self.categories, self.targets)
        else:
            keys = self.targets
        return list(sorted(set(keys)))

    def _get_confusion_entries(self):
        if self.categories:
            result = (
                ((category, target), prediction)
                for category, target, prediction in zip(
                    self.categories, self.targets, self.predictions
                )
            )
        else:
            result = zip(self.targets, self.predictions)
        result = list(result)
        return result

    def _index_lookup(self, items):
        return {item: idx for idx, item in enumerate(items)}

    def clear(self):
        """Clears the collected statistics"""
        self.ids = []
        self.predictions = []
        self.targets = []
        self.categories = []

    def write_stats(self, filestream):
        """Outputs the stats to the specified filestream in a human-readable format

        Arguments
        ---------
        filestream: file
            a file-like object
        """
        if self.summary is None:
            self.summarize()
        print(
            f"Overall Accuracy: {self.summary['accuracy']:.0%}", file=filestream
        )
        print(file=filestream)
        self._write_classwise_stats(filestream)
        print(file=filestream)
        self._write_confusion(filestream)

    def _write_classwise_stats(self, filestream):
        self._write_header("Class-Wise Accuracy", filestream=filestream)
        key_labels = {
            key: self._format_key_label(key) for key in self._available_keys
        }
        longest_key_label = max(len(label) for label in key_labels.values())
        for key in self._available_keys:
            stats = self.summary["classwise_stats"][key]
            padded_label = self._pad_to_length(
                self._format_key_label(key), longest_key_label
            )
            print(
                f"{padded_label}: {int(stats['correct'])} / {int(stats['total'])} ({stats['accuracy']:.2%})",
                file=filestream,
            )

    def _write_confusion(self, filestream):
        self._write_header("Confusion", filestream=filestream)
        longest_prediction = max(
            len(prediction) for prediction in self._available_predictions
        )
        confusion_matrix = self.summary["confusion_matrix"].int()
        totals = confusion_matrix.sum(dim=-1)
        for key, key_predictions, total in zip(
            self._available_keys, confusion_matrix, totals
        ):
            target_label = self._format_key_label(key)
            print(f"Target: {target_label}", file=filestream)
            (indexes,) = torch.where(key_predictions > 0)
            total = total.item()
            for index in indexes:
                count = key_predictions[index].item()
                prediction = self._available_predictions[index]
                padded_label = self._pad_to_length(
                    prediction, longest_prediction
                )
                print(
                    f"  -> {padded_label}: {count} / {total} ({count / total:.2%})",
                    file=filestream,
                )

    def _write_header(self, header, filestream):
        print(header, file=filestream)
        print("-" * len(header), file=filestream)

    def _pad_to_length(self, label, length):
        padding = max(0, length - len(label))
        return label + (" " * padding)

    def _format_key_label(self, key):
        if self.categories:
            category, target = key
            label = f"{category} -> {target}"
        else:
            label = key
        return label


class MultiMetricStats:
    """A wrapper that evaluates multiple metrics simultaneously

    Arguments
    ---------
    metric : function
        The function to use to compute the relevant metrics. Should take
        at least two arguments (predictions and targets) and can
        optionally take the relative lengths of either or both arguments.
        The function should return a dict or a namedtuple
    n_jobs : int
        The number of jobs to use for computing the metric. If this is
        more than one, every sample is processed individually, otherwise
        the whole batch is passed at once.
    batch_eval : bool
        When True it feeds the evaluation metric with the batched input.
        When False and n_jobs=1, it performs metric evaluation one-by-one
        in a sequential way. When False and n_jobs>1, the evaluation
        runs in parallel over the different inputs using joblib.

    Example
    -------
    >>> def metric(a, b):
    ...    return {
    ...        "sum": a + b,
    ...        "diff": a - b,
    ...        "sum_sq": a**2 + b**2
    ...    }
    >>> multi_metric = MultiMetricStats(metric, batch_eval=True)
    >>> multi_metric.append([1, 2], a=torch.tensor([2.0, 1.0]), b=torch.tensor([1.0, 2.0]))
    >>> multi_metric.append([3, 4], a=torch.tensor([4.0, 5.0]), b=torch.tensor([0.0, 1.0]))
    >>> multi_metric.append([5, 6], a=torch.tensor([2.0, 4.0]), b=torch.tensor([4.0, 2.0]))
    >>> multi_metric.append([7, 8], a=torch.tensor([2.0, 4.0]), b=torch.tensor([4.0, 2.0]))
    >>> multi_metric.summarize() #doctest: +NORMALIZE_WHITESPACE
    {'sum': {'average': 5.0,
      'min_score': 3.0,
      'min_id': 1,
      'max_score': 6.0,
      'max_id': 4},
     'diff': {'average': 1.0,
      'min_score': -2.0,
      'min_id': 5,
      'max_score': 4.0,
      'max_id': 3},
     'sum_sq': {'average': 16.5,
      'min_score': 5.0,
      'min_id': 1,
      'max_score': 26.0,
      'max_id': 4}}
    >>> multi_metric.summarize(flat=True) #doctest: +NORMALIZE_WHITESPACE
    {'sum_average': 5.0,
     'sum_min_score': 3.0,
     'sum_min_id': 1,
     'sum_max_score': 6.0,
     'sum_max_id': 4,
     'diff_average': 1.0,
     'diff_min_score': -2.0,
     'diff_min_id': 5,
     'diff_max_score': 4.0,
     'diff_max_id': 3,
     'sum_sq_average': 16.5,
     'sum_sq_min_score': 5.0,
     'sum_sq_min_id': 1,
     'sum_sq_max_score': 26.0,
     'sum_sq_max_id': 4}
    """

    def __init__(self, metric, n_jobs=1, batch_eval=False):
        self.metric = _dictify(metric)
        self.n_jobs = n_jobs
        self.batch_eval = batch_eval
        self.ids = []
        self.metrics = {}

    def append(self, ids, *args, **kwargs):
        """Store a particular set of metric scores.

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        *args : tuple
            Arguments to pass to the metric function.
        **kwargs : dict
            Arguments to pass to the metric function.
        """
        self.ids.extend(ids)

        # Batch evaluation
        if self.batch_eval:
            scores = self.eval_simple(*args, **kwargs)

        else:
            if "predict" not in kwargs or "target" not in kwargs:
                raise ValueError(
                    "Must pass 'predict' and 'target' as kwargs if batch_eval=False"
                )
            if self.n_jobs == 1:
                # Sequence evaluation (loop over inputs)
                scores_raw = sequence_evaluation(self.metric, **kwargs)
            else:
                # Multiprocess evaluation
                scores_raw = multiprocess_evaluation(
                    metric=self.metric, n_jobs=self.n_jobs, **kwargs
                )

            keys = scores_raw[0].keys()
            scores = {
                key: torch.tensor([score[key] for score in scores_raw])
                for key in keys
            }

        for key, metric_scores in scores.items():
            if key not in self.metrics:
                self.metrics[key] = MetricStats(lambda x: x, batch_eval=True)
            self.metrics[key].append(ids, metric_scores)

    def eval_simple(self, *args, **kwargs):
        """Evaluates the metric in a simple, sequential manner"""
        scores = self.metric(*args, **kwargs)
        return {key: score.detach() for key, score in scores.items()}

    def summarize(self, field=None, flat=False):
        """Summarize the metric scores, returning relevant stats.

        Arguments
        ---------
        field : str
            If provided, only returns selected statistic. If not,
            returns all computed statistics.
        flat : bool
            whether to flatten the dictionary

        Returns
        -------
        dict
            Returns a dictionary of all computed stats
        """
        result = {
            key: metric.summarize(field) for key, metric in self.metrics.items()
        }
        if flat:
            result = {
                f"{key}_{field}": value
                for key, fields in result.items()
                for field, value in fields.items()
            }
        return result


def _dictify(f):
    """A wrapper that converts functions returning
    namedtuples to functions returning dicts while leaving
    functions returning dicts intact

    Arguments
    ---------
    f : callable
        a function

    Returns
    -------
    result : callable
        a wrapped function
    """
    has_asdict = None

    def wrapper(*args, **kwargs):
        """The wrapper function"""
        nonlocal has_asdict
        result = f(*args, **kwargs)
        if has_asdict is None:
            has_asdict = hasattr(result, "_asdict")
        return result._asdict() if has_asdict else result

    return wrapper


def dialogue_state_str2dict(
    dialogue_state: str, slot_type_filtering: List[str] = None
):
    """
    Converts the ; separated Dialogue State linearization to a domain-slot-value dictionary.
    When *slot_type_filtering* is provided, it filters out the slots which are not part of this list.
    """
    dict_state = {}

    # Considering every word after "[State] " to discard the transcription if present.
    dialogue_state = dialogue_state.split("[State] ")[-1]
    if ";" not in dialogue_state:
        # One slot or none
        if "=" not in dialogue_state:
            return {}
        else:
            slot_value = dialogue_state.split("=")
            slot, value = (slot_value[0], slot_value[1])
            if "-" not in slot:
                return {}
            else:
                domain_slot = slot.split("-")
                domain, slot_type = (domain_slot[0], domain_slot[1])
                return {domain: {slot_type: value}}
    else:
        # Multiple slots
        slots = dialogue_state.split(";")
        for slot_value in slots:
            if "=" not in slot_value:
                continue
            else:
                slot, value = (
                    slot_value.split("=")[0].strip(),
                    slot_value.split("=")[1].strip(),
                )
                if slot_type_filtering and slot not in slot_type_filtering:
                    continue
                elif "-" in slot:
                    domain, slot_type = (
                        slot.split("-")[0].strip(),
                        slot.split("-")[1].strip(),
                    )
                    if domain not in dict_state.keys():
                        dict_state[domain] = {}
                    dict_state[domain][slot_type] = value

        return dict_state


class JointGoalAccuracyTracker:
    """
    Class to track the Joint-Goal Accuracy during training.
    Keeps track of the number of correct and total dialogue states considered.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        """
        Resets the correct and total counters of the metric.
        """
        self.correct = 0
        self.total = 0

    def append(self, predictions: List[str], targets: List[str]):
        """
        This function is for updating the stats according to the a batch of predictions and targets.

        Arguments
        ---------
        predictions : list[str]
            Predicted dialogue states.
        targets : list[str]
            Target dialogue states.
        """
        for prediction, reference in zip(predictions, targets):
            pred = dialogue_state_str2dict(prediction)
            ref = dialogue_state_str2dict(reference)
            if pred == ref:
                self.correct += 1
            self.total += 1

    def summarize(self):
        """
        Averages the current Joint-Goal Accuracy (JGA).
        Returns
        -------
        average_jga
           Current average Joint-Goal Accuracy (JGA).
        """
        average_jga = (
            round(100 * self.correct / self.total, 2)
            if self.total != 0
            else 100.00
        )
        return average_jga
