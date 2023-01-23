"""The ``metric_stats`` module provides an abstract class for storing
statistics produced over the course of an experiment and summarizing them.

Authors:
 * Peter Plantinga 2020
 * Mirco Ravanelli 2020
 * Gaelle Laperriere 2021
 * Sahar Ghannay 2021
"""

import torch
from joblib import Parallel, delayed
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.edit_distance import wer_summary, wer_details_for_batch
from speechbrain.dataio.dataio import (
    merge_char,
    split_word,
    extract_concepts_values,
)
from speechbrain.dataio.wer import print_wer_summary, print_alignments


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
    batch_eval: bool
        When True it feeds the evaluation metric with the batched input.
        When False and n_jobs=1, it performs metric evaluation one-by-one
        in a sequential way. When False and n_jobs>1, the evaluation
        runs in parallel over the different inputs using joblib.
    n_jobs : int
        The number of jobs to use for computing the metric. If this is
        more than one, every sample is processed individually, otherwise
        the whole batch is passed at once.

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
        *args, **kwargs
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
        Start of the concept ('<' for exemple).
    tag_out : str
        End of the concept ('>' for exemple).

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
    ):
        self.clear()
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token
        self.extract_concepts_values = extract_concepts_values
        self.keep_values = keep_values
        self.tag_in = tag_in
        self.tag_out = tag_out

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

        scores = wer_details_for_batch(ids, target, predict, True)

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


class BinaryMetricStats(MetricStats):
    """Tracks binary metrics, such as precision, recall, F1, EER, etc.
    """

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
            The string ids for the samples

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
            How many samples to keep for postive/negative scores.
            If no max_samples is provided, all scores are kept.
            Only effective when threshold is None.
        beta : float
            How much to weight precision vs recall in F-score. Default
            of 1. is equal weight, while higher values weight recall
            higher, and lower values weight precision higher.
        eps : float
            A small value to avoid dividing by zero.
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
            (1.0 + beta ** 2.0)
            * TP
            / ((1.0 + beta ** 2.0) * TP + beta ** 2.0 * FN + FP)
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
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

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
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

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
    """Computes statistics pertaining to multi-label
    classification tasks, as well as tasks that can be loosely interpreted as such for the purpose of
    evaluations

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
        predictions: all available predictions all predicions the model
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
