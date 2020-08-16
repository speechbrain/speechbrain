"""
The ``metric_stats`` module provides an abstract class for storing
statistics produced over the course of an experiment, and summarizing them.

Author:
 * Peter Plantinga 2020
"""
import torch
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.edit_distance import wer_summary


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

    Example
    -------
    >>> from speechbrain.nnet.losses import l1_loss
    >>> loss_stats = MetricStats(
    ...      metric=lambda x, y: l1_loss(x, y, reduction='batch')
    ... )
    >>> loss_stats.append(
    ...      ids=['utterance1', 'utterance2'],
    ...      predict=torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
    ...      target=torch.tensor([[0.1, 0.2], [0.1, 0.2]]),
    ... )
    >>> stats = loss_stats.summarize(just_avg=False)
    >>> stats['avg']
    tensor(0.0500)
    >>> stats['max_score']
    tensor(0.1000)
    >>> stats['max_id']
    'utterance2'
    """

    def __init__(self, metric):
        self.metric = metric
        self.clear()

    def clear(self):
        """Creates empty container for storage, removing existing stats."""
        self.scores = []
        self.ids = []
        self.summary = {}

    def append(self, ids, predict, target, pred_len=None, target_len=None):
        """Store a particular set of metric scores.

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : torch.tensor
            A predicted output, for comparison with the target output
        target : torch.tensor
            The correct reference output, for comparison with the prediction.
        pred_len : torch.tensor
            The predicted outputs' relative lengths.
        target_len : torch.tensor
            The target outputs' relative lengths.
        """
        self.ids.extend(ids)

        args = [predict, target]
        if pred_len is not None:
            args.append(pred_len)
        if target_len is not None:
            args.append(target_len)
        scores = self.metric(*args).detach()

        self.scores.extend(scores)

    def summarize(self, just_avg=True):
        """Summarize the metric scores, returning relevant stats.

        Arguments
        ---------
        just_avg : bool
            Whether to return just the average, or all stats in a dict

        Returns
        -------
        stats : float if just_avg, else dict
            Returns a float if just_avg option is True, otherwise
            returns a dictionary containing all computed stats.
        """
        min_index = torch.argmin(torch.tensor(self.scores))
        max_index = torch.argmax(torch.tensor(self.scores))
        self.summary = {
            "avg": sum(self.scores) / len(self.scores),
            "min_score": self.scores[min_index],
            "min_id": self.ids[min_index],
            "max_score": self.scores[max_index],
            "max_id": self.ids[max_index],
        }

        if just_avg:
            return self.summary["avg"]
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

        message = f"Average score: {self.summary['avg']}\n"
        message += f"Min error: {self.summary['min_score']} "
        message += f"id: {self.summary['min_id']}\n"
        message += f"Max error: {self.summary['max_score']} "
        message += f"id: {self.summary['max_id']}\n"

        filestream.write(message)
        if verbose:
            print(message)


class ErrorRateStats(MetricStats):
    """A class for tracking error rates (e.g. WER, PER).

    Example
    -------
    >>> cer_stats = ErrorRateStats()
    >>> cer_stats.append(
    ...     ids=['utterance1'],
    ...     predict=torch.tensor([[0, 1, 1]]),
    ...     target=torch.tensor([[0, 1, 0]]),
    ...     target_len=torch.ones(1),
    ...     ind2lab={0: 'a', 1: 'b'},
    ... )
    >>> stats = cer_stats.summarize(just_avg=False)
    >>> stats['WER']
    33.33...
    >>> stats['insertions']
    0
    >>> stats['deletions']
    0
    >>> stats['substitutions']
    1
    """

    def __init__(self):
        self.clear()

    def append(self, ids, predict, target, target_len, ind2lab=None):
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
        target_len : torch.tensor
            The target outputs' relative lengths.
        ind2lab : dict
            Mapping from indices to labels, for writing alignments.
        """
        self.ids.extend(ids)
        target_lab = undo_padding(target, target_len)

        if ind2lab is not None:
            predict = convert_index_to_lab(predict, ind2lab)
            target_lab = convert_index_to_lab(target_lab, ind2lab)

        scores = edit_distance.wer_details_for_batch(
            ids, target_lab, predict, compute_alignments=ind2lab is not None
        )

        self.scores.extend(scores)

    def summarize(self, just_avg=True):
        """Summarize the error_rate and return relevant statistics.

        * See MetricStats.summarize()
        """
        self.summary = wer_summary(self.scores)

        if just_avg:
            return self.summary["WER"]
        else:
            return self.summary

    def write_stats(self, filestream):
        """Write all relevant info (e.g. error rate alignments) to file.

        * See MetricStats.write_stats()
        """
        if not self.summary:
            self.summarize()

        wer_io.print_wer_summary(self.summary, filestream)
        wer_io.print_alignments(self.scores, filestream)
