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
    """A abstract class for storing and summarizing metrics from an experiment.
    """

    def clear(self):
        """Creates empty container for storage, removing existing stats."""
        self.scores = []
        self.ids = []

    def append(self, ids, predict, target, pred_len=None, target_len=None):
        """Add stats to the relevant containers.

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
        raise NotImplementedError

    def summarize(self):
        """Summarize the statistic according to selected summary type."""
        raise NotImplementedError

    def write_stats(self, filestream, verbose=False):
        """Write all relevant info (e.g. error rate alignments) to file.

        Arguments
        ---------
        filestream : file-like object
            A stream for the stats to be written to.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        raise NotImplementedError


class AverageStats(MetricStats):
    """Class for summarizing a metric by averaging.

    Arguments
    ---------
    metric : function
        The function to use to compute the relevant metric. Should take
        at least two arguments (predictions and targets) and can
        optionally take the relative lengths of either or both arguments.

    Example
    -------
    >>> from speechbrain.nnet.losses import l1_loss
    >>> loss_stats = AverageStats(
    ...      metric=lambda x, y: l1_loss(x, y, reduction='batch')
    ... )
    >>> loss_stats.append(
    ...      ids=['utterance1', 'utterance2'],
    ...      predict=torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
    ...      target=torch.tensor([[0.1, 0.2], [0.1, 0.2]]),
    ... )
    >>> loss_stats.summarize()
    tensor(0.0500)
    """

    def __init__(self, metric):
        self.metric = metric
        self.clear()

    def append(self, ids, predict, target, pred_len=None, target_len=None):
        """Add stats to the relevant containers.

        * See MetricStats.append()
        """
        self.ids.extend(ids)

        args = [predict, target]
        if pred_len is not None:
            args.append(pred_len)
        if target_len is not None:
            args.append(target_len)
        score = self.metric(*args).detach()

        self.scores.append(score)

    def summarize(self):
        """Average the appended metric scores

        * See MetricStats.summarize()
        """
        return sum(self.scores) / len(self.scores)

    def write_stats(self, filestream, verbose=True):
        """Write all relevant info (e.g. error rate alignments) to file.

        * See MetricStats.write_stats()
        """
        min_index = torch.argmin(torch.tensor(self.scores))
        max_index = torch.argmax(torch.tensor(self.scores))

        message = f"Average error: {self.summarize()}\n"
        message += f"Min error: {self.scores[min_index]} "
        message += f"id: {self.ids[min_index]}\n"
        message += f"Max error: {self.scores[max_index]} "
        message += f"id: {self.ids[max_index]}\n"

        filestream.write(message)
        if verbose:
            print(message)


class ErrorRateStats(MetricStats):
    """A class for tracking error rates (e.g. WER, PER).

    Example
    -------
    >>> ind2lab = {0: 'a', 1: 'b'}
    >>> cer_stats = ErrorRateStats()
    >>> cer_stats.append(
    """

    def __init__(self):
        self.clear()

    def append(self, ids, predict, target, target_len, ind2lab):
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

        predict_lab = convert_index_to_lab(predict, ind2lab)
        target_lab = undo_padding(target, target_len)
        target_lab = convert_index_to_lab(target_lab, ind2lab)
        scores = edit_distance.wer_details_for_batch(
            ids, target_lab, predict_lab, compute_alignments=True
        )

        self.scores.extend(scores)

    def summarize(self, full_summary=False):
        """Summarize the error_rate and return relevant statistics.

        * See MetricStats.summarize()

        Arguments
        ---------
        full_summary : bool
            Whether to return all the error rate stats, or just the error rate.
        """
        summary = wer_summary(self.scores)

        if full_summary:
            return summary
        else:
            return summary["WER"]

    def write_stats(self, filestream):
        """Write all relevant info (e.g. error rate alignments) to file.

        * See MetricStats.write_stats()
        """
        summary = wer_summary(self.scores)
        wer_io.print_wer_summary(summary, filestream)
        wer_io.print_alignments(self.scores, filestream)
