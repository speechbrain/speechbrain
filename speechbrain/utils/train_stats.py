"""
The ``train_stats`` module provides a convenient mechanism for capturing
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


class TrainStats:
    """A class for storing and summarizing statistics from an experiment.

    Arguments
    ---------
    index2label : dict of int:str pairs
        A mapping from indices used during training to string labels.

    Example
    -------
    >>> from speechbrain.nnet.losses import l1_loss
    >>> loss_stats = TrainStats(
    ...      summary_fn='average',
    ...      error_fn=lambda x, y: l1_loss(x, y, reduction='batch')
    ... )
    >>> loss_stats.append(
    ...      ids=['utterance1', 'utterance2'],
    ...      predict=torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
    ...      target=torch.tensor([[0.1, 0.2], [0.1, 0.2]]),
    ... )
    >>> loss_stats.summarize()
    tensor(0.0500)
    """

    def __init__(self, summary_fn, error_fn=None):
        self.summary_fn = summary_fn
        self.error_fn = error_fn

        # Create empty containers for storage
        self.clear()

    def clear(self):
        """Creates empty container for storage, removing existing stats.

        Example usage: at end of epoch to start over for new epoch."""
        self.errors = []
        self.ids = []

    def append(
        self, ids, predict, target, pred_len=None, target_len=None, ind2lab=None
    ):
        """Add stats to the relevant containers.

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        *args, **kwargs
            Arguments to be forwarded to ``error_fn``.
        """
        self.ids.extend(ids)

        if self.summary_fn == "average":
            args = [predict, target]
            if pred_len is not None:
                args.append(pred_len)
            if target_len is not None:
                args.append(target_len)
            errors = self.error_fn(*args).detach()
        elif self.summary_fn == "error_rate":
            predict_lab = convert_index_to_lab(predict, ind2lab)
            target_lab = undo_padding(target, target_len)
            target_lab = convert_index_to_lab(target_lab, ind2lab)
            errors = edit_distance.wer_details_for_batch(
                ids, target_lab, predict_lab, compute_alignments=True
            )

        self.errors.extend(errors)

    def summarize(self):
        """Summarize the statistic according to selected summary type."""
        if self.summary_fn == "average":
            return sum(self.errors) / len(self.errors)
        elif self.summary_fn == "error_rate":
            summary = wer_summary(self.errors)
            return summary["WER"]
        elif self.summary_fn == "EER":
            raise NotImplementedError
        elif self.summary_fn == "accuracy":
            raise NotImplementedError

    def write_stats(self, filename, verbose=True):
        """Write all relevant info (e.g. error rate alignments) to file.

        Arguments
        ---------
        filename : str
            A location for storing the stats to disk.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        message = ""
        if self.summary_fn == "average":
            min_index = torch.argmin(torch.tensor(self.errors))
            max_index = torch.argmax(torch.tensor(self.errors))
            message += f"Average error: {self.summarize()}\n"
            message += f"Min error: {self.errors[min_index]} "
            message += f"id: {self.ids[min_index]}\n"
            message += f"Max error: {self.errors[max_index]} "
            message += f"id: {self.ids[max_index]}\n"
            with open(filename, "w") as w:
                w.write(message)
            if verbose:
                print(message)
        elif self.summary_fn == "error_rate":
            with open(filename, "w") as w:
                summary = wer_summary(self.errors)
                wer_io.print_wer_summary(summary, w)
                wer_io.print_alignments(self.errors, w)
