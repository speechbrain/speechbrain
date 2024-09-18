"""Utilities for Text-to-Speech systems

Authors
 * Artem Ploujnikov 2023
"""

import csv
import logging
from contextlib import ExitStack
from io import StringIO

import torch

logger = logging.getLogger(__name__)

try:
    from matplotlib import pyplot as plt
except ImportError:
    logger.warn("matplotlib is not available, cannot save plots")
    plt = None


class TTSProgressReport:
    """A progress reporter for text-to-speech systems

    Arguments
    ---------
    logger : speechbrain.utils.train_logger.ArchiveTrainLogegr or compatible
        The logger that will be used to save results

    sample_rate : int
        The sample rate for audio

    eos_threshold : float
        The threshold probability at which the end-of-sequence gate
        output is considered as positive
    """

    def __init__(self, logger, sample_rate=24000, eos_threshold=0.5):
        self.logger = logger
        self.sample_rate = sample_rate
        self.eos_threshold = eos_threshold
        self._exit_stack = ExitStack()
        self._writing = False
        self._clear()

    def __enter__(self):
        self._clear()
        self._exit_stack.enter_context(self.logger)
        self._writing = True

    def __exit__(self, exc_type, exc_value, traceback):
        self._write_length_report()
        self._write_details_file()
        self._exit_stack.close()
        self._writing = False
        self._clear()

    def _clear(self):
        """Clears accumulator variables"""
        self.ids = []
        self.length_pred = []
        self.length = []
        self.details = None

    def _ensure_writing(self):
        """Throws ContextError if invoked without first entering the
        context"""
        if not self._writing:
            raise ContextError()

    def write(
        self,
        ids,
        audio,
        length_pred,
        length=None,
        tgt_max_length=None,
        alignments=None,
        p_eos=None,
    ):
        """Reports TTS inferents results

        Arguments
        ---------
        ids : list
            The list of IDs, from the dataset
        audio : torch.Tensor
            A padded tensor of audio samples
        length_pred : torch.Tensor
            A tensor of predicted relative lengths
        length : torch.Tensor
            A tensor of ground truth relative lengths
        tgt_max_length : int
            The maximum length of audio targets
        alignments : torch.Tensor, optional
            Attention alignments
        p_eos : torch.Tensor
            A (Batch x Length) tensor of end-of-sequence
            probabilities
        """
        self._ensure_writing()
        self.ids.extend(ids)
        self._write_audio(ids, audio, length_pred)
        self._write_details(ids, alignments, p_eos)
        if plt is not None:
            if alignments is not None:
                self._write_alignments(ids, alignments)
            if p_eos is not None:
                self._write_eos(ids, p_eos, length, tgt_max_length)

    def _write_audio(self, ids, audio, length):
        """Saves the raw audio

        Arguments
        ---------
        ids : list
            The list of IDs, from the dataset
        audio : torch.Tensor
            A padded tensor of audio samples
        length : torch.Tensor
            A tensor of relative lengths
        """
        length_abs = (length * audio.size(1)).int()
        for item_id, item_audio, item_length in zip(ids, audio, length_abs):
            item_audio_cut = item_audio[: item_length.item()]
            self.logger.save(
                name=f"{item_id}.wav",
                content=item_audio_cut.detach().cpu(),
                mode="audio",
                samplerate=self.sample_rate,
            )

    def _write_alignments(self, ids, alignments):
        """Outputs a plot of attention alignments

        Arguments
        ---------
        ids : list
            The list of IDs, from the dataset

        alignments : torch.Tensor
            Attention alignments
        """
        for item_id, item_alignment in zip(ids, alignments):
            try:
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.imshow(
                    item_alignment.transpose(-1, -2).detach().cpu(),
                    origin="lower",
                )
                ax.set_title(f"{item_id} Alignment")
                ax.set_xlabel("Audio")
                ax.set_ylabel("Text")
                self.logger.save(
                    name=f"{item_id}_alignment.png",
                    content=fig,
                    mode="figure",
                )
            finally:
                plt.close(fig)

    def _write_eos(self, ids, p_eos, length, tgt_max_length):
        """Outputs a plot of end-of-sequence gate outputs

        Arguments
        ---------
        ids : list
            The list of IDs, from the dataset
        p_eos : torch.Tensor
            A (Batch x Length) tensor of end-of-sequence
            probabilities
        length : torch.Tensor
            Ground truth lengths (relative)
        tgt_max_length : torch.Tensor
            The maximum length of the targets
        """
        p_eos, length = [x.detach().cpu() for x in [p_eos, length]]
        max_len = p_eos.size(1)
        length_abs = length * tgt_max_length
        gate_act = p_eos >= self.eos_threshold
        length_pred = torch.where(
            gate_act.sum(dim=-1) == 0, max_len, gate_act.int().argmax(dim=-1)
        )

        self.length.extend(length_abs.tolist())
        self.length_pred.extend(length_pred.tolist())
        for item_id, item_length, item_p_eos in zip(ids, length_abs, p_eos):
            fig, ax = plt.subplots(figsize=(8, 2))
            try:
                ax.set_title(f"{item_id} Gate")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Gate Output")
                ax.plot(item_p_eos)
                x = [item_length, item_length]
                y = [0.0, 1.0]
                ax.plot(x, y, color="blue", marker="o", label="Ground Truth")
                x = [0.0, max_len - 1]
                y = [self.eos_threshold, self.eos_threshold]
                ax.plot(x, y, color="red", marker="x", label="Threshold")
                ax.legend()
                self.logger.save(
                    name=f"{item_id}_gate.png",
                    content=fig,
                    mode="figure",
                )
            finally:
                plt.close(fig)

    def _write_length_report(self):
        """Outputs the length report"""
        with StringIO() as length_report:
            csv_writer = csv.DictWriter(
                length_report, ["uttid", "length_pred", "length", "length_diff"]
            )
            csv_writer.writeheader()
            for uttid, length_pred, length in zip(
                self.ids, self.length_pred, self.length
            ):
                csv_writer.writerow(
                    {
                        "uttid": uttid,
                        "length_pred": length_pred,
                        "length": length,
                        "length_diff": length_pred - length,
                    }
                )
            self.logger.save(
                name="length.csv", content=length_report.getvalue(), mode="text"
            )

    def _write_details_file(self):
        """Outputs the concatenated details file"""
        if self.details is not None:
            details = {key: value for key, value in self.details.items()}
            self.logger.save(name="details.pt", content=details, mode="tensor")

    def _write_details(self, ids, alignments, p_eos):
        """Writes raw details (alignments, p_eos) as a
        PyTorch file

        Arguments
        ---------
        ids : list
            The list of IDs, from the dataset
        alignments : torch.Tensor
            Attention alignments
        p_eos : torch.Tensor
            A (Batch x Length) tensor of end-of-sequence
            probabilities
        """
        details = {
            "ids": ids,
            "alignments": list(alignments),
            "p_eos": list(p_eos),
        }
        if self.details is None:
            self.details = details
        else:
            for key, value in self.details.items():
                value.extend(details[key])


class ContextError(Exception):
    """Thrown when the various write methods are called without a context"""

    def __init__(self):
        super(
            "TTSProgressReport must be used in a context (with report: report.write(...))"
        )
