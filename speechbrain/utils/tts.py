"""Utilities for Text-to-Speech systems

Authors
 * Artem Ploujnikov 2023
"""
import logging
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

    def write(
        self,
        ids,
        audio,
        length_pred,
        length,
        alignments=None,
        p_eos=None,
    ):
        """Reports TTS inferents results

        Arguments
        ---------
        ids : list
            the list of IDs, from the dataset
        audio : torch.Tensor
            a padded tensor of audio samples
        length_pred : torch.Tensor
            a tensor of predicted relative lengths
        length : torch.Tensor
            a tensor of ground truth relative lengths

        alignments : torch.Tensor, optional
            Attention alignments
        p_eos : torch.Tensor
            A (Batch x Length) tensor of end-of-sequence
            probabilities        
        """
        with self.logger:
            self.write_audio(ids, audio, length_pred)
            self.write_details(ids, alignments, p_eos)
            if plt is not None:
                if alignments is not None:
                    self.write_alignments(ids, alignments)
                if p_eos is not None:
                    self.write_eos(ids, p_eos, length)

    def write_audio(self, ids, audio, length):
        """Saves the raw audio

        Arguments
        ---------
        ids : list
            the list of IDs, from the dataset
        audio : torch.Tensor
            a padded tensor of audio samples
        length : torch.Tensor
            a tensor of relative lengths
        """
        length_abs = (length * audio.size(1)).int()
        for item_id, item_audio, item_length in zip(ids, audio, length_abs):
            item_audio_cut = item_audio[:item_length.item()]
            self.logger.save(
                name=f"{item_id}.wav",
                content=item_audio_cut.detach().cpu(),
                mode="audio",
                samplerate=self.sample_rate
            )

    def write_alignments(self, ids, alignments):
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
                    item_alignment.detach().cpu(),
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

    def write_eos(self, ids, p_eos, length):
        """Outputs a plot of end-of-sequence gate outputs

        Arguments
        ---------
        ids : list
            The list of IDs, from the dataset
        p_eos : torch.Tensor
            A (Batch x Length) tensor of end-of-sequence
            probabilities
        """
        p_eos, length = [x.detach().cpu() for x in [p_eos, length]]
        max_len = length.max().item()
        for item_id, item_length, item_p_eos in zip(ids, length, p_eos):
            fig, ax = plt.subplots(figsize=(8, 2))
            try:
                ax.set_title(f"{item_id} Gate")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Gate Output")
                ax.plot(item_p_eos)
                x = [item_length, item_length]
                y = [0., 1.]
                ax.plot(x, y, color="blue", marker="o", label="Ground Truth")
                x = [0., max_len]
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

    def write_details(self, ids, alignments, p_eos):
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
            "alignments": alignments,
            "p_eos": p_eos,
        }
        self.logger.save(
            name="details.pt",
            content=details,
            mode="tensor"
        )