"""Force alignment using k2 for CTC models.

Author:
    * Zeyu Zhao 2023
"""
import torch
from typing import Dict, List, Optional, Union
import logging
import speechbrain as sb
import abc
import torchaudio
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

try:
    import k2
except ImportError:
    MSG = "Cannot import k2, so training and decoding with k2 will not work.\n"
    MSG += "Please refer to https://k2-fsa.github.io/k2/installation/from_wheels.html for installation.\n"
    MSG += "You may also find the precompiled wheels for your platform at https://download.pytorch.org/whl/torch_stable.html"
    raise ImportError(MSG)


class Aligner(abc.ABC):
    """
    Abstract class for aligner.
    """
    @abc.abstractmethod
    def encode_texts(self, texts: List[str]) -> List[List[int]]:
        """Encode texts to list of tokens.

        Arguments
        ---------
        texts : List[str], the texts to be encoded.

        Returns
        -------
        List[List[int]], the encoded texts.
        """
        pass

    @abc.abstractmethod
    def get_log_prob_and_targets(
        self,
        audio_files: List[str],
        transcripts: List[str],
    ) -> (torch.Tensor, torch.Tensor):
        """Align transcripts to input_speech.

        Arguments
        ---------
        audio_files: List[str], the input audio directory.
        transcripts: List[str], the input transcripts.

        Returns
        -------
        torch.Tensor: the log-probabilities over the tokens.
        torch.Tensor: the lengths of the log-probabilities.
        list: the encoded targets.
        """
        pass

    def align(
        self,
        log_prob: torch.Tensor,
        log_prob_len: torch.Tensor,
        targets: List[List[int]],
    ) -> List[List[int]]:
        """Align targets to log_probs.

        Arguments
        ---------
            log_prob: torch.Tensor
                A tensor of shape (N, T, C) containing the log-probabilities.
                Please make sure that index 0 of the C dimension corresponds
                to the blank symbol.
            log_prob_len: torch.Tensor
                A tensor of shape (N,) containing the lengths of the log_probs.
                This is needed because the log_probs may have been padded.
                All elements in this tensor must be integers and <= T, and
                in descending order.
            targets: list
                A list of list of integers containing the targets.
                Note that the targets should not contain the blank symbol.
                The blank symbol is assumed to be index 0 in log_prob.
        Returns
        -------
            List
            List of lists of integers containing the alignments.
        """
        # Basic checks.
        assert log_prob.ndim == 3
        assert log_prob_len.ndim == 1
        assert log_prob.shape[0] == log_prob_len.shape[0]
        assert isinstance(targets, list)
        assert isinstance(targets[0], list)
        assert log_prob.shape[0] == len(targets)

        N, T, C = log_prob.shape

        graph = k2.ctc_graph(targets)

        lattice = k2.get_lattice(
            log_prob=log_prob,
            log_prob_len=log_prob_len,
            decoding_graph=graph,
        )

        best_path = k2.shortest_path(lattice, use_double_scores=True)
        labels = best_path.labels

        alignments = []
        alignment = []
        for e in labels.tolist():
            if e == -1:
                alignments.append(alignment)
                alignment = []
            else:
                alignment.append(e)

        logger.info(f"Number of alignments: {len(alignments)}")
        logger.info(f"Targets: {targets}")
        logger.info(f"Alignments: {alignments}")
        return alignments


class CTCAligner(Aligner):
    """
    Aligner class for CTC models.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            tokenizer: sb.dataio.encoder.CTCTextEncoder,
    ):
        """
        Arguments
        ---------

        model : torch.nn.Module, the model applied for alignment.
        tokenizer : sb.dataio.encoder.CTCTextEncoder, the tokenizer used for
            encoding the text.
        """
        self.model = model
        self.tokenizer = tokenizer

        assert hasattr(
            self.model, "encode_batch"), "The model must have an encode_batch method."

    def encode_texts(self, texts: List[str]) -> List[List[int]]:
        """Encode texts to list of tokens.

        Arguments
        ---------
        texts : List[str], the texts to be encoded.

        Returns
        -------
        List[List[int]], the encoded texts.
        """
        encoded_texts = []
        for text in texts:
            chars = list(text)
            encoded_text = self.tokenizer.encode_sequence(chars)
            encoded_texts.append(encoded_text)
        return encoded_texts

    def get_log_prob_and_targets(
        self,
        audio_files: List[str],
        transcripts: List[str],
    ) -> (torch.Tensor, torch.Tensor):
        """Align transcripts to input_speech.

        Arguments
        ---------
        audio_files: List[str], the input audio directory.
        transcripts: List[str], the input transcripts.

        Returns
        -------
        torch.Tensor: the log-probabilities over the tokens.
        torch.Tensor: the lengths of the log-probabilities.
        """
        encoded_texts = self.encode_texts(transcripts)
        sigs = []
        lens = []
        for audio_file in audio_files:
            snt, fs = torchaudio.load(audio_file)
            sigs.append(snt.squeeze())
            lens.append(snt.shape[1])

        # sort by length descending but keep original order
        sigs, lens, encoded_texts = zip(
            *sorted(zip(sigs, lens, encoded_texts), key=lambda x: x[1], reverse=True))

        batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)

        lens = torch.Tensor(lens) / batch.shape[1]

        with torch.no_grad():
            log_probs = self.model.encode_batch(batch, lens)

        # convert lens to log-prob lens
        lens = (lens * log_probs.shape[1]).round().int().cpu()
        log_probs = log_probs.cpu()

        return log_probs, lens, list(encoded_texts)

    def align_batch(
        self,
        audio_files: List[str],
        transcripts: List[str],
    ) -> List[List[int]]:
        """Align targets to log_probs.

        Arguments
        ---------
        audio_files: List[str], the input audio directory.
        transcripts: List[str], the input transcripts.

        Returns
        -------
        List[List[int]], the alignments.
        """
        log_probs, log_prob_len, targets = self.get_log_prob_and_targets(
            audio_files, transcripts)
        return self.align(log_probs, log_prob_len, targets)


def align(
    log_prob: torch.Tensor,
    log_prob_len: torch.Tensor,
    targets: list,
) -> List[List[int]]:
    """Align targets to log_probs.

    Arguments
    ---------
        log_prob: torch.Tensor
            A tensor of shape (N, T, C) containing the log-probabilities.
            Please make sure that index 0 of the C dimension corresponds
            to the blank symbol.
        log_prob_len: torch.Tensor
            A tensor of shape (N,) containing the lengths of the log_probs.
            This is needed because the log_probs may have been padded.
            All elements in this tensor must be integers and <= T, and
            in descending order.
        targets: list
            A list of list of integers containing the targets.
            Note that the targets should not contain the blank symbol.
            The blank symbol is assumed to be index 0 in log_prob.
    Returns
    -------
        List
        List of lists of integers containing the alignments.
    """
    # Basic checks.
    assert log_prob.ndim == 3
    assert log_prob_len.ndim == 1
    assert log_prob.shape[0] == log_prob_len.shape[0]
    assert isinstance(targets, list)
    assert isinstance(targets[0], list)
    assert log_prob.shape[0] == len(targets)

    N, T, C = log_prob.shape

    graph = k2.ctc_graph(targets)

    lattice = k2.get_lattice(
        log_prob=log_prob,
        log_prob_len=log_prob_len,
        decoding_graph=graph,
    )

    best_path = k2.shortest_path(lattice, use_double_scores=True)
    labels = best_path.labels

    alignments = []
    alignment = []
    for e in labels.tolist():
        if e == -1:
            alignments.append(alignment)
            alignment = []
        else:
            alignment.append(e)

    return alignments


def simple_test():
    # test align function
    log_prob = torch.tensor([[[0.1, 0.6, 0.1, 0.1, 0.1],
                              [0.1, 0.1, 0.6, 0.1, 0.1],
                              [0.1, 0.1, 0.1, 0.6, 0.1],
                              [0.6, 0.1, 0.1, 0.1, 0.1],
                              [0.1, 0.1, 0.1, 0.1, 0.6],
                              [0.6, 0.1, 0.1, 0.1, 0.1]]])
    log_prob = torch.log(log_prob)
    log_prob_len = torch.tensor([6])
    targets = [[1, 2, 3, 4]]
    alignment = align(log_prob, log_prob_len, targets)
    print("Simple test alignment:", alignment)

    # test alignment with different lengths
    log_prob = torch.tensor([[[0.1, 0.6, 0.1, 0.1, 0.1],
                              [0.1, 0.1, 0.6, 0.1, 0.1],
                              [0.1, 0.1, 0.1, 0.6, 0.1],
                              [0.6, 0.1, 0.1, 0.1, 0.1],
                              [0.1, 0.1, 0.1, 0.1, 0.6],
                              [0.6, 0.1, 0.1, 0.1, 0.1]],
                             [[0.1, 0.6, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.6, 0.1, 0.1],
                                 [0.1, 0.1, 0.1, 0.6, 0.1],
                                 [0.6, 0.1, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.1, 0.1, 0.1]]])
    log_prob = torch.log(log_prob)
    log_prob_len = torch.tensor([6, 4])
    targets = [[1, 2, 3, 4], [1, 2, 3]]
    alignment = align(log_prob, log_prob_len, targets)
    print("Batch test alignment:", alignment)


def class_test():
    audio_files = [
        "/disk/scratch3/zzhao/data/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac",
        "/disk/scratch3/zzhao/data/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac",
        "/disk/scratch3/zzhao/data/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0002.flac",
        "/disk/scratch3/zzhao/data/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0003.flac",
    ]
    trans = [
        "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE",
        "STUFF IT INTO YOU HIS BELLY COUNSELLED HIM",
        "AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS",
        "HELLO BERTIE ANY GOOD IN YOUR MIND",
    ]
    from speechbrain.pretrained import EncoderASR

    asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="pretrained_models/asr-wav2vec2-librispeech")
    aligner = CTCAligner(asr_model, asr_model.tokenizer)
    alignments = aligner.align_batch(audio_files, trans)
    return alignments


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    class_test()
