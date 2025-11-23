"""Force alignment using k2 for CTC models.
This module provides an abstract class, Aligner, for force alignment using k2 for CTC models.
Besides, it also provides a concrete class, CTCAligner, for force alignment using k2
specifically for a pre-trained CTC model and a tokeniser (CTCTextEncoder).
Note that we must make sure that the blank symbol is index 0 in the tokeniser's vocabulary.

Users can simply mimic the usage of CTCAligner to implement their own aligner.
There are two methods in the Aligner class that users need to implement:
    1. encode_texts: encode texts (List[str]) to a list of lists of token indexes (List[List[int]]).
    2. get_log_prob_and_targets: get log-probabilities (torch.Tensor), its length (torch.Tensor) and targets (List[List[int]])
        from audio files and transcripts.

The align method is implemented in the Aligner class, so users do not need to implement it.
We support three different ways of conducting force alignment:
    1. One audio file and one transcript at a time.
    2. A batch of audio files and transcripts.
    3. A csv file containing the audio file paths and transcripts.
        In this case, the csv file should follow the standard speechbrain csv format with a header line as follows:
        ID, duration, wav, spk_id, wrd
at two different levels (tokens and words).

When token-level alignment is conducted, for one single audio file or a batch of audio files,
the aligning method will return a list of lists of integers,
where each integer represents the index of the token in the tokeniser's vocabulary.
For example, if the tokeniser's vocabulary is ['<blank>', '<unk>', 'a', 'b', 'c'],
then the returned list of lists of integers may look like [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]].
For an input of csv file, the aligning method will return a dictionary (Dict[str, List[int]]),
where the keys are the IDs of the audio files and the values are the list of token indexes.

When word-level alignment is conducted, for one single audio file or a batch of audio files,
the aligning method will return a list of lists of tuples,
where each tuple represents (start_frame (int, including), end_frame (int, including), word (str)).
For example, if the transcript is 'hello word', and there are 20 frames in the audio file,
then the returned list of lists of tuples may look like [[(3, 10, 'hello'), (11, 16, 'word')]].
For an input of csv file, the aligning method will return a pandas.DataFrame,
where the columns are ['ID', 'word', 'start', 'end'], and note that the start and end are in seconds.
However, if the frame_shift for the method, align_csv_word, is None, then the start and end will be in frames.

Author:
    * Zeyu Zhao 2024
"""

import abc
import logging
from typing import List, Tuple

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import speechbrain as sb
from speechbrain.dataio import audio_io

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

    To implement your own aligner, you need to implement two methods:
        1. encode_texts: encode texts (List[str]) to a list of lists of token indexes (List[List[int]]).
        2. get_log_prob_and_targets: get log-probabilities (torch.Tensor), its length (torch.Tensor) and targets (List[List[int]])

    The align method is implemented in the Aligner class, so users do not need to implement it.
    We support three different ways of conducting force alignment:
        1. One audio file and one transcript at a time.
        2. A batch of audio files and transcripts.
        3. A csv file containing the audio file paths and transcripts.

    When token-level alignment is conducted, for one single audio file,
    the aligning method will return a list of integers,
    where each integer represents the index of the token in the tokeniser's vocabulary.
    For example, if the tokeniser's vocabulary is ['<blank>', '<unk>', 'a', 'b', 'c'],
    then the returned list of integers may look like [0, 1, 2, 3, 4].

    For a batch of audio files, the aligning method will return a list of lists of integers,
    where each integer represents the index of the token in the tokeniser's vocabulary.
    For example, if the tokeniser's vocabulary is ['<blank>', '<unk>', 'a', 'b', 'c'],
    then the returned list of lists of integers may look like [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]].

    For an input of csv file, the aligning method will return a dictionary (Dict[str, List[int]]),
    where the keys are the IDs of the audio files and the values are the list of token indexes.

    When word-level alignment is conducted, for one single audio file,
    the aligning method will return a list of tuples,
    where each tuple represents (start_frame (int, including), end_frame (int, including), word (str)).
    For example, if the transcript is 'hello word', and there are 20 frames in the audio file,
    then the returned list of tuples may look like [(3, 10, 'hello'), (11, 16, 'word')].
    If the frame_shift for the method, align_csv_word, is None, then the start and end will be in frames.
    If the frame_shift for the method, align_csv_word, is not None, then the start and end will be in seconds.

    For a batch of audio files, the aligning method will return a list of lists of tuples,
    where each tuple represents (start_frame (int, including), end_frame (int, including), word (str)).
    For example, if the transcript is ['hello world', 'hello speechbrain'], and there are 20 frames in each audio file,
    then the returned list of lists of tuples may look like [[(3, 10, 'hello'), (11, 16, 'world')], [(3, 10, 'hello'), (11, 20, 'speechbrain')]].

    For an input of csv file, the aligning method will return nothing but save the alignment results to a csv file.
    The columns of the csv file are ['ID', 'word', 'start', 'end'], and note that the start and end are in seconds,
    if the frame_shift is not None, else the start and end will be in frames.
    """

    @abc.abstractmethod
    def encode_texts(self, texts: List[str]) -> List[List[int]]:
        """
        Encode texts to list of tokens.

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
        """
        Align transcripts to input_speech.

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
        """
        Align targets to log_probs.

        Arguments
        ---------
        log_prob: torch.Tensor
            A tensor of shape (N, T, C) containing the log-probabilities.
            Please make sure that index 0 of the C dimension corresponds
            to the blank symbol.
        log_prob_len: torch.Tensor
            A tensor of shape (N,) containing the lengths of the log_probs.
            This is needed because the log_probs may have been padded.
            All elements in this tensor must be integers and <= T.
        targets: list
            A list of list of integers containing the targets.
            Note that the targets should not contain the blank symbol.
            The blank symbol is assumed to be index 0 in log_prob.
        Returns
        -------
        alignments: List[List[int]], containing the alignments.
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

    def align_batch(
        self,
        audio_files: List[str],
        transcripts: List[str],
    ) -> List[List[int]]:
        """
        Align targets to log_probs.

        Arguments
        ---------
        audio_files: List[str], the input audio directory.
        transcripts: List[str], the input transcripts.

        Returns
        -------
        List[List[int]], the alignments.
        """
        log_probs, log_prob_len, targets = self.get_log_prob_and_targets(
            audio_files, transcripts
        )
        return self.align(log_probs, log_prob_len, targets)

    def get_word_alignment(
        self,
        alignments: List[List[int]],
        transcripts: List[str],
    ) -> List[List[Tuple[int, int, str]]]:
        """
        Get word alignment from character alignment.

        Arguments
        ---------
        alignments: List[List[int]], the character alignments.
        transcripts: List[str], the input transcripts.

        Returns
        -------
        List[List[Tuple[int, int, str]]], the word alignments.
        Each tuple contains the start (include) and end (include) frame index of the word, and the word itself.
        """
        word_alignments = []
        for alignment, transcript in zip(alignments, transcripts):
            words = transcript.split()
            word_alignment = []
            align_pointer = 0
            for word in words:
                found = False
                last_found = False
                word_pointer = 0
                word_start = 0
                word_end = 0
                char_ids = self.encode_texts([word])[0]
                while word_pointer <= len(char_ids):
                    if (
                        not found
                        and alignment[align_pointer] == char_ids[word_pointer]
                    ):
                        found = True
                        word_pointer += 1
                        word_start = align_pointer
                        if word_pointer == len(char_ids):
                            last_found = True
                            word_end = align_pointer
                    elif last_found:
                        if (
                            alignment[align_pointer]
                            == char_ids[word_pointer - 1]
                        ):
                            word_end = align_pointer
                        else:
                            break
                    elif found:
                        if alignment[align_pointer] == char_ids[word_pointer]:
                            word_pointer += 1
                            if word_pointer == len(char_ids):
                                last_found = True
                                word_end = align_pointer
                    align_pointer += 1
                word_alignment.append((word_start, word_end, word))
            word_alignments.append(word_alignment)
        return word_alignments

    def align_audio_to_tokens(
        self,
        audio_file: str,
        transcript: str,
    ) -> List[int]:
        """
        Align audio to tokens.

        Arguments
        ---------
        audio_file: str, the input audio file path.
        transcript: str, the input transcript.

        Returns
        -------
        alignment: List[int], the token-level alignments for the audio file.
            Note that the length of the alignments is the same as the number of frames in the audio file,
            i.e., the length of the output of the NN model.
        """
        audio_files = [audio_file]
        transcripts = [transcript]
        log_probs, log_prob_len, targets = self.get_log_prob_and_targets(
            audio_files, transcripts
        )
        alignments = self.align(log_probs, log_prob_len, targets)
        if not alignments:
            logger.warn(f"No alignment found for {audio_file}")
            return []
        else:
            return alignments[0]

    def align_audio_to_words(
        self,
        audio_file: str,
        transcript: str,
        frame_shift: float = 0.02,
    ) -> List[Tuple[int, int, str]]:
        """
        Align audio to words.

        Arguments
        ---------
        audio_file: str, the input audio file path.
        transcript: str, the input transcript.
        frame_shift: float, the frame shift in seconds, default to 0.02.

        Returns
        -------
        alignment: List[Tuple[int, int, str]], the word-level alignments for the audio file.
            Each tuple contains the start (include) and end (include) frame index of the word, and the word itself.
        """
        audio_files = [audio_file]
        transcripts = [transcript]
        log_probs, log_prob_len, targets = self.get_log_prob_and_targets(
            audio_files, transcripts
        )
        alignments = self.align(log_probs, log_prob_len, targets)
        word_alignments = self.get_word_alignment(alignments, transcripts)

        if frame_shift > 0:
            for word_alignment in word_alignments:
                for i, (start, end, word) in enumerate(word_alignment):
                    word_alignment[i] = (
                        (start * frame_shift),
                        (end * frame_shift),
                        word,
                    )

        if not word_alignments:
            logger.warn(f"No alignment found for {audio_file}")
            return []
        else:
            return word_alignments[0]

    def align_batch_to_tokens(
        self,
        audio_files: List[str],
        transcripts: List[str],
    ) -> List[List[int]]:
        """
        Align a batch of audio files to tokens.

        Arguments
        ---------
        audio_files: List[str], the input audio files.
        transcripts: List[str], the input transcripts.

        Returns
        -------
        alignments: List[List[int]], the token-level alignments for the audio files.
            Note that the length of the alignments is the same as the number of frames in the audio file,
            i.e., the length of the output of the NN model.
        """
        log_probs, log_prob_len, targets = self.get_log_prob_and_targets(
            audio_files, transcripts
        )
        alignments = self.align(log_probs, log_prob_len, targets)
        return alignments

    def align_batch_to_words(
        self,
        audio_files: List[str],
        transcripts: List[str],
        frame_shift: float = 0.02,
    ) -> List[List[Tuple[int, int, str]]]:
        """
        Align a batch of audio files to words.

        Arguments
        ---------
        audio_files: List[str], the input audio files.
        transcripts: List[str], the input transcripts.
        frame_shift: float, the frame shift in seconds, default to 0.02.

        Returns
        -------
        alignments: List[List[Tuple[int, int, str]]], the word-level alignments for the audio files.
            Each tuple contains the start (include) and end (include) frame index of the word, and the word itself.

        Note that, the batch size should be small enough to fit into the GPU memory.
        """
        log_probs, log_prob_len, targets = self.get_log_prob_and_targets(
            audio_files, transcripts
        )
        alignments = self.align(log_probs, log_prob_len, targets)
        word_alignments = self.get_word_alignment(alignments, transcripts)

        if frame_shift > 0:
            for i, word_alignment in enumerate(word_alignments):
                for j, (start, end, word) in enumerate(word_alignment):
                    word_alignments[i][j] = (
                        (start * frame_shift),
                        (end * frame_shift),
                        word,
                    )
        return word_alignments

    def align_csv_to_tokens(
        self,
        input_csv: str,
        output_file: str,
        batch_size: int = 4,
    ):
        """
        Align all the audio files in the input_csv and write the token alignments to output_csv.
        The output file will have the format:
        <audio id> <token alignment>

        Arguments
        ---------
        input_csv: str, the input csv file.
        output_file: str, the output file.
        batch_size: int, the batch size, default 4.
        """
        df = pd.read_csv(input_csv)
        audio_files = df["wav"].tolist()
        transcripts = df["wrd"].tolist()
        ids = df["ID"].tolist()

        fc = ""
        with open(output_file, "w", encoding="utf-8") as f:
            for i in range(0, len(audio_files), batch_size):
                batch_audio_files = audio_files[
                    i : min(i + batch_size, len(audio_files))
                ]
                batch_transcripts = transcripts[
                    i : min(i + batch_size, len(audio_files))
                ]
                batch_ids = ids[i : min(i + batch_size, len(audio_files))]
                alignments = self.align_batch_to_tokens(
                    batch_audio_files, batch_transcripts
                )
                for audio_id, alignment in zip(batch_ids, alignments):
                    fc += (
                        audio_id
                        + " "
                        + " ".join([str(a) for a in alignment])
                        + "\n"
                    )
            f.write(fc)

    def align_csv_to_words(
        self,
        input_csv: str,
        output_csv: str,
        batch_size: int = 4,
        frame_shift: float = 0.02,
    ):
        """
        Align all the audio files in the input_csv and write the word alignments to output_csv.
        The output file will have the format:
        <audio id> <word> <start> <end>

        Arguments
        ---------
        input_csv: str, the input csv file.
        output_csv: str, the output csv file.
        batch_size: int, the batch size, default 4.
        frame_shift: float, the frame shift in seconds at the output end of the NN model, default 0.02.
        """
        df = pd.read_csv(input_csv)
        audio_files = df["wav"].tolist()
        transcripts = df["wrd"].tolist()
        ids = df["ID"].tolist()

        if frame_shift is None or frame_shift == 1:
            logger.info("No frame shift is provided or the frame shift is 1.")
            logger.info("The resulting alignment will be in frame index.")
            logger.info("The frame index starts from 0.")
            frame_shift = 1

        alignment = {"ID": [], "word": [], "start": [], "end": []}
        for i in tqdm(range(0, len(audio_files), batch_size)):
            batch_audio_files = audio_files[
                i : min(i + batch_size, len(audio_files))
            ]
            batch_transcripts = transcripts[
                i : min(i + batch_size, len(audio_files))
            ]
            batch_ids = ids[i : min(i + batch_size, len(audio_files))]
            batch_alignments = self.align_batch(
                batch_audio_files, batch_transcripts
            )
            batch_word_alignments = self.get_word_alignment(
                batch_alignments, batch_transcripts
            )
            for batch_id, batch_word_alignment in zip(
                batch_ids, batch_word_alignments
            ):
                for word_start, word_end, word in batch_word_alignment:
                    alignment["ID"].append(batch_id)
                    alignment["word"].append(word)
                    alignment["start"].append(word_start * frame_shift)
                    alignment["end"].append(word_end * frame_shift)
        if frame_shift != 1:
            logger.info("The frame shift is %f seconds.", frame_shift)
            logger.info("The resulting alignment will be in seconds.")
            pd.DataFrame(alignment).round(3).to_csv(output_csv, index=False)
        else:
            pd.DataFrame(alignment).to_csv(output_csv, index=False)


class CTCAligner(Aligner):
    """
    Aligner class for CTC models.
    There are six methods designed to be applied by users directly:
        * align_audio_to_tokens
        * align_audio_to_words
        * align_batch_to_tokens
        * align_batch_to_words
        * align_csv_to_tokens
        * align_csv_to_words
    For more details, please refer to the documentation of each method.

    Arguments
    ---------
    model : torch.nn.Module, the model applied for alignment.
    tokenizer : sb.dataio.encoder.CTCTextEncoder, the tokenizer used for
        encoding the text.
    device : torch.device, the device to run the model on, default torch.device("cpu").

    Example
    -------
    >>> import torch
    >>> from speechbrain.inference import EncoderASR
    >>> from speechbrain.integrations.k2_fsa.align import CTCAligner
    >>> asr_model = EncoderASR.from_hparams(
    ...     source="speechbrain/asr-wav2vec2-librispeech",
    ...     savedir="pretrained_models/asr-wav2vec2-librispeech",
    ... )
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> aligner = CTCAligner(
    ...     model=asr_model, tokenizer=asr_model.tokenizer, device=device
    ... )
    >>> audio_files = ["tests/samples/ASR/spk1_snt1.wav"]
    >>> transcripts = ["THE CHILD ALMOST HURT THE SMALL DOG"]
    >>> # align one audio file to tokens
    >>> # alignment = aligner.align_audio_to_tokens(audio_files[0], transcripts[0])
    >>> # align one audio file to words
    >>> alignment = aligner.align_audio_to_words(
    ...     audio_files[0], transcripts[0], frame_shift=0.02
    ... )
    >>> alignment
    [(0.04, 0.1, 'THE'), (0.26, 0.6, 'CHILD'), (0.84, 1.18, 'ALMOST'), (1.380..., 1.58, 'HURT'), (1.84, 1.880..., 'THE'), (2.04, 2.32, 'SMALL'), (2.46, 2.72, 'DOG')]
    >>> # align a batch of audio files to tokens
    >>> # alignments = aligner.align_batch_to_tokens(audio_files, transcripts)
    >>> # align a batch of audio files to words
    >>> # alignments = aligner.align_batch_to_words(audio_files, transcripts, frame_shift=0.02)
    >>> # align a csv file to tokens
    >>> # aligner.align_csv_to_tokens("samples/audio_samples/example.csv", "samples/audio_samples/example_token_alignment.txt")
    >>> # align a csv file to words
    >>> # aligner.align_csv_to_words("samples/audio_samples/example.csv", "samples/audio_samples/example_word_alignment.csv", frame_shift=0.02)

    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: sb.dataio.encoder.CTCTextEncoder,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model = self.model.to(self.device)
        self.model.device = self.device

    def encode_texts(self, texts: List[str]) -> List[List[int]]:
        """
        Encode texts to list of tokens.

        Arguments
        ---------
        texts : List[str], the texts to be encoded.

        Returns
        -------
        List[List[int]], the encoded texts.

        Note
        ----
        This method is specific to the tokeniser used in the model.
        In this case, we use the CTCTextEncoder.
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
        """
        Align transcripts to input_speech.

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

        assert hasattr(self.model, "encode_batch"), (
            "The model must have an encode_batch method."
        )

        encoded_texts = self.encode_texts(transcripts)
        sigs = []
        lens = []
        for audio_file in audio_files:
            snt, fs = audio_io.load(audio_file)
            sigs.append(snt.squeeze())
            lens.append(snt.shape[1])

        batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)
        lens = torch.Tensor(lens) / batch.shape[1]

        with torch.no_grad():
            batch = batch.to(self.device)
            lens = lens.to(self.device)
            log_probs = self.model.encode_batch(batch, lens)

        # convert lens to log-prob lens
        lens = (lens * log_probs.shape[1]).round().int().cpu()
        log_probs = log_probs.cpu()

        return log_probs, lens, list(encoded_texts)
