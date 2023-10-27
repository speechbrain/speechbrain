"""Force alignment using k2 for CTC models.

Author:
    * Zeyu Zhao 2023
"""
import torch
from typing import List, Tuple
import logging
import speechbrain as sb
import abc
import torchaudio
import pandas as pd
from tqdm import tqdm
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
                All elements in this tensor must be integers and <= T.
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

    def get_word_alignment(
            self,
            alignments: List[List[int]],
            transcripts : List[str],
    ) -> List[List[Tuple[int, int, str]]]:
        """Get word alignment from character alignment.

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
                    if not found and alignment[align_pointer] == char_ids[word_pointer]:
                        found = True
                        word_pointer += 1
                        word_start = align_pointer
                        if word_pointer == len(char_ids):
                            last_found = True
                            word_end = align_pointer
                    elif last_found:
                        if alignment[align_pointer] == char_ids[word_pointer - 1]:
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

    def align_csv_word(
            self,
            input_csv: str,
            output_csv: str,
            batch_size: int = 4,
            frame_shift: float = 0.02,
    ):
        """Align all the audio files in the input_csv and write the word alignments to output_csv.

        Arguments
        ---------
        input_csv: str, the input csv file.
        output_csv: str, the output csv file.
        batch_size: int, the batch size, default 4.
        frame_shift: float, the frame shift in seconds at the output end of the NN model, default 0.02.

        Outputs
        -------
        A csv file with the following columns:
        ID: str, the ID of the utterance.
        word: str, the word.
        start: float, the start time of the word in seconds.
        end: float, the end time of the word in seconds.
        """
        df = pd.read_csv(input_csv)
        audio_files = df["wav"].tolist()
        transcripts = df["wrd"].tolist()
        ids = df["ID"].tolist()

        alignment = {"ID": [], "word": [], "start": [], "end": []}
        for i in tqdm(range(0, len(audio_files), batch_size)):
            batch_audio_files = audio_files[i:min(i + batch_size, len(audio_files))]
            batch_transcripts = transcripts[i:min(i + batch_size, len(audio_files))]
            batch_ids = ids[i:min(i + batch_size, len(audio_files))]
            batch_alignments = self.align_batch(batch_audio_files, batch_transcripts)
            batch_word_alignments = self.get_word_alignment(batch_alignments, batch_transcripts)
            for batch_id, batch_word_alignment in zip(batch_ids, batch_word_alignments):
                for word_start, word_end, word in batch_word_alignment:
                    alignment["ID"].append(batch_id)
                    alignment["word"].append(word)
                    alignment["start"].append(word_start * frame_shift)
                    alignment["end"].append(word_end * frame_shift)
        pd.DataFrame(alignment).round(3).to_csv(output_csv, index=False)


class CTCAligner(Aligner):
    """
    Aligner class for CTC models.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            tokenizer: sb.dataio.encoder.CTCTextEncoder,
            device: torch.device = torch.device("cpu"),
    ):
        """
        Arguments
        ---------

        model : torch.nn.Module, the model applied for alignment.
        tokenizer : sb.dataio.encoder.CTCTextEncoder, the tokenizer used for
            encoding the text.
        device : torch.device, the device to run the model on, default torch.device("cpu").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model = self.model.to(self.device)
        self.model.device = self.device

    def encode_texts(self, texts: List[str]) -> List[List[int]]:
        """Encode texts to list of tokens.

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

        assert hasattr(
            self.model, "encode_batch"), "The model must have an encode_batch method."

        encoded_texts = self.encode_texts(transcripts)
        sigs = []
        lens = []
        for audio_file in audio_files:
            snt, fs = torchaudio.load(audio_file)
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

    input_csv = "/disk/scratch3/zzhao/speechbrain/recipes/LibriSpeech/ASR/CTC/results/train_wav2vec2_char_k2/1368/test-clean.csv"
    output_csv = "/disk/scratch3/zzhao/speechbrain/recipes/LibriSpeech/ASR/CTC/results/train_wav2vec2_char_k2/1368/test-clean-align.csv"

    from speechbrain.pretrained import EncoderASR

    asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="pretrained_models/asr-wav2vec2-librispeech")
    # device = torch.device("cuda")
    # asr_model.to(device)

    aligner = CTCAligner(asr_model, asr_model.tokenizer, device=torch.device("cuda"))
    aligner.align_csv_word(input_csv, output_csv)
    # log_prob, log_prob_len, targets = aligner.get_log_prob_and_targets(audio_files, trans)
    # alignments = aligner.align(log_prob, log_prob_len, targets)
    # word_alignments = aligner.get_word_alignment(alignments, trans)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    class_test()
