"""Library for Byte-pair-encoding (BPE) tokenization.

Authors
 * Abdelwahab Heba 2020
 * Loren Lugosch 2020
"""

import os.path
import torch
import logging
import csv
import sentencepiece as spm
from speechbrain.data_io.data_io import merge_char
from speechbrain.utils import edit_distance
import speechbrain as sb

logger = logging.getLogger(__name__)


class SentencePiece:
    """
    BPE class call the SentencePiece unsupervised text tokenizer from Google.
    Ref: https://github.com/google/sentencepiece

    SetencePiece lib is an unsupervised text tokenizer and detokenizer.
    It implements subword units like Byte-pair-encoding (BPE),
    Unigram language model and char/word tokenizer.

    Arguments
    ---------
    model_dir: str
        The directory where the model is saved.
    vocab_size: int, None, optional
        Vocab size for the choosen tokenizer type (BPE, Unigram).
        The vocab_size is optional for char, and mandatory for BPE & unigram
        tokenization.
    csv_train: str
        Path of the csv file which is used for learn of create the tokenizer.
    csv_read: str
        The data entry which contain the word sequence in the csv file.
    model_type: str
        (bpe, char, unigram).
        If "bpe", train unsupervised tokenization of piece of words. see:
        https://www.aclweb.org/anthology/P16-1162/
        If "word" take the vocabulary from the input text.
        If "unigram" do piece of word tokenization using unigram language
        model, see: https://arxiv.org/abs/1804.10959
    char_format_input : bool
        Default : False
        Whether the csv_read entry contains characters format input.
        (ex. a p p l e _ i s _ g o o d)
    character_coverage: int
        Default: 1.0, Amount of characters covered by the model, good defaults
        are: 0.9995 for languages with rich character set like Japanse or
        Chinese and 1.0 for other languages with small character set.
    user_defined_symbols: string
        Default: None,
        String contained a list of symbols separated by comma.
        User defined symbols are handled as one piece in any context.
    max_sentencepiece_length: int
        Deault: 10,
        Maximum number of characters for the tokens.
    bos_id: int
        Default: -1, if -1 the bos_id = unk_id = 0. otherwise, bos_id = int.
    eos_id: int
        Default: -1, if -1 the bos_id = unk_id = 0. otherwise, bos_id = int.
    split_by_whitespace: bool,
        Default: True,
        If False, allow the sentenciepiece to extract piece crossing multiple words.
        This feature is important for : Chinese/Japenese/Korean.
    num_sequences: int
        Default: None
        If not none, use at most this many sequences to train the tokenizer (for large datasets).
    csv_list_to_check: list,
        List of the csv file which is used for checking the accuracy of recovering words from the tokenizer.
    Example
    -------
    >>> import torch
    >>> dict_int2lab = {1: "HELLO", 2: "MORNING"}
    >>> model_dir = "tests/unittests/tokenizer_data/"
    >>> csv_train = "tests/unittests/tokenizer_data/dev-clean.csv"
    >>> csv_read = "wrd"
    >>> model_type = "bpe"
    >>> bpe = SentencePiece(model_dir,2000, csv_train, csv_read, model_type)
    >>> batch_seq = torch.Tensor([[1, 2, 2, 1],[1, 2, 1, 0]])
    >>> batch_lens = torch.Tensor([1.0, 0.75])
    >>> encoded_seq_ids, encoded_seq_pieces = bpe(
    ...     batch_seq, batch_lens, dict_int2lab, task="encode"
    ... )
    """

    def __init__(
        self,
        model_dir,
        vocab_size,
        csv_train=None,
        csv_read=None,
        model_type="unigram",
        char_format_input=False,
        character_coverage=1.0,
        user_defined_symbols=None,
        max_sentencepiece_length=10,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        split_by_whitespace=True,
        num_sequences=None,
        csv_list_to_check=None,
    ):
        if model_type not in ["unigram", "bpe", "char"]:
            raise ValueError("model_type must be one of : [unigram, bpe, char]")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        if not isinstance(vocab_size, int):
            raise ValueError("vocab_size must be integer.")

        self.csv_train = csv_train
        self.csv_read = csv_read
        if self.csv_train is not None:
            self.text_file = self.csv_train.replace(".csv", ".txt")

        self.prefix_model_file = os.path.join(
            model_dir, str(vocab_size) + "_" + model_type
        )
        self.vocab_size = str(vocab_size)
        self.model_type = model_type
        self.char_format_input = char_format_input
        self.character_coverage = str(character_coverage)
        self.max_sentencepiece_length = str(max_sentencepiece_length)
        self.bos_id = str(bos_id)
        self.eos_id = str(eos_id)
        self.pad_id = str(pad_id)
        self.unk_id = str(unk_id)
        self.num_sequences = num_sequences
        self.split_by_whitespace = split_by_whitespace
        self.user_defined_symbols = user_defined_symbols

        if not os.path.isfile(self.prefix_model_file + ".model"):
            logger.info("Train tokenizer with type:" + self.model_type)
            if not os.path.isfile(self.text_file):
                try:
                    if sb.utils.distributed.if_main_process():
                        self._csv2text()
                finally:
                    sb.utils.distributed.ddp_barrier()
            try:
                if sb.utils.distributed.if_main_process():
                    self._train_BPE()
            finally:
                sb.utils.distributed.ddp_barrier()
        else:
            logger.info("Tokenizer is already trained.")
        logger.info("==== Loading Tokenizer ===")
        logger.info("Tokenizer path: " + self.prefix_model_file + ".model")
        logger.info("Tokenizer vocab_size: " + str(self.vocab_size))
        logger.info("Tokenizer type: " + self.model_type)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.prefix_model_file + ".model")
        try:
            if sb.if_main_process():
                if csv_list_to_check is not None:
                    self._check_coverage_from_bpe(csv_list_to_check)
        finally:
            sb.ddp_barrier()

    def _csv2text(self):
        """
        Read CSV file and convert specific data entries into text file.
        """
        if not os.path.isfile(os.path.abspath(self.csv_train)):
            raise ValueError(
                self.csv_train
                + " is not a file. please provide csv file for training."
            )
        logger.info(
            "Extract " + self.csv_read + " sequences from:" + self.csv_train
        )
        csv_file = open(self.csv_train, "r")
        reader = csv.reader(csv_file)
        headers = next(reader, None)
        if self.csv_read not in headers:
            raise ValueError(self.csv_read + " must exist in:" + self.csv_train)
        index_label = headers.index(self.csv_read)
        text_file = open(self.text_file, "w+")
        row_idx = 0
        for row in reader:
            if self.num_sequences is not None and row_idx > self.num_sequences:
                print(
                    "Using %d sequences to train the tokenizer."
                    % self.num_sequences
                )
                break
            row_idx += 1
            sent = row[index_label]
            if self.char_format_input:
                (sent,) = merge_char([sent.split()])
                sent = " ".join(sent)
            text_file.write(sent + "\n")
        text_file.close()
        csv_file.close()
        logger.info("Text file created at: " + self.text_file)

    def _train_BPE(self):
        """
        Train tokenizer with unsupervised techniques (BPE, Unigram) using
        SentencePiece Library. If you use "char" mode, the SentencePiece
        creates a char dict so the vocab_size attribute is not needed.
        """
        query = (
            "--input="
            + self.text_file
            + " --model_prefix="
            + self.prefix_model_file
            + " --model_type="
            + self.model_type
            + " --bos_id="
            + self.bos_id
            + " --eos_id="
            + self.eos_id
            + " --pad_id="
            + self.pad_id
            + " --unk_id="
            + self.unk_id
            + " --max_sentencepiece_length="
            + self.max_sentencepiece_length
            + " --character_coverage="
            + self.character_coverage
        )
        if self.model_type not in ["char"]:
            # include vocab_size
            query += " --vocab_size=" + str(self.vocab_size)
        if self.user_defined_symbols is not None:
            query += " --user_defined_symbols=" + self.user_defined_symbols
        if not self.split_by_whitespace:
            query += " --split_by_whitespace=false"
        # Train tokenizer
        spm.SentencePieceTrainer.train(query)

    def _check_coverage_from_bpe(self, list_csv_files=[]):
        """
        Logging the accuracy of the BPE model to recover words from the training text.

        Arguments
        ---------
        csv_list_to_check: list,
            List of the csv file which is used for checking the accuracy of recovering words from the tokenizer.
        """
        for csv_file in list_csv_files:
            if os.path.isfile(os.path.abspath(csv_file)):
                logger.info(
                    "==== Accuracy checking for recovering text from tokenizer ==="
                )
                fcsv_file = open(csv_file, "r")
                reader = csv.reader(fcsv_file)
                headers = next(reader, None)
                if self.csv_read not in headers:
                    raise ValueError(
                        self.csv_read + " must exist in:" + csv_file
                    )
                index_label = headers.index(self.csv_read)
                wrong_recover_list = []
                for row in reader:
                    row = row[index_label]
                    if self.char_format_input:
                        (row,) = merge_char([row.split()])
                        row = " ".join(row)
                    row = row.split("\n")[0]
                    encoded_id = self.sp.encode_as_ids(row)
                    decode_text = self.sp.decode_ids(encoded_id)
                    (details,) = edit_distance.wer_details_for_batch(
                        ["utt1"],
                        [row.split(" ")],
                        [decode_text.split(" ")],
                        compute_alignments=True,
                    )
                    if details["WER"] > 0:
                        for align in details["alignment"]:
                            if align[0] != "=" and align[1] is not None:
                                if align[1] not in wrong_recover_list:
                                    wrong_recover_list.append(align[1])
                fcsv_file.close()
                logger.info("recover words from: " + csv_file)
                if len(wrong_recover_list) > 0:
                    logger.warn(
                        "Wrong recover words: " + str(len(wrong_recover_list))
                    )
                    logger.warn(
                        "Tokenizer vocab size: " + str(self.sp.vocab_size())
                    )
                    logger.warn(
                        "accuracy recovering words: "
                        + str(
                            1
                            - float(len(wrong_recover_list))
                            / self.sp.vocab_size()
                        )
                    )
                else:
                    logger.info("Wrong recover words: 0")
                    logger.warning("accuracy recovering words: " + str(1.0))
            else:
                logger.info("No accuracy recover checking for" + csv_file)

    def __call__(
        self, batch, batch_lens=None, ind2lab=None, task="encode",
    ):
        """
        This __call__ function implements the tokenizer encoder and decoder
        (restoring the string of word) for BPE, Regularized BPE (with unigram),
        and char (speechbrain/nnet/RNN.py).

        Arguments
        ----------
        batch : tensor.IntTensor or list
            list if ( batch_lens = None and task = "decode_from_list")
            Contains the original labels. Shape: [batch_size, max_length]
        batch_lens : tensor.LongTensor
            Default: None,
            Cotaining the relative length of each label sequences. Must be 1D
            tensor of shape: [batch_size].
        ind2lab : dict
            Dictionnary which map the index from label sequences
            (batch tensor) to string label.
        task: str
            ("encode", "decode", "decode_from_list)
            "encode": convert the batch tensor into sequence of tokens.
                the output contain a list of (tokens_seq, tokens_lens)
            "decode": convert a tensor of tokens to a list of word sequences.
            "decode_from_list": convert a list of token sequences to a list
                of word sequences.
        """
        if task == "encode" and ind2lab is None:
            raise ValueError("Tokenizer encoder must have the ind2lab function")

        if task == "encode":
            # Convert list of words/chars to bpe ids
            bpe = []
            max_bpe_len = 0
            batch_lens = (batch_lens * batch.shape[1]).int()
            for i, utt_seq in enumerate(batch):
                tokens = [
                    ind2lab[int(index)] for index in utt_seq[: batch_lens[i]]
                ]
                if self.char_format_input:
                    (words_list,) = merge_char([tokens])
                    sent = " ".join(words_list)
                else:
                    sent = " ".join(tokens)
                bpe_encode = self.sp.encode_as_ids(sent)
                bpe.append(bpe_encode)
                # save the longest bpe sequence
                # it help to compute the relative length of each utterance
                if len(bpe_encode) > max_bpe_len:
                    max_bpe_len = len(bpe_encode)
            # Create bpe tensor
            bpe_tensor = torch.zeros(
                (batch.shape[0], max_bpe_len), device=batch.device
            )
            bpe_lens = torch.zeros((batch.shape[0]), device=batch.device)
            for i, bpe_utt in enumerate(bpe):
                bpe_tensor[i, : len(bpe_utt)] = torch.Tensor(bpe_utt)
                bpe_lens[i] = len(bpe_utt) / max_bpe_len
            return bpe_tensor, bpe_lens
        elif task == "decode_from_list":
            # From list of hyps (not padded outputs)
            # do decoding
            return [self.sp.decode_ids(utt_seq).split(" ") for utt_seq in batch]
        elif task == "decode":
            # From a batch tensor and a length tensor
            # find the absolute batch lengths and do decoding
            batch_lens = (batch_lens * batch.shape[1]).int()
            return [
                self.sp.decode_ids(
                    utt_seq[: batch_lens[i]].int().tolist()
                ).split(" ")
                for i, utt_seq in enumerate(batch)
            ]
