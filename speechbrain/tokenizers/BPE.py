"""Library for Byte-pair-encoding (BPE) tokenization.

Authors
 * Abdelwahab Heba 2020
"""

import os.path
import torch
import logging
import csv
import sentencepiece as spm

logger = logging.getLogger(__name__)


class BPE:
    """
    BPE class call the SentencePiece unsupervised text tokenizer from Google.
    Ref: https://github.com/google/sentencepiece

    SetencePiece lib is an unsupervised text tokenizer and detokenizer.
    It implements subword units like Byte-pair-encoding (BPE),
    Unigram language model and char/word tokenizer.


    Example
    -------
    >>> import torch
    >>> dict_int2lab = {1: "HELLO", 2: "MORNING"}
    >>> model_dir = "tests/unittests/tokenizer_data/"
    >>> csv_train = "tests/unittests/tokenizer_data/dev-clean.csv"
    >>> csv_read = "wrd"
    >>> model_type = "bpe"
    >>> bpe = BPE(model_dir,2000, csv_train, csv_read, model_type)
    >>> batch_seq = torch.Tensor([[1, 2, 2, 1],[1, 2, 1, 0]])
    >>> batch_lens = torch.Tensor([1.0,0.75])
    >>> encoded_seq_ids, encoded_seq_pieces = bpe(
    >>>    batch_seq,
    >>>    batch_lens,
    >>>    dict_int2lab,
    >>>    task="encode",
    >>>    init_params=True)

    """

    def __init__(
        self,
        model_dir,
        vocab_size,
        csv_train=None,
        csv_read=None,
        model_type="unigram",
    ):
        if model_type not in ["unigram", "bpe", "char", "word"]:
            raise ValueError(
                "model_type must be one of : [unigram, bpe, char, word]"
            )
        if not os.path.isfile(os.path.abspath(csv_train)):
            if os.path.isfile(
                os.path.join(
                    model_dir, str(vocab_size) + "_" + model_type + ".model"
                )
            ):
                logger.info(
                    "Tokenizer is already trained. Training file is no needed"
                )
            else:
                raise ValueError(
                    csv_train
                    + " is not a file. please provide text file for training."
                )
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        if not isinstance(vocab_size, int):
            raise ValueError("vocab_size must be integer.")
        if not os.path.isdir(model_dir):
            raise ValueError(
                model_dir
                + "must be a directory.\n"
                + "The model files should be: "
                + os.path.join(model_dir, str(vocab_size) + ".{model,vocab}")
            )

        self.csv_train = csv_train
        self.csv_read = csv_read
        self.text_file = os.path.join(
            os.path.dirname(csv_train),
            os.path.splitext(os.path.basename(csv_train))[0] + ".txt",
        )
        self.prefix_model_file = os.path.join(
            model_dir, str(vocab_size) + "_" + model_type
        )
        self.vocab_size = vocab_size
        self.model_type = model_type

    def _csv2text(self):
        logger.info(
            "Extract " + self.csv_read + " sequences from:" + self.csv_train
        )
        csv_file = open(self.csv_train, "r")
        reader = csv.reader(csv_file)
        headers = next(reader, None)
        if self.csv_read not in headers:
            raise ValueError(self.csv_read + "must exist in:" + self.csv_train)
        index_label = headers.index(self.csv_read)
        text_file = open(self.text_file, "w+")
        for row in reader:
            text_file.write(row[index_label] + "\n")
        text_file.close()
        csv_file.close()
        logger.info("Text file created at: " + self.text_file)

    def _train_BPE(self):
        spm.SentencePieceTrainer.train(
            "--input="
            + self.text_file
            + " --model_prefix="
            + self.prefix_model_file
            + " --vocab_size="
            + str(self.vocab_size)
            + " --model_type="
            + self.model_type
            + " --bos_id=-1 --eos_id=-1"
        )

    def init_params(self):
        if not os.path.isfile(self.prefix_model_file + ".model"):
            logger.info("Train tokenizer with type:" + self.model_type)
            if not os.path.isfile(self.text_file):
                self._csv2text()
            self._train_BPE()
        else:
            logger.info("Tokenizer is already trained.")
        logger.info("==== Loading Tokenizer ===")
        logger.info("Tokenizer path: " + self.prefix_model_file + ".model")
        logger.info("Tokenizer vocab_size: " + str(self.vocab_size))
        logger.info("Tokenizer type: " + self.model_type)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.prefix_model_file + ".model")

    def __call__(
        self,
        batch,
        batch_lens=None,
        ind2lab=None,
        task="encode",
        init_params=False,
    ):
        if init_params:
            self.init_params()
        if task == "encode" and ind2lab is None:
            raise ValueError("Tokenizer encoder must have the ind2lab function")

        if task == "encode":
            # Convert list of words to bpe ids
            bpe = []
            max_bpe_len = 0
            batch_lens = (batch_lens * batch.shape[1]).int()
            for i, utt_seq in enumerate(batch):
                bpe_encode = self.sp.encode_as_ids(
                    " ".join(
                        [
                            ind2lab[int(index)]
                            for index in utt_seq[: batch_lens[i]]
                        ]
                    )
                )
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
