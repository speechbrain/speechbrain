"""Library for Byte-pair-encoding (BPE) tokenization.

Authors
 * Abdelwahab Heba 2020
"""

import os.path
import torch
import logging
import sentencepiece as spm

logger = logging.getLogger(__name__)


class BPE:
    """ BPE
    Update later
    """

    def __init__(
        self, model_dir, vocab_size, text_file=None, model_type="unigram"
    ):
        if model_type not in ["unigram", "bpe", "char", "word"]:
            raise ValueError(
                "model_type must be one of : [unigram, bpe, char, word]"
            )
        if not os.path.isfile(os.path.abspath(text_file)):
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
                    text_file
                    + " is not a file. please provide text file for training."
                )
        if not isinstance(vocab_size, int):
            raise ValueError("vocab_size must be integer.")
        if not os.path.isdir(model_dir):
            raise ValueError(
                model_dir
                + "must be a directory.\n"
                + "The model files should be: "
                + os.path.join(model_dir, str(vocab_size) + ".{model,vocab}")
            )
        self.text_file = text_file
        self.prefix_model_file = os.path.join(
            model_dir, str(vocab_size) + "_" + model_type
        )
        self.vocab_size = vocab_size
        self.model_type = model_type

    def _train_BPE(self):
        spm.SentencePieceTrainer.train(
            "--pad_id=0 --unk_id=3 "
            + " --input="
            + self.text_file
            + " --model_prefix="
            + self.prefix_model_file
            + " --vocab_size="
            + str(self.vocab_size)
            + " --model_type="
            + self.model_type
        )

    def init_params(self):
        if not os.path.isfile(self.prefix_model_file + ".model"):
            logger.info("Train tokenizer with type:" + self.model_type)
            self._train_BPE()
        else:
            logger.info("Tokenizer is already trained.")
        logger.info("==== Loading Tokenizer ===")
        logger.info("Tokenizer path: " + self.prefix_model_file + ".model")
        logger.info("Tokenizer vocab_size: " + str(self.model_type))
        logger.info("Tokenizer type: " + self.model_type)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.prefix_model_file + ".model")

    def __call__(self, batch, int2lab=None, task="encode", init_params=False):
        if init_params:
            self.init_params()
        if task == "encode" and int2lab is None:
            raise ValueError("Tokenizer encoder must have the int2lab function")
        if task == "encode":
            # we will have only the encode_as_ids
            return (
                [
                    torch.Tensor(self.sp.encode_as_ids(int2lab(utt_seq)))
                    for utt_seq in batch
                ],
                [
                    self.sp.encode_as_pieces(int2lab(utt_seq))
                    for utt_seq in batch
                ],
            )
        elif task == "decode":
            # return ["".join(self.sp.decode_ids(utt_seq.int().tolist())) for utt_seq in batch]
            return [self.sp.decode_ids(utt_seq).split(" ") for utt_seq in batch]
