#!/usr/bin/env python3
""" Train a SentencePience model on the whole Lahjoita Puhetta dataset.
This corresponds to roughly 1500 hours of spoken data.

The LibriSpeechTokenizer class is adapted from
`speechbrain/speechbrain/tokenizers/SentencePiece.py`

Authors
 * Georgios Karakasidis 2023
"""

import os
import logging
import re
from pathlib import Path
from typing import List

import torch
import sentencepiece as spm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import merge_char
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.distributed import run_on_main

# torch.cuda.set_device(0)
logger = logging.getLogger(__name__)


class LibriSpeechTokenizer:
    """This function train a SentencePiece model and saved it in the corresponding
    directory.
    """
    def __init__(self,
        data_folder: Path,
        output_folder: str,
        splits: List[str] =["train-clean-100", "train-clean-360", "train-other-500"],
        vocab_size: int =5000,
        token_type: str = "unigram",
        bos_id=-1,
        eos_id=-1,
        character_coverage=1.0,
        split_by_whitespace=True,
        char_format_input=False,
    ):
        self.vocab_size = str(vocab_size)
        self.model_type = token_type
        self.user_defined_symbols = ["<blk>", "<sos/eos>"]
        self.unk_id = len(self.user_defined_symbols)
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.character_coverage = character_coverage
        self.char_format_input = char_format_input
        self.split_by_whitespace = split_by_whitespace
        
        # 1. Now create a .txt file of all transcripts combined
        os.makedirs(output_folder, exist_ok=True)
        self.text_file: str = os.path.join(output_folder, "train-sp.txt")
        if not os.path.isfile(self.text_file):
            all_texts = []
            for split in splits:
                p = os.path.join(data_folder, split)
                all_texts += self.get_texts(p)
            with open(self.text_file, "w") as f:
                f.writelines(all_texts)
        # 2. Define tokenizer's model file
        self.prefix_model_file = os.path.join(
            output_folder, 
            self.vocab_size + "-" + self.model_type
        )
        if not os.path.isfile(self.prefix_model_file+".model"):
            logger.info("Creating a filelist path and training the BPE model.")
            # 3. Now train a bpe model
            run_on_main(self._train_BPE)
        
        # Create SentencePiece model
        logger.info("==== Loading Tokenizer ===")
        logger.info("Tokenizer path: " + self.prefix_model_file + ".model")
        logger.info("Tokenizer vocab_size: " + str(self.vocab_size))
        logger.info("Tokenizer type: " + self.model_type)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.prefix_model_file + ".model")
    
    def get_texts(self, trainset_folder):
        text_lst = get_all_files(
            trainset_folder, match_and=["trans.txt"]
        )
        texts = set()
        # Reading all the transcription files is text_lst
        for file in text_lst:
            with open(file, "r") as f:
                # Reading all line of the transcription file
                for line in f:
                    line_lst = line.strip().split(" ")
                    texts.add("_".join(line_lst[1:]) + "\n")
        return texts
    
    def _check_coverage_from_bpe(self, *args, **kwargs):
        raise NotImplementedError("This method hasn't been implemented in the "+
            "FileListTokenizer. If you need it, you may simply copy-paste the "+
            "default speechbrain SentencePiece tokenizer's method.")
    
    def _train_BPE(self):
        """Train tokenizer with unsupervised techniques (BPE, Unigram) using
        SentencePiece Library. If you use "char" mode, the SentencePiece
        creates a char dict so the vocab_size attribute is not needed.
        """
        input_sentence_size = 10000000
        spm.SentencePieceTrainer.train(
            input=self.text_file,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            model_prefix=self.prefix_model_file,
            input_sentence_size=input_sentence_size,
            character_coverage=self.character_coverage,
            user_defined_symbols=self.user_defined_symbols,
            unk_id=self.unk_id,
            bos_id=-1,
            eos_id=-1,
        )

    def __call__(
        self, batch, batch_lens=None, ind2lab=None, task="encode",
    ):
        """ Copy-pasted from the original file (check top).
        The reason this is copy-pasted is not to require changes to the current code.
        Ofc this could lead to issues if the main branch of speechbrain makes some 
        changes here, so please keep an eye.
        =============================================================================
        This __call__ function implements the tokenizer encoder and decoder
        (restoring the string of word) for BPE, Regularized BPE (with unigram),
        and char (speechbrain/nnet/RNN.py).
        Arguments
        ----------
        batch : tensor.IntTensor or list
            List if ( batch_lens = None and task = "decode_from_list")
            Contains the original labels. Shape: [batch_size, max_length]
        batch_lens : tensor.LongTensor
            Containing the relative length of each label sequences. Must be 1D
            tensor of shape: [batch_size]. (default: None)
        ind2lab : dict
            Dictionary which maps the index from label sequences
            (batch tensor) to string label.
        task : str
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