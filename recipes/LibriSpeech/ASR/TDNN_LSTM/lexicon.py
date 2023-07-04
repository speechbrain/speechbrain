import os
import re
import ast
import argparse
import logging
from typing import Generator, List, Dict, Optional, Tuple
from pathlib import Path

import torch
import k2
import sentencepiece as spm
from tqdm import tqdm

from k2_utils import (
    add_disambig_symbols,
    write_mapping,
    write_lexicon,
    lexicon_to_fst_no_sil
)


UNK = "<UNK>"
EXCLUDED = ["<eps>", "!SIL", "<SPOKEN_NOISE>", UNK, "#0", "<s>", "</s>"]
logger = logging.getLogger(__name__)


class LexiconBPE:
    def __init__(
            self, 
            librispeech_dir: Path, 
            train_sets: List, 
            bpe_model_path: Path,
            tokenizer: Optional[spm.SentencePieceProcessor] = None,
            use_disambig: Optional[bool] = True,
        ):
        """Initialize a lexicon for LibriSpeech using BPE as tokens.
        Args:
            librispeech_dir: Path to the LibriSpeech directory.
            train_sets: A list of training sets, e.g., ["train-clean-100", "train-clean-360"].
            bpe_model_path: Path to the BPE model.
            use_disambig: If True, add disambiguation symbols to the lexicon.
        """
        assert isinstance(train_sets, list), f"train_sets must be a list. {train_sets}"
        assert "train-clean-100" in train_sets, f"train-clean-100 must be in train_sets. {train_sets}"
        self.librispeech_paths = [
            (Path(librispeech_dir) / part) for part in train_sets
        ]
        self.use_disambig = use_disambig
        self.bpe_model_path = Path(bpe_model_path)
        self.bpe_dir = self.bpe_model_path.parent
        self.transcript_words_path = self.bpe_dir / "transcript_words.txt"
        self.disambig_pattern = re.compile(r"^#\d+$")
        self._words_path = self.bpe_dir / "words.txt"
        self.tokenizer = tokenizer
        if tokenizer is None:
            assert os.path.isfile(self.bpe_model_path), f"{self.bpe_model_path} does not exist."
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(self.bpe_model_path))
        self._word2id = None
        self._words = None
        self._token_table = None
        self._word_table = None
        self._L = None
        self._L_inv = None
        self._L_disambig = None

    @property
    def words(self) -> List[str]:
        """Return a list of words excluding special symbols.
        """
        if self._words is None:
            _words = self.word_table.symbols
            self._words = [w for w in _words if w not in EXCLUDED]
        return self._words
    
    @property
    def word_table(self) -> k2.SymbolTable:
        """Return a symbol table for words.
        """
        if self._word_table is None:
            self._word_table = k2.SymbolTable.from_file(self.words_path)
        return self._word_table
    
    @property
    def words_path(self) -> Path:
        """Return a path to a file containing words and their IDs. If
        the file does not exist, create it.
        """
        if self._words_path.exists():
            return self._words_path
        with open(self._words_path, "w") as f:
            for word, idx in self.word2id.items():
                f.write(f"{word} {idx}\n")
        return self._words_path
    
    @property
    def word2id(self) -> Dict[str, int]:
        """Return a dictionary mapping words to their IDs.
        If the dictionary does not exist, create it.
        """
        if self._word2id is None:
            self._word2id = self._word2i_compute()
        return self._word2id
    
    @property
    def tokens(self) -> List[int]:
        """Return a list of token IDs excluding those from
        disambiguation symbols.

        Caution:
          0 is not a token ID so it is excluded from the return value.
        
        Taken from icefall's lexicon implementation.
        """
        symbols = self.token_table.symbols
        ans = []
        for s in symbols:
            if not self.disambig_pattern.match(s):
                ans.append(self.token_table[s])
        if 0 in ans:
            ans.remove(0)
        ans.sort()
        return ans
    
    @property
    def token_table(self) -> k2.SymbolTable:
        """Return a symbol table for tokens.
        """
        if self._token_table is None:
            self.build_lexicon()
        return self._token_table
    
    @property
    def L_disambig(self):
        """Build/Load L transducer. If use_disambig is False then this is the 
        same as self.L
        """
        if self._L is None:
            self.build_lexicon()
        return self._L_disambig
    
    @property
    def L(self):
        """Build/Load L transducer. If use_disambig is True then this is the 
        same as self.L_disambig
        """
        if self._L is None:
            self.build_lexicon()
        if self.use_disambig:
            return self._L_disambig
        return self._L

    @property
    def L_inv(self):
        """Build/Load L_inv transducer."""
        if self._L_inv is None:
            self.build_lexicon()
        return self._L_inv
    
    def build_lexicon(self):
        if self._lexicon_exists():
            self._L = k2.Fsa.from_dict(torch.load(self.bpe_dir / "L.pt"))
            self._L_disambig = k2.Fsa.from_dict(torch.load(self.bpe_dir / "L_disambig.pt"))
            self._L_inv = k2.Fsa.from_dict(torch.load(self.bpe_dir / "Linv.pt"))
            self._token_table = k2.SymbolTable.from_file(self.bpe_dir / "tokens.txt")
            self._word_table = k2.SymbolTable.from_file(self.words_path)
            return
        lexicon, token_sym_table = self._generate_lexicon()
        lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)
        next_token_id = max(token_sym_table.values()) + 1
        for i in range(max_disambig + 1):
            disambig = f"#{i}"
            assert disambig not in token_sym_table
            token_sym_table[disambig] = next_token_id
            next_token_id += 1

        self.word_table.add("#0")
        self.word_table.add("<s>")
        self.word_table.add("</s>")

        write_mapping(self.bpe_dir / "tokens.txt", token_sym_table)
        self._token_table = k2.SymbolTable.from_file(self.bpe_dir / "tokens.txt")

        write_lexicon(self.bpe_dir / "lexicon.txt", lexicon)
        write_lexicon(self.bpe_dir / "lexicon_disambig.txt", lexicon_disambig)

        self._L = lexicon_to_fst_no_sil(
            lexicon,
            token2id=token_sym_table,
            word2id=self.word_table,
        )

        self._L_disambig = lexicon_to_fst_no_sil(
            lexicon_disambig,
            token2id=token_sym_table,
            word2id=self.word_table,
            need_self_loops=True,
        )
        self._L_inv = k2.arc_sort(self._L.invert())  # TODO: maybe L_disambig is needed for L_inv
        torch.save(self._L_inv.as_dict(), self.bpe_dir / "Linv.pt")
        torch.save(self._L.as_dict(), self.bpe_dir / "L.pt")
        torch.save(self._L_disambig.as_dict(), self.bpe_dir / "L_disambig.pt")
    
    def _lexicon_exists(self) -> bool:
        """Check if the lexicon has already been created.
        """
        req_paths = [
            self.bpe_dir / "lexicon.txt",
            self.bpe_dir / "lexicon_disambig.txt",
            self.bpe_dir / "tokens.txt",
            self.bpe_dir / "L.pt",
            self.bpe_dir / "L_disambig.pt",
        ]
        for path in req_paths:
            if not path.exists():
                return False
        return True

    def _word2i_compute(self) -> Dict[str, int]:
        """Read a lexicon from a file.

        Args:
          filename:
            Path to the BPE vocab file.
        Returns:
          Return a mapping from uniques words to their ids.
        """
        words = set()
        for line in self._get_transcripts():
            line_lst = line.split(" ")
            words.update(line_lst[1:])  # ignore the utterance id
        pre_words = ["<eps>", "!SIL", "<SPOKEN_NOISE>", UNK]
        after_words = ["#0", "<s>", "</s>"]
        words = pre_words + sorted(list(words)) + after_words
        word2id = {word: i for i, word in enumerate(words)}

        return word2id
    
    def _get_transcripts(self) -> Generator[str, None, None]:
        """ Read the transcripts from the LibriSpeech dataset.
            Returns:
                A generator of the transcripts. If the file is already created, 
                it will read the file. If not, it will create the file and
                yield the transcripts line by line.
        """
        if self.transcript_words_path.exists():
            with open(self.transcript_words_path, "r") as f:
                for line in f:
                    yield line
            return
        with open(self.transcript_words_path, "w") as fw:
            for part_path in self.librispeech_paths:
                for trans_path in tqdm(
                    part_path.rglob("*.trans.txt"), desc="Distributing tasks", leave=False
                ):
                    with open(trans_path, "r") as fr:
                        # Reading all line of the transcription file
                        for line in fr:
                            line_lst = " ".join(line.strip().replace("\n", "").split(" ")[1:])
                            fw.write(f"{line_lst}\n")
                            yield line_lst
    
    def _generate_lexicon(self) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """ Generate a lexicon from a bpe model.
            See icefall/egs/librispeech/ASR/local/prepare_lang_bpe.py
            
            Returns:
              Return a tuple with two elements:
                - A dict whose keys are words and values are the corresponding
                word pieces.
                - A dict representing the token symbol, mapping from tokens to IDs.
        """
        # Convert word to word piece IDs instead of word piece strings
        # to avoid OOV tokens.
        words_pieces_ids: List[List[int]] = self.tokenizer.encode(self.words, out_type=int)

        # Now convert word piece IDs back to word piece strings.
        words_pieces: List[List[str]] = [self.tokenizer.id_to_piece(ids) for ids in words_pieces_ids]

        lexicon = []
        for word, pieces in zip(self.words, words_pieces):
            lexicon.append((word, pieces))

        lexicon.append((UNK, ["▁", self.tokenizer.id_to_piece(self.tokenizer.unk_id())]))

        token2id: Dict[str, int] = {self.tokenizer.id_to_piece(i): i for i in range(self.tokenizer.vocab_size())}

        return lexicon, token2id

    
    def __getitem__(self, word: str):
        return self.word2id[word]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bpe-model-path", "-t",
        type=str,
        default="exp/librispeech_bpe_5000_units/sp.model",
        help="Path to the sentencepiece model.",
    )
    parser.add_argument(
        "--librispeech-dir", "-d",
        type=str,
        default="data/LibriSpeech",
        help="Path to the LibriSpeech dataset.",
    )
    parser.add_argument(
        "--train-sets", "-s",
        type=str,
        # nargs="+",
        default=["train-clean-100", "train-clean-360", "train-other-500"],
        help="LibriSpeech training sets.",
    )
    parser.add_argument(
        "--build-words", "-b",
        type=bool,
        action="store_true",
        default=True,  # true by default since this is the only reason this file has a cli
        help="LibriSpeech training sets.",
    )
    args = parser.parse_args()
    lex = LexiconBPE(
        bpe_model_path=args.bpe_model_path,
        librispeech_dir=args.librispeech_dir,
        train_sets=ast.literal_eval(args.train_sets),
    )
    if args.build_words:
        lex.words_path  # create the words.txt file