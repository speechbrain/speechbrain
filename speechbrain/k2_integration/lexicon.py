"""Lexicon class and utilities. Provides functions to read/write
lexicon files and convert them to k2 ragged tensors. The Lexicon
class provides a way to convert a list of words to a ragged tensor
containing token IDs. It also stores the lexicon graph which can
be used by a graph compiler to decode sequences.

This code was adjusted, and therefore heavily inspired or taken from
from icefall's (https://github.com/k2-fsa/icefall) Lexicon class and
its utility functions.


Authors:
  * Zeyu Zhao 2023
  * Georgios Karakasidis 2023
"""


import logging
import re
import sys
import os
from pathlib import Path
from typing import List, Tuple, Union

from . import k2 # import k2 from ./__init__.py
from . import utils

import torch

logger = logging.getLogger(__name__)


def prepare_lexicon(lang_dir, csv_files, extra_vocab_files, add_word_boundary=True):
    """Read csv_files to generate a $lang_dir/lexicon.txt for k2 training.
    This usually includes the csv files of the training set and the dev set in the
    output_folder. During training, we need to make sure that the lexicon.txt contains
    all (or the majority of) the words in the training set and the dev set.

    NOTE: This assumes that the csv files contain the transcription in the last column.

    Also note that in each csv_file, the first line is the header, and the remaining
    lines are in the following format:

    ID, duration, wav, spk_id, wrd (transcription)

    We only need the transcription in this function.

    Writes out $lang_dir/lexicon.txt

    Note that the lexicon.txt is a text file with the following format:
    word1 phone1 phone2 phone3 ...
    word2 phone1 phone2 phone3 ...

    In this code, we simply use the characters in the word as the phones.
    You can use other phone sets, e.g., phonemes, BPEs, to train a better model.

    Arguments
    ---------
    lang_dir: str
        The directory to store the lexicon.txt
    csv_files: List[str]
        A list of csv file paths
    extra_vocab_files: List[str]
        A list of extra vocab files. For example, for librispeech this could be the
        librispeech-vocab.txt file.
    add_word_boundary: bool
        whether to add word boundary symbols <eow> at the end of each line to the
        lexicon for every word.

    Example
    -------
    >>> from speechbrain.k2_integration.lexicon import prepare_lexicon
    >>> # Create some dummy csv files containing only the words `hello`, `world`.
    >>> # The first line is the header, and the remaining lines are in the following
    >>> # format:
    >>> # ID, duration, wav, spk_id, wrd (transcription)
    >>> csv_file = getfixture('tmpdir').join("train.csv")
    >>> with open(csv_file, "w") as f:
    ...     f.write("ID,duration,wav,spk_id,wrd\n")
    ...     f.write("1,1,1,1,hello world\n")
    ...     f.write("2,0.5,1,1,hello\n")
    >>> csv_files = [csv_file]
    >>> lang_dir = getfixture('tmpdir')
    >>> extra_vocab_files = []
    >>> prepare_lexicon(lang_dir, csv_files, extra_vocab_files, add_word_boundary=False)
    >>> with open(lang_dir.join("lexicon.txt"), "r") as f:
    ...     assert f.read() == "<UNK> <unk>\nhello h e l l o\nworld w o r l d\n"
    """
    # Read train.csv, dev-clean.csv to generate a lexicon.txt for k2 training
    lexicon = dict()
    for file in csv_files:
        with open(file) as f:
            # Omit the first line
            f.readline()
            # Read the remaining lines
            for line in f:
                # Split the line
                trans = line.strip().split(",")[-1]
                # Split the transcription into words
                words = trans.split()
                for word in words:
                    if word not in lexicon:
                        if add_word_boundary:
                            lexicon[word] = list(word) + ["<eow>"]
                        else:
                            lexicon[word] = list(word)

    for file in extra_vocab_files:
        with open(file) as f:
            for line in f:
                # Split the line
                word = line.strip().split()[0]
                # Split the transcription into words
                if word not in lexicon:
                    if add_word_boundary:
                        lexicon[word] = list(word) + ["<eow>"]
                    else:
                        lexicon[word] = list(word)
    # Write the lexicon to lang_dir/lexicon.txt
    os.makedirs(lang_dir, exist_ok=True)
    with open(os.path.join(lang_dir, "lexicon.txt"), "w") as f:
        fc = "<UNK> <unk>\n"
        for word in lexicon:
            fc += word + " " + " ".join(lexicon[word]) + "\n"
        f.write(fc)


def read_lexicon(filename: str) -> List[Tuple[str, List[str]]]:
    """Read a lexicon from `filename`.

    Each line in the lexicon contains "word p1 p2 p3 ...".
    That is, the first field is a word and the remaining
    fields are tokens. Fields are separated by space(s).

    Arguments
    ---------
    filename: str
        Path to the lexicon.txt

    Returns
    -------
    ans:
        A list of tuples., e.g., [('w', ['p1', 'p2']), ('w1', ['p3, 'p4'])]
    """
    ans = []

    with open(filename, "r", encoding="utf-8") as f:
        whitespace = re.compile("[ \t]+")
        for line in f:
            a = whitespace.split(line.strip(" \t\r\n"))
            if len(a) == 0:
                continue

            if len(a) < 2:
                logger.info(f"Found bad line {line} in lexicon file {filename}")
                logger.info(
                    "Every line is expected to contain at least 2 fields"
                )
                sys.exit(1)
            word = a[0]
            if word == "<eps>":
                logger.info(f"Found bad line {line} in lexicon file {filename}")
                logger.info("<eps> should not be a valid word")
                sys.exit(1)

            tokens = a[1:]
            ans.append((word, tokens))

    return ans


def write_lexicon(
    filename: Union[str, Path], lexicon: List[Tuple[str, List[str]]]
) -> None:
    """Write a lexicon to a file.

    Arguments
    ---------
    filename: str
        Path to the lexicon file to be generated.
    lexicon: List[Tuple[str, List[str]]]
        It can be the return value of :func:`read_lexicon`.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for word, tokens in lexicon:
            f.write(f"{word} {' '.join(tokens)}\n")


class Lexicon(object):
    """Unit based lexicon. It is used to map a list of words to each word's
    sequence of tokens (characters). It also stores the lexicon graph which
    can be used by a graph compiler to decode sequences.

    Arguments
    ---------
    lang_dir: str
        Path to the lang directory. It is expected to contain the following
        files:
            - tokens.txt
            - words.txt
            - L.pt
    disambig_pattern: str
        It contains the pattern for disambiguation symbols.

    Example
    -------
    >>> import k2
    >>> from speechbrain.k2_integration.lexicon import Lexicon
    >>> from speechbrain.k2_integration.graph_compiler import CharCtcTrainingGraphCompiler
    >>> from speechbrain.k2_integration.prepare_lang import prepare_lang

    >>> # Create a small lexicon containing only two words and write it to a file.
    >>> lang_tmpdir = getfixture('tmpdir')
    >>> lexicon_sample = '''hello h e l l o
    ... world w o r l d'''
    >>> lexicon_file = lang_tmpdir.join("lexicon.txt")
    >>> lexicon_file.write(lexicon_sample)
    >>> # Create a lang directory with the lexicon and L.pt, L_inv.pt, L_disambig.pt
    >>> prepare_lang(lang_tmpdir)
    >>> # Create a lexicon object
    >>> lexicon = Lexicon(lang_tmpdir)
    >>> # Make sure the lexicon was loaded correctly
    >>> assert isinstance(lexicon.token_table, k2.SymbolTable)
    >>> assert isinstance(lexicon.L, k2.Fsa)
    """

    def __init__(
        self,
        lang_dir: Path,
        disambig_pattern: re.Pattern = re.compile(r"^#\d+$"),  # type: ignore
    ):
        self.lang_dir = lang_dir = Path(lang_dir)
        self.token_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
        self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")
        self._L_disambig = None
        self._mapping_loaded = False

        if (lang_dir / "L.pt").exists():
            logger.info(f"Loading pre-compiled {lang_dir}/L.pt")
            L = k2.Fsa.from_dict(torch.load(lang_dir / "L.pt"))
        else:
            raise RuntimeError(
                f"{lang_dir}/L.pt does not exist. Please make sure "
                f"you have successfully created L.pt in {lang_dir}"
            )

        if (lang_dir / "Linv.pt").exists():
            logger.info(f"Loading pre-compiled {lang_dir}/Linv.pt")
            L_inv = k2.Fsa.from_dict(torch.load(lang_dir / "Linv.pt"))
        else:
            logger.info("Converting L.pt to Linv.pt")
            L_inv = k2.arc_sort(L.invert())
            torch.save(L_inv.as_dict(), lang_dir / "Linv.pt")

        # We save L_inv instead of L because it will be used to intersect with
        # transcript FSAs, both of whose labels are word IDs.
        self.L_inv = L_inv
        self.L = L
        self.disambig_pattern = disambig_pattern

    @property
    def L_disambig(self) -> k2.Fsa:
        """Return the lexicon FSA (with disambiguation symbols).
        Needed for HLG construction.
        """
        if self._L_disambig is None:
            logger.info(f"Loading pre-compiled {self.lang_dir}/L_disambig.pt")
            self._L_disambig = k2.Fsa.from_dict(
                torch.load(self.lang_dir / "L_disambig.pt")
            )
        return self._L_disambig

    def __getattribute__(self, attr):
        """Lazy load mapping *2* attributes"""
        if attr in ["token2idx", "idx2token", "word2idx", "idx2word", "word2tids"]:
            self._load_mapping()
        return object.__getattribute__(self, attr)

    def _load_mapping(self):
        """Load mappings including token2idx idx2token word2idx idx2word word2tids,
        each of which is a dict.

        self.token2idx: Dict[str, int]
        self.idx2token: Dict[int, str]
        self.word2idx: Dict[str, int]
        self.idx2word: Dict[int, str]
        self.word2tids: Dict[str, List[int]]
        """
        if self._mapping_loaded:
            return

        token2idx = {}
        idx2token = {}
        with open(self.lang_dir / "tokens.txt", "r", encoding="utf-8") as f:
            for line in f:
                token, idx = line.strip().split()
                token2idx[token] = int(idx)
                idx2token[int(idx)] = token
        word2idx = {}
        idx2word = {}
        with open(self.lang_dir / "words.txt", "r", encoding="utf-8") as f:
            for line in f:
                word, idx = line.strip().split()
                word2idx[word] = int(idx)
                idx2word[int(idx)] = word
        word2tids = {}
        with open(self.lang_dir / "lexicon.txt", "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().split()[0]
                tokens = line.strip().split()[1:]
                tids = [token2idx[t] for t in tokens]
                if word not in word2tids:
                    word2tids[word] = []
                word2tids[word].append(tids)
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.word2idx  = word2idx
        self.idx2word  = idx2word
        self.word2tids = word2tids
        _mapping_loaded = True


    @property
    def tokens(self) -> List[int]:
        """Return a list of token IDs excluding those from
        disambiguation symbols.

        NOTE:
          0 is not a token ID so it is excluded from the return value.
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
