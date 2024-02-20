"""Lexicon class and utilities. Provides functions to read/write
lexicon files and convert them to k2 ragged tensors. The Lexicon
class provides a way to convert a list of words to a ragged tensor
containing token IDs. It also stores the lexicon graph which can
be used by a graph compiler to decode sequences.

This code was adjusted, and therefore heavily inspired or taken from
from icefall's (https://github.com/k2-fsa/icefall) Lexicon class and
its utility functions.


Authors:
  * Pierre Champion 2023
  * Zeyu Zhao 2023
  * Georgios Karakasidis 2023
"""


import logging
import re
import os
import csv
from pathlib import Path
from typing import List, Union, Tuple, Optional

from . import k2  # import k2 from ./__init__.py

import torch

logger = logging.getLogger(__name__)

UNK = "<UNK>"  # unknow word
UNK_t = "<unk>"  # unknow token
EOW = "<eow>"  # end of word
EPS = "<eps>"  # epsilon

DISAMBIG_PATTERN: re.Pattern = re.compile(
    r"^#\d+$"
)  # pattern for disambiguation symbols.


class Lexicon(object):
    """
    Unit based lexicon. It is used to map a list of words to each word's
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

    Example
    -------
    >>> from speechbrain.k2_integration import k2
    >>> from speechbrain.k2_integration.lexicon import Lexicon
    >>> from speechbrain.k2_integration.graph_compiler import CtcGraphCompiler
    >>> from speechbrain.k2_integration.prepare_lang import prepare_lang

    >>> # Create a small lexicon containing only two words and write it to a file.
    >>> lang_tmpdir = getfixture('tmpdir')
    >>> lexicon_sample = '''hello h e l l o\\nworld w o r l d'''
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
        self, lang_dir: Path,
    ):
        self.lang_dir = lang_dir = Path(lang_dir)
        self.token_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
        self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")
        self.word2tokenids = {}
        with open(lang_dir / "lexicon.txt", "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().split()[0]
                tokens = line.strip().split()[1:]
                tids = [self.token_table[t] for t in tokens]
                # handle multiple pronunciation
                if word not in self.word2tokenids:
                    self.word2tokenids[word] = []
                self.word2tokenids[word].append(tids)

        self._L_disambig = None

        if (lang_dir / "L.pt").exists():
            logger.info(f"Loading compiled {lang_dir}/L.pt")
            L = k2.Fsa.from_dict(torch.load(lang_dir / "L.pt"))
        else:
            raise RuntimeError(
                f"{lang_dir}/L.pt does not exist. Please make sure "
                f"you have successfully created L.pt in {lang_dir}"
            )

        if (lang_dir / "Linv.pt").exists():
            logger.info(f"Loading compiled {lang_dir}/Linv.pt")
            L_inv = k2.Fsa.from_dict(torch.load(lang_dir / "Linv.pt"))
        else:
            logger.info("Converting L.pt to Linv.pt")
            L_inv = k2.arc_sort(L.invert())
            torch.save(L_inv.as_dict(), lang_dir / "Linv.pt")

        # We save L_inv instead of L because it will be used to intersect with
        # transcript FSAs, both of whose labels are word IDs.
        self.L_inv = L_inv
        self.L = L

    @property
    def tokens(self) -> List[int]:
        """
        Return a list of token IDs excluding those from
        disambiguation symbols and epsilon.
        """
        symbols = self.token_table.symbols
        ans = []
        for s in symbols:
            if not DISAMBIG_PATTERN.match(s) or s != EPS:
                ans.append(self.token_table[s])
        ans.sort()
        return ans

    @property
    def L_disambig(self) -> k2.Fsa:
        """
        Return the lexicon FSA (with disambiguation symbols).
        Needed for HLG construction.
        """
        if self._L_disambig is None:
            logger.info(f"Loading compiled {self.lang_dir}/L_disambig.pt")
            if (self.lang_dir / "L_disambig.pt").exists():
                self._L_disambig = k2.Fsa.from_dict(
                    torch.load(self.lang_dir / "L_disambig.pt")
                )
            else:
                raise RuntimeError(
                    f"{self.lang_dir}/L_disambig.pt does not exist. Please make sure "
                    f"you have successfully created L_disambig.pt in {self.lang_dir}"
                )
        return self._L_disambig

    def remove_G_rescoring_disambig_symbols(self, G: k2.Fsa):
        """
        Remove the disambiguation symbols of a G graph

        Arguments
        ---------
        G: k2.Fsa
            The G graph to be modified
        """
        G.labels[G.labels >= self.word_table["#0"]] = 0

    def remove_LG_disambig_symbols(self, LG: k2.Fsa) -> k2.Fsa:
        """
        Remove the disambiguation symbols of an LG graph
        Needed for HLG construction.

        Arguments
        ---------
        LG: k2.Fsa
            The LG graph to be modified
        """

        first_token_disambig_id = self.token_table["#0"]
        first_word_disambig_id = self.word_table["#0"]

        logger.debug("Removing disambiguation symbols on LG")
        # NOTE: We need to clone here since LG.labels is just a reference to a tensor
        #       and we will end up having issues with misversioned updates on fsa's
        #       properties.
        labels = LG.labels.clone()
        labels[labels >= first_token_disambig_id] = 0
        LG.labels = labels

        assert isinstance(LG.aux_labels, k2.RaggedTensor)
        LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0
        return LG

    def texts_to_word_ids(
        self,
        texts: List[str],
        add_sil_token_as_separator=False,
        sil_token_id: Optional[int] = None,
        log_unknown_warning=True,
    ) -> List[List[int]]:
        """
        Convert a list of texts into word IDs.

        This method performs the mapping of each word in the input texts to its corresponding ID.
        The result is a list of lists, where each inner list contains the word IDs for a sentence.
        If the `add_sil_token_as_separator` flag is True, a silence token is inserted between words,
        and the `sil_token_id` parameter specifies the ID for the silence token.
        If a word is not found in the vocabulary, a warning is logged if `log_unknown_warning` is True.

        Arguments
        ---------
        texts: List[str]
            A list of strings where each string represents a sentence.
            Each sentence is composed of space-separated words.

        add_sil_token_as_separator: bool
            Flag indicating whether to add a silence token as a separator between words.

        sil_token_id: Optional[int]
            The ID of the silence token. If not provided, the separator is not added.

        log_unknown_warning: bool
            Flag indicating whether to log a warning for unknown words.

        Returns
        -------
        word_ids: List[List[int]]
            A list of lists where each inner list represents the word IDs for a sentence.
            The word IDs are obtained based on the vocabulary mapping.
        """
        word_ids = self._texts_to_ids(
            texts, log_unknown_warning, _mapper="word_table"
        )
        if add_sil_token_as_separator:
            assert (
                sil_token_id is not None
            ), "sil_token_id=None while add_sil_token_as_separator=True"
            for i in range(len(word_ids)):
                word_ids[i] = [
                    x for item in word_ids[i] for x in (item, sil_token_id)
                ][:-1]
        return word_ids

    def texts_to_token_ids(
        self, texts: List[str], log_unknown_warning=True,
    ) -> List[List[List[int]]]:
        """
        Convert a list of text sentences into token IDs.

        Parameters
        ----------
        texts: List[str]
            A list of strings, where each string represents a sentence.
            Each sentence consists of space-separated words.
            Example:
                ['hello world', 'tokenization with lexicon']

        log_unknown_warning: bool
            Flag indicating whether to log warnings for out-of-vocabulary tokens.
            If True, warnings will be logged when encountering unknown tokens.

        Returns
        -------
        token_ids: List[List[List[int]]]
            A list containing token IDs for each sentence in the input.
            The structure of the list is as follows:
            [
                [  # For the first sentence
                    [token_id_1, token_id_2, ..., token_id_n],
                    [token_id_1, token_id_2, ..., token_id_m],
                    ...
                ],
                [  # For the second sentence
                    [token_id_1, token_id_2, ..., token_id_p],
                    [token_id_1, token_id_2, ..., token_id_q],
                    ...
                ],
                ...
            ]
            Each innermost list represents the token IDs for a word in the sentence.
        """
        return self._texts_to_ids(
            texts, log_unknown_warning, _mapper="word2tokenids"
        )

    def texts_to_token_ids_with_multiple_pronunciation(
        self, texts: List[str], log_unknown_warning=True,
    ) -> List[List[List[List[int]]]]:
        """
        Convert a list of input texts to token IDs with multiple pronunciation variants.

        This method converts input texts into token IDs, considering multiple pronunciation variants.
        The resulting structure allows for handling various pronunciations of words within the given texts.

        Arguments
        ---------
        texts: List[str]
            A list of strings, where each string represents a sentence for an utterance.
            Each sentence consists of space-separated words.

        log_unknown_warning: bool
            Indicates whether to log warnings for out-of-vocabulary (OOV) tokens.
            If set to True, warnings will be logged for OOV tokens during the conversion.

        Returns
        -------
        token_ids: List[List[List[List[int]]]]
            A nested list structure containing token IDs for each utterance. The structure is as follows:
            - Outer List: Represents different utterances.
            - Middle List: Represents different pronunciation variants for each utterance.
            - Inner List: Represents the sequence of token IDs for each pronunciation variant.
            - Innermost List: Represents the token IDs for each word in the sequence.
        """
        return self._texts_to_ids(
            texts,
            log_unknown_warning,
            _mapper="word2tokenids",
            _multiple_pronunciation=True,
        )

    def _texts_to_ids(
        self,
        texts: List[str],
        log_unknown_warning: bool,
        _mapper: str,
        _multiple_pronunciation=False,
    ):
        """
        Convert a list of texts to a list of IDs, which can be either word IDs or
        a list of token IDs.

        Arguments
        ---------
        texts: List[str]
            A list of strings where each string consists of space-separated words.
            Example:
                ['hello world', 'tokenization with lexicon']

        log_unknown_warning: bool
            Log a warning if a word is not found in the token-to-IDs mapping.

        _mapper: str
            The mapper to use, either "word_table" (e.g., "TEST" -> 176838) or
            "word2tokenids" (e.g., "TEST" -> [23, 8, 22, 23]).

        _multiple_pronunciation: bool
            Allow returning all pronunciations of a word from the lexicon.
            If False, only return the first pronunciation.

        Returns
        -------
        ids_list: List[List[int] or int]
            Returns a list-of-list of word IDs or a list of token IDs.
        """
        oov_token_id = self.word_table[UNK]
        if _mapper == "word2tokenids":
            oov_token_id = [self.token_table[UNK_t]]
        ids = getattr(self, _mapper)

        ids_list = []
        for text in texts:
            word_ids = []
            words = text.split()
            for i, word in enumerate(words):
                if word in ids:
                    idword = ids[word]
                    if isinstance(idword, list) and not _multiple_pronunciation:
                        idword = idword[
                            0
                        ]  # only first spelling of a word (for word2tokenids mapper)
                    word_ids.append(idword)
                else:
                    word_ids.append(oov_token_id)
                    if log_unknown_warning:
                        logger.warning(
                            f"Cannot find word {word} in the mapper {_mapper}."
                            f" Replacing it with OOV token."
                            f" Note that it is fine if you are testing."
                        )

            ids_list.append(word_ids)
        return ids_list

    def arc_sort(self):
        """
        Sort L, L_inv, L_disambig arcs of every state.
        """
        self.L = k2.arc_sort(self.L)
        self.L_inv = k2.arc_sort(self.L_inv)
        if self._L_disambig is not None:
            self._L_disambig = k2.arc_sort(self._L_disambig)

    def to(self, device: str = "cpu"):
        """
        Device to move L, L_inv and L_disambig to

        Arguments
        ---------
        device: str
            The device
        """
        self.L = self.L.to(device)
        self.L_inv = self.L_inv.to(device)
        if self._L_disambig is not None:
            self._L_disambig = self._L_disambig.to(device)


def prepare_char_lexicon(
    lang_dir,
    vocab_files,
    extra_csv_files=[],
    column_text_key="wrd",
    add_word_boundary=True,
):
    """
    Read extra_csv_files to generate a $lang_dir/lexicon.txt for k2 training.
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
    vocab_files: List[str]
        A list of extra vocab files. For example, for librispeech this could be the
        librispeech-vocab.txt file.
    extra_csv_files: List[str]
        A list of csv file paths
    column_text_key: str
        The column name of the transcription in the csv file. By default, it is "wrd".
    add_word_boundary: bool
        whether to add word boundary symbols <eow> at the end of each line to the
        lexicon for every word.

    Example
    -------
    >>> from speechbrain.k2_integration.lexicon import prepare_char_lexicon
    >>> # Create some dummy csv files containing only the words `hello`, `world`.
    >>> # The first line is the header, and the remaining lines are in the following
    >>> # format:
    >>> # ID, duration, wav, spk_id, wrd (transcription)
    >>> csv_file = getfixture('tmpdir').join("train.csv")
    >>> # Data to be written to the CSV file.
    >>> import csv
    >>> data = [
    ...    ["ID", "duration", "wav", "spk_id", "wrd"],
    ...    [1, 1, 1, 1, "hello world"],
    ...    [2, 0.5, 1, 1, "hello"]
    ... ]
    >>> with open(csv_file, "w", newline="") as f:
    ...    writer = csv.writer(f)
    ...    writer.writerows(data)
    >>> extra_csv_files = [csv_file]
    >>> lang_dir = getfixture('tmpdir')
    >>> vocab_files = []
    >>> prepare_char_lexicon(lang_dir, vocab_files, extra_csv_files=extra_csv_files, add_word_boundary=False)
    """
    # Read train.csv, dev-clean.csv to generate a lexicon.txt for k2 training
    lexicon = dict()
    if len(extra_csv_files) != 0:
        for file in extra_csv_files:
            with open(file, "r") as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    # Split the transcription into words
                    words = row[column_text_key].split()
                    for word in words:
                        if word not in lexicon:
                            if add_word_boundary:
                                lexicon[word] = list(word) + [EOW]
                            else:
                                lexicon[word] = list(word)

    for file in vocab_files:
        with open(file) as f:
            for line in f:
                # Split the line
                word = line.strip().split()[0]
                # Split the transcription into words
                if word not in lexicon:
                    if add_word_boundary:
                        lexicon[word] = list(word) + [EOW]
                    else:
                        lexicon[word] = list(word)
    # Write the lexicon to lang_dir/lexicon.txt
    os.makedirs(lang_dir, exist_ok=True)
    with open(os.path.join(lang_dir, "lexicon.txt"), "w") as f:
        fc = f"{UNK} {UNK_t}\n"
        for word in lexicon:
            fc += word + " " + " ".join(lexicon[word]) + "\n"
        f.write(fc)


def read_lexicon(filename: str) -> List[Tuple[str, List[str]]]:
    """
    Read a lexicon from `filename`.

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
                raise RuntimeError(
                    f"Found bad line {line} in lexicon file {filename}"
                    "Every line is expected to contain at least 2 fields"
                )
            word = a[0]
            if word == EPS:
                raise RuntimeError(
                    f"Found bad line {line} in lexicon file {filename}"
                    f"{EPS} should not be a valid word"
                )
            tokens = a[1:]
            ans.append((word, tokens))
    return ans


def write_lexicon(
    filename: Union[str, Path], lexicon: List[Tuple[str, List[str]]]
) -> None:
    """
    Write a lexicon to a file.

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
