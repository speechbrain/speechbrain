# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import k2
import torch


def read_lexicon(filename: str) -> List[Tuple[str, List[str]]]:
    """Read a lexicon from `filename`.

    Each line in the lexicon contains "word p1 p2 p3 ...".
    That is, the first field is a word and the remaining
    fields are tokens. Fields are separated by space(s).

    Args:
      filename:
        Path to the lexicon.txt

    Returns:
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
                logging.info(
                    f"Found bad line {line} in lexicon file {filename}"
                )
                logging.info(
                    "Every line is expected to contain at least 2 fields"
                )
                sys.exit(1)
            word = a[0]
            if word == "<eps>":
                logging.info(
                    f"Found bad line {line} in lexicon file {filename}"
                )
                logging.info("<eps> should not be a valid word")
                sys.exit(1)

            tokens = a[1:]
            ans.append((word, tokens))

    return ans


def write_lexicon(filename: str, lexicon: List[Tuple[str, List[str]]]) -> None:
    """Write a lexicon to a file.

    Args:
      filename:
        Path to the lexicon file to be generated.
      lexicon:
        It can be the return value of :func:`read_lexicon`.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for word, tokens in lexicon:
            f.write(f"{word} {' '.join(tokens)}\n")


class Lexicon(object):
    """Phone based lexicon."""

    def __init__(
        self,
        lang_dir: Path,
        disambig_pattern: str = re.compile(r"^#\d+$"),
    ):
        """
        Args:
          lang_dir:
            Path to the lang director. It is expected to contain the following
            files:
                - tokens.txt
                - words.txt
                - L.pt
            The above files are produced by the script `prepare.sh`. You
            should have run that before running the training code.
          disambig_pattern:
            It contains the pattern for disambiguation symbols.
        """
        lang_dir = Path(lang_dir)
        self.token_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
        self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")

        if (lang_dir / "Linv.pt").exists():
            logging.info(f"Loading pre-compiled {lang_dir}/Linv.pt")
            L_inv = k2.Fsa.from_dict(torch.load(lang_dir / "Linv.pt"))
        else:
            logging.info("Converting L.pt to Linv.pt")
            L = k2.Fsa.from_dict(torch.load(lang_dir / "L.pt"))
            L_inv = k2.arc_sort(L.invert())
            torch.save(L_inv.as_dict(), lang_dir / "Linv.pt")

        # We save L_inv instead of L because it will be used to intersect with
        # transcript, both of whose labels are word IDs.
        self.L_inv = L_inv
        self.disambig_pattern = disambig_pattern

    @property
    def tokens(self) -> List[int]:
        """Return a list of token IDs excluding those from
        disambiguation symbols.

        Caution:
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


class BpeLexicon(Lexicon):
    def __init__(
        self,
        lang_dir: Path,
        disambig_pattern: str = re.compile(r"^#\d+$"),
    ):
        """
        Refer to the help information in Lexicon.__init__.
        """
        super().__init__(lang_dir=lang_dir, disambig_pattern=disambig_pattern)

        self.ragged_lexicon = self.convert_lexicon_to_ragged(
            lang_dir / "lexicon.txt"
        )

    def convert_lexicon_to_ragged(self, filename: str) -> k2.RaggedTensor:
        """Read a BPE lexicon from file and convert it to a
        k2 ragged tensor.

        Args:
          filename:
            Filename of the BPE lexicon, e.g., data/lang/bpe/lexicon.txt
        Returns:
          A k2 ragged tensor with two axes [word_id]
        """
        disambig_id = self.word_table["#0"]
        # We reuse the same words.txt from the phone based lexicon
        # so that we can share the same G.fst. Here, we have to
        # exclude some words present only in the phone based lexicon.
        excluded_words = ["<eps>", "!SIL", "<SPOKEN_NOISE>"]

        # epsilon is not a word, but it occupies on position
        #
        row_splits = [0]
        token_ids = []

        lexicon = read_lexicon(filename)
        lexicon = dict(lexicon)

        for i in range(disambig_id):
            w = self.word_table[i]
            if w in excluded_words:
                row_splits.append(row_splits[-1])
                continue
            pieces = lexicon[w]
            piece_ids = [self.token_table[k] for k in pieces]

            row_splits.append(row_splits[-1] + len(piece_ids))
            token_ids.extend(piece_ids)

        cached_tot_size = row_splits[-1]
        row_splits = torch.tensor(row_splits, dtype=torch.int32)

        shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=cached_tot_size
        )
        values = torch.tensor(token_ids, dtype=torch.int32)

        return k2.RaggedTensor(shape, values)

    def words_to_piece_ids(self, words: List[str]) -> k2.RaggedTensor:
        """Convert a list of words to a ragged tensor contained
        word piece IDs.
        """
        word_ids = [self.word_table[w] for w in words]
        word_ids = torch.tensor(word_ids, dtype=torch.int32)

        ragged, _ = self.ragged_lexicon.index(
            indexes=word_ids,
            axis=0,
            need_value_indexes=False,
        )
        return ragged
