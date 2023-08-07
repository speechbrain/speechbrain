import os
import re
import ast
import argparse
import logging
from collections import defaultdict
from typing import Generator, List, Dict, Optional, Tuple, Any
from pathlib import Path

import torch
import k2
import sentencepiece as spm
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm


UNK = "<UNK>"
EXCLUDED = ["<eps>", "!SIL", "<SPOKEN_NOISE>", UNK, "#0", "<s>", "</s>"]
logger = logging.getLogger(__name__)


class LexiconBPE:
    def __init__(
            self, 
            librispeech_dir: Path, 
            train_sets: List, 
            tokenizer_model_path: Path,
            tokenizer: Optional[spm.SentencePieceProcessor] = None,
            use_disambig: Optional[bool] = True,
        ):
        """Initialize a lexicon for LibriSpeech using BPE as tokens.
        Args:
            librispeech_dir: Path to the LibriSpeech directory.
            train_sets: A list of training sets, e.g., ["train-clean-100", "train-clean-360"].
            tokenizer_model_path: Path to the tokenizer model (BPE).
            use_disambig: If True, add disambiguation symbols to the lexicon.
        """
        assert isinstance(train_sets, list), f"train_sets must be a list. {train_sets}"
        assert "train-clean-100" in train_sets, f"train-clean-100 must be in train_sets. {train_sets}"
        self.librispeech_paths = [
            (Path(librispeech_dir) / part) for part in train_sets
        ]
        self.use_disambig = use_disambig
        self.tokenizer_model_path = Path(tokenizer_model_path)
        self.tokenizer_dir = self.tokenizer_model_path.parent
        self.transcript_words_path = self.tokenizer_dir / "transcript_words.txt"
        self.disambig_pattern = re.compile(r"^#\d+$")
        self._words_path = self.tokenizer_dir / "words.txt"
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = self._init_tokenizer()
        self._word2id = None
        self._words = None
        self._token_table = None
        self._word_table = None
        self._L = None
        self._L_inv = None
        self._L_disambig = None
    
    def _init_tokenizer(self):
        assert os.path.isfile(self.tokenizer_model_path), f"{self.tokenizer_model_path} does not exist."
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(self.tokenizer_model_path))
        return tokenizer

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
            self._L = k2.Fsa.from_dict(torch.load(self.tokenizer_dir / "L.pt"))
            self._L_disambig = k2.Fsa.from_dict(torch.load(self.tokenizer_dir / "L_disambig.pt"))
            self._L_inv = k2.Fsa.from_dict(torch.load(self.tokenizer_dir / "Linv.pt"))
            self._token_table = k2.SymbolTable.from_file(self.tokenizer_dir / "tokens.txt")
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

        write_mapping(self.tokenizer_dir / "tokens.txt", token_sym_table)
        self._token_table = k2.SymbolTable.from_file(self.tokenizer_dir / "tokens.txt")

        write_lexicon(self.tokenizer_dir / "lexicon.txt", lexicon)
        write_lexicon(self.tokenizer_dir / "lexicon_disambig.txt", lexicon_disambig)

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
        self._L_inv = k2.arc_sort(self._L_disambig.invert())  # TODO: maybe L_disambig is needed for L_inv
        torch.save(self._L_inv.as_dict(), self.tokenizer_dir / "Linv.pt")
        torch.save(self._L.as_dict(), self.tokenizer_dir / "L.pt")
        torch.save(self._L_disambig.as_dict(), self.tokenizer_dir / "L_disambig.pt")
    
    def _lexicon_exists(self) -> bool:
        """Check if the lexicon has already been created.
        """
        req_paths = [
            self.tokenizer_dir / "lexicon.txt",
            self.tokenizer_dir / "lexicon_disambig.txt",
            self.tokenizer_dir / "tokens.txt",
            self.tokenizer_dir / "L.pt",
            self.tokenizer_dir / "L_disambig.pt",
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
    
    def _generate_lexicon(self) -> Tuple[List[Tuple[str, List[str]]], Dict[str, int]]:
        """ Generate a lexicon from a bpe model.
            See icefall/egs/librispeech/ASR/local/prepare_lang_bpe.py
            
            Returns:
              Return a tuple with two elements:
                - A list representing mappings from words to their corresponding
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
    
    def tokenize_to_ids(self, sentence: str) -> List[int]:
        """Tokenize a sentence.

        Args:
          sentence:
            A sentence to be tokenized.
        Returns:
          Return a list of token ids.
        """
        return self.tokenizer.encode(sentence, out_type=int)
    
    def build_ctc_topo2(self):
        # See https://github.com/k2-fsa/k2/issues/746#issuecomment-856421616
        phones = self.tokens.copy()
        assert 0 in phones, 'We assume 0 is the ID of the blank symbol'
        phones = phones.copy()
        phones.remove(0)

        num_phones = len(phones)

        start = 0
        final = num_phones + 1

        arcs = []
        arcs.append([start, start, 0, 0, 0])
        arcs.append([start, final, -1, -1, 0])
        arcs.append([final])
        for i, p in enumerate(phones):
            i += 1
            arcs.append([start, start, p, p, 0])

            arcs.append([start, i, p, p, 0])
            arcs.append([i, i, p, 0, 0])

            arcs.append([i, start, p, 0, 0])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [' '.join(arc) for arc in arcs]
        arcs = '\n'.join(arcs)
        ctc_topo = k2.Fsa.from_str(arcs, False)
        return k2.arc_sort(ctc_topo)
    
    def __getitem__(self, word: str):
        return self.word2id[word]


class LexiconChar(LexiconBPE):
    
    def _init_tokenizer(self):
        label_encoder = sb.dataio.encoder.CTCTextEncoder()
        return label_encoder
    
    def _generate_lexicon(self) -> Tuple[List[Tuple[str, List[str]]], Dict[str, int]]:
        """ Generate a lexicon from a bpe model.
            See icefall/egs/librispeech/ASR/local/prepare_lang_bpe.py
            
            Returns:
              Return a tuple with two elements:
                - A list representing mappings from words to their corresponding
                word pieces.
                - A dict representing the token symbol, mapping from tokens to IDs.
        """
        # Convert word to sequences of characters.
        words_pieces_ids: List[List[int]] = [self.tokenizer.encode_sequence(list(wrd)) for wrd in self.words]

        # Now convert word piece IDs back to word piece strings.
        words_pieces: List[List[str]] = [self.tokenizer.decode_ndim(ids) for ids in words_pieces_ids]

        lexicon = []
        for word, pieces in zip(self.words, words_pieces):
            lexicon.append((word, pieces))

        # TODO: no unk_id since we assume all characters exist
        if hasattr(self.tokenizer, "unk_label"):
            lexicon.append((UNK, ["▁", self.tokenizer.decode_ndim([self.unk_label])[0]]))

        token2id: Dict[str, int] = {self.tokenizer.decode_ndim([i])[0]: i for i in range(len(self.tokenizer))}

        return lexicon, token2id
    
    def tokenize_to_ids(self, sentence: str) -> List[int]:
        """Tokenize a sentence.

        Args:
          sentence:
            A sentence to be tokenized.
        Returns:
          Return a list of token ids.
        """
        return self.tokenizer.encode_sequence(list(sentence))



def lexicon_to_fst_no_sil(
    lexicon: LexiconBPE,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Convert a lexicon to an FST (in k2 format).

    Args:
      lexicon:
        The input lexicon. See also :func:`read_lexicon`
      token2id:
        A dict mapping tokens to IDs.
      word2id:
        A dict mapping words to IDs.
      need_self_loops:
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state. The input label for this
        self loop is `token2id["#0"]` and the output label is `word2id["#0"]`.
    Returns:
      Return an instance of `k2.Fsa` representing the given lexicon.
    """
    loop_state = 0  # words enter and leave from here
    next_state = 1  # the next un-allocated state, will be incremented as we go

    arcs = []

    # The blank symbol <blk> is defined in local/train_bpe_model.py
    assert token2id["<blk>"] == 0
    assert word2id["<eps>"] == 0

    eps = 0

    for word, pieces in lexicon:
        assert len(pieces) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        pieces = [token2id[i] for i in pieces]

        for i in range(len(pieces) - 1):
            w = word if i == 0 else eps
            arcs.append([cur_state, next_state, pieces[i], w, 0])

            cur_state = next_state
            next_state += 1

        # now for the last piece of this word
        i = len(pieces) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, pieces[i], w, 0])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs,
            disambig_token=disambig_token,
            disambig_word=disambig_word,
        )

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa

def add_disambig_symbols(lexicon: LexiconBPE) -> Tuple[LexiconBPE, int]:
    """It adds pseudo-token disambiguation symbols #1, #2 and so on
    at the ends of tokens to ensure that all pronunciations are different,
    and that none is a prefix of another.

    See also add_lex_disambig.pl from kaldi.

    Args:
      lexicon:
        It is returned by :func:`read_lexicon`.
    Returns:
      Return a tuple with two elements:

        - The output lexicon with disambiguation symbols
        - The ID of the max disambiguation symbol that appears
          in the lexicon
    """

    # (1) Work out the count of each token-sequence in the
    # lexicon.
    count = defaultdict(int)
    for _, tokens in lexicon:
        count[" ".join(tokens)] += 1

    # (2) For each left sub-sequence of each token-sequence, note down
    # that it exists (for identifying prefixes of longer strings).
    issubseq = defaultdict(int)
    for _, tokens in lexicon:
        tokens = tokens.copy()
        tokens.pop()
        while tokens:
            issubseq[" ".join(tokens)] = 1
            tokens.pop()

    # (3) For each entry in the lexicon:
    # if the token sequence is unique and is not a
    # prefix of another word, no disambig symbol.
    # Else output #1, or #2, #3, ... if the same token-seq
    # has already been assigned a disambig symbol.
    ans = []

    # We start with #1 since #0 has its own purpose
    first_allowed_disambig = 1
    max_disambig = first_allowed_disambig - 1
    last_used_disambig_symbol_of = defaultdict(int)

    for word, tokens in lexicon:
        tokenseq = " ".join(tokens)
        assert tokenseq != ""
        if issubseq[tokenseq] == 0 and count[tokenseq] == 1:
            ans.append((word, tokens))
            continue

        cur_disambig = last_used_disambig_symbol_of[tokenseq]
        if cur_disambig == 0:
            cur_disambig = first_allowed_disambig
        else:
            cur_disambig += 1

        if cur_disambig > max_disambig:
            max_disambig = cur_disambig
        last_used_disambig_symbol_of[tokenseq] = cur_disambig
        tokenseq += f" #{cur_disambig}"
        ans.append((word, tokenseq.split()))
    return ans, max_disambig

def add_self_loops(
    arcs: List[List[Any]], disambig_token: int, disambig_word: int
) -> List[List[Any]]:
    """Adds self-loops to states of an FST to propagate disambiguation symbols
    through it. They are added on each state with non-epsilon output symbols
    on at least one arc out of the state.

    See also fstaddselfloops.pl from Kaldi. One difference is that
    Kaldi uses OpenFst style FSTs and it has multiple final states.
    This function uses k2 style FSTs and it does not need to add self-loops
    to the final state.

    The input label of a self-loop is `disambig_token`, while the output
    label is `disambig_word`.

    Args:
      arcs:
        A list-of-list. The sublist contains
        `[src_state, dest_state, label, aux_label, score]`
      disambig_token:
        It is the token ID of the symbol `#0`.
      disambig_word:
        It is the word ID of the symbol `#0`.

    Return:
      Return new `arcs` containing self-loops.
    """
    states_needs_self_loops = set()
    for arc in arcs:
        src, dst, ilabel, olabel, score = arc
        if olabel != 0:
            states_needs_self_loops.add(src)

    ans = []
    for s in states_needs_self_loops:
        ans.append([s, s, disambig_token, disambig_word, 0])

    return arcs + ans

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

def write_mapping(filename: str, sym2id: Dict[str, int]) -> None:
    """Write a symbol to ID mapping to a file.

    Note:
      No need to implement `read_mapping` as it can be done
      through :func:`k2.SymbolTable.from_file`.

    Args:
      filename:
        Filename to save the mapping.
      sym2id:
        A dict mapping symbols to IDs.
    Returns:
      Return None.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for sym, i in sym2id.items():
            f.write(f"{sym} {i}\n")

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
        "--build-words", "-w",
        action="store_true",
        default=True,  # true by default since this is the only reason this file has a cli
        help="If provided we will build the words.txt file.",
    )
    parser.add_argument(
        "--build-lexicon", "-l",
        action="store_true",
        default=False,
        help="If provided we will build the tokenizer directory (containing L, L_inv, tokens.txt, e.t.c.).",
    )
    parser.add_argument(
        "--hparams-file",
        type=str,
        help="Path to the hyperparameters file. Required if --build_lexicon is true",
        required=False,
    )
    # add the output-recipe-dir argument
    parser.add_argument(
        "--output-recipe-dir",
        type=str,
        help="Path to the output folder. Required if --build_lexicon is true",
        required=False,
    )
    args = parser.parse_args()
    if args.build_lexicon:
        assert os.path.isfile(args.hparams_file), f"{args.hparams_file} does not exist."
        with open(args.hparams_file) as fin:
            hparams = load_hyperpyyaml(
                fin,
                overrides={
                    "data_folder": args.librispeech_dir,
                    "output_folder": args.output_recipe_dir,
                }
            )
        lex = LexiconBPE(
            tokenizer_model_path=args.bpe_model_path,
            librispeech_dir=args.librispeech_dir,
            train_sets=ast.literal_eval(args.train_sets),
            tokenizer=hparams["tokenizer"].sp,  # dummy tokenizer
        )
        lex.build_lexicon()
    elif args.build_words:
        lex = LexiconBPE(
            tokenizer_model_path=args.bpe_model_path,
            librispeech_dir=args.librispeech_dir,
            train_sets=ast.literal_eval(args.train_sets),
            tokenizer=spm.SentencePieceProcessor(),  # dummy tokenizer
        )
        lex.words_path  # create the words.txt file