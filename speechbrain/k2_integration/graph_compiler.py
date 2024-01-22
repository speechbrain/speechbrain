"""Graph compiler class to create, store, and use k2 decoding graphs in
speechbrain. Limits the output words to the ones in the lexicon.

This code is an extension, and therefore heavily inspired or taken from
icefall's (https://github.com/k2-fsa/icefall) graph compiler.

Authors:
  * Pierre Champion 2023
  * Zeyu Zhao 2023
  * Georgios Karakasidis 2023
"""


import os
from typing import List, Optional, Tuple
import abc
import torch
import logging

from . import k2  # import k2 from ./__init__.py
from . import lexicon

logger = logging.getLogger(__name__)


class GraphCompiler(abc.ABC):
    """
    This abstract class is used to compile graphs for training and decoding.
    """

    @abc.abstractproperty
    def topo(self) -> k2.Fsa:
        """
        Return the topology used to compile the graph.
        """
        pass

    @abc.abstractproperty
    def lexicon(self) -> lexicon.Lexicon:
        """
        Return the lexicon used to compile the graph.
        """
        pass

    @abc.abstractproperty
    def device(self):
        """
        Return the device used to compile the graph.
        """
        pass

    @abc.abstractmethod
    def compile(
        self, texts: List[str], is_training: bool = True
    ) -> Tuple[k2.Fsa, torch.Tensor]:
        """
        Compile the graph for the given texts.

        Arguments
        ---------
        texts: List[str]
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello world', 'CTC training with k2']

        is_training: bool
            Indictating whether this is for training or not
            (OOV warning in training).
        Returns
        -------
        graph: GraphCompiler
            An FsaVec, the composition result of `self.ctc_topo` and the
            transcript FSA.
        target_lens: Torch.tensor
            It is an long tensor of shape (batch,). It contains lengths of
            each target sequence.
        """
        pass

    def compile_HL(self, cache_dir: Optional[str] = None, cache: bool = False):
        """
        Compile the decoding graph by composing H with L.
        This is for decoding without language model.

        Arguments
        ---------
        cache_dir: str
            The path to store the composition in a .pt format.
        cache: bool
            Whether or not to load the composition from the .pt format (in the
            cache_dir dir).

        Returns
        -------
        HL: k2.Fsa
            The HL composition
        """
        logger.info("Arc sorting L")
        L = k2.arc_sort(self.lexicon.L).to("cpu")
        H = self.topo.to("cpu")

        file_hash = str(hash(H.shape[0])) + str(hash(L.shape[0]))
        if cache and cache_dir is not None:
            path = cache_dir + "/.HL_" + file_hash + ".pt"
            if os.path.exists(path):
                logger.warning(
                    f"Loading HL '{path}' from its cached .pt format."
                    " Set 'caching: False' in the yaml"
                    " if this is not what you want."
                )
                HL = k2.Fsa.from_dict(torch.load(path, map_location="cpu"))
                return HL

        logger.info("Composing H and L")
        HL = k2.compose(H, L, inner_labels="tokens")

        logger.info("Connecting HL")
        HL = k2.connect(HL)

        logger.info("Arc sorting HL")
        HL = k2.arc_sort(HL)
        logger.debug(f"HL.shape: {HL.shape}")

        if cache_dir is not None:
            path = cache_dir + "/.HL_" + file_hash + ".pt"
            logger.info("Caching HL to: " + path)
            torch.save(HL.as_dict(), path)

        return HL

    def compile_HLG(
        self, G, cache_dir: Optional[str] = None, cache: bool = False
    ):
        """
        Compile the decoding graph by composing H with LG.
        This is for decoding with small language model.

        Arguments
        ---------
        G: k2.Fsa
            The language model FSA.
        cache_dir: str
            The path to store the composition in a .pt format.
        cache: bool
            Whether or not to load the composition from the .pt format (in the
            cache_dir dir).

        Returns
        -------
        HL: k2.Fsa
            The HLG composition
        """
        logger.info("Arc sorting L")
        L = k2.arc_sort(self.lexicon.L_disambig).to("cpu")
        G = k2.arc_sort(G).to("cpu")
        H = self.topo.to("cpu")

        file_hash = (
            str(hash(H.shape[0]))
            + str(hash(L.shape[0]))
            + str(hash(G.shape[0]))
        )
        if cache and cache_dir is not None:
            path = cache_dir + "/.HLG_" + file_hash + ".pt"
            if os.path.exists(path):
                logger.warning(
                    f"Loading HLG '{path}' from its cached .pt format."
                    " Set 'caching: False' in the yaml"
                    " if this is not what you want."
                )
                HLG = k2.Fsa.from_dict(torch.load(path, map_location="cpu"))
                return HLG

        logger.info("Intersecting L and G")
        LG = k2.compose(L, G)

        logger.info("Connecting LG")
        LG = k2.connect(LG)

        logger.info("Determinizing LG")
        LG = k2.determinize(LG)

        logger.info("Connecting LG after k2.determinize")
        LG = k2.connect(LG)
        LG = self.lexicon.remove_LG_disambig_symbols(LG)

        LG = k2.remove_epsilon(LG)

        LG = k2.connect(LG)
        LG.aux_labels = LG.aux_labels.remove_values_eq(0)
        logger.info("Arc sorting LG")
        LG = k2.arc_sort(LG)

        logger.info("Composing H and LG")
        HLG = k2.compose(H, LG, inner_labels="tokens")

        logger.info("Connecting HLG")
        HLG = k2.connect(HLG)

        logger.info("Arc sorting HLG")
        HLG = k2.arc_sort(HLG)
        logger.debug(f"HLG.shape: {HLG.shape}")

        if cache_dir is not None:
            path = cache_dir + "/.HLG_" + file_hash + ".pt"
            logger.info("Caching HLG to: " + path)
            torch.save(HLG.as_dict(), path)

        return HLG


class CtcGraphCompiler(GraphCompiler):
    """
    This class is used to compile decoding graphs for CTC training.

    Arguments
    ---------
    lexicon: Lexicon
        It is built from `data/lang/lexicon.txt`.
    device: torch.device
        The device to use for operations compiling transcripts to FSAs.
    need_repeat_flag: bool
        If True, will add an attribute named `_is_repeat_token_` to ctc_topo
        indicating whether this token is a repeat token in ctc graph.
        This attribute is needed to implement delay-penalty for phone-based
        ctc loss. See https://github.com/k2-fsa/k2/pull/1086 for more
        details. Note: The above change MUST be included in k2 to enable this
        flag so make sure you have an up-to-date version.

    Example
    -------
    >>> import torch
    >>> from speechbrain.k2_integration.losses import ctc_k2
    >>> from speechbrain.k2_integration.graph_compiler import CtcGraphCompiler
    >>> from speechbrain.k2_integration.lexicon import Lexicon
    >>> from speechbrain.k2_integration.prepare_lang import prepare_lang

    >>> # Create a random batch of log-probs
    >>> batch_size = 4

    >>> log_probs = torch.randn(batch_size, 100, 30)
    >>> log_probs.requires_grad = True
    >>> # Assume all utterances have the same length so no padding was needed.
    >>> input_lens = torch.ones(batch_size)
    >>> # Create a samll lexicon containing only two words and write it to a file.
    >>> lang_tmpdir = getfixture('tmpdir')
    >>> lexicon_sample = "hello h e l l o\\nworld w o r l d\\n<UNK> <unk>"
    >>> lexicon_file = lang_tmpdir.join("lexicon.txt")
    >>> lexicon_file.write(lexicon_sample)
    >>> # Create a lang directory with the lexicon and L.pt, L_inv.pt, L_disambig.pt
    >>> prepare_lang(lang_tmpdir)
    >>> # Create a lexicon object
    >>> lexicon = Lexicon(lang_tmpdir)
    >>> # Create a random decoding graph
    >>> graph = CtcGraphCompiler(
    ...     lexicon,
    ...     log_probs.device,
    ... )
    >>> isinstance(graph.topo, k2.Fsa)
    True

    """

    def __init__(
        self,
        _lexicon: lexicon.Lexicon,
        device: torch.device,
        need_repeat_flag: bool = False,
    ):
        self._device = device

        self._lexicon = _lexicon
        self.lexicon.to(device)
        assert self.lexicon.L_inv.requires_grad is False
        self.lexicon.arc_sort()

        max_token_id = max(self.lexicon.tokens)
        ctc_topo = k2.ctc_topo(max_token_id, modified=False)

        self.ctc_topo = ctc_topo.to(device)

        if need_repeat_flag:
            self.ctc_topo._is_repeat_token_ = (
                self.ctc_topo.labels != self.ctc_topo.aux_labels
            )

    @property
    def topo(self):
        """
        Return the ctc_topo.
        """
        return self.ctc_topo

    @property
    def lexicon(self):
        """
        Return the lexicon.
        """
        return self._lexicon

    @property
    def device(self):
        """Return the device used for compiling graphs."""
        return self._device

    def compile(
        self, texts: List[str], is_training: bool = True
    ) -> Tuple[k2.Fsa, torch.Tensor]:
        """
        Build decoding graphs by composing ctc_topo with given transcripts.

        Arguments
        ---------
        texts: List[str]
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello world', 'CTC training with k2']

        is_training: bool
            Indictating whether this is for training or not
            (OOV warning in training).

        Returns
        -------
        graph: GraphCompiler
            An FsaVec, the composition result of `self.ctc_topo` and the
            transcript FSA.
        target_lens: Torch.tensor
            It is an long tensor of shape (batch,). It contains lengths of
            each target sequence.
        """

        word_idx = self.lexicon.texts_to_word_ids(
            texts, log_unknown_warning=is_training
        )

        # ["test", "testa"] -> [[23, 8, 22, 23], [23, 8, 22, 23, 5]] -> [4, 5]
        word2tids = self.lexicon.texts_to_token_ids(
            texts, log_unknown_warning=is_training
        )
        scentence_ids = [sum(inner, []) for inner in word2tids]

        target_lens = torch.tensor(
            [len(t) for t in scentence_ids], dtype=torch.long
        )

        word_fsa_with_self_loops = k2.add_epsilon_self_loops(
            k2.linear_fsa(word_idx, self.device)
        )

        fsa = k2.intersect(
            self.lexicon.L_inv,
            word_fsa_with_self_loops,
            treat_epsilons_specially=False,
        )
        # fsa has word ID as labels and token ID as aux_labels, so
        # we need to invert it
        ans_fsa = fsa.invert_()
        transcript_fsa = k2.arc_sort(ans_fsa)

        # NOTE: k2.compose runs on CUDA only when treat_epsilons_specially
        # is False, so we add epsilon self-loops here
        fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(
            transcript_fsa
        )

        fsa_with_self_loops = k2.arc_sort(fsa_with_self_loops)

        graph = k2.compose(
            self.ctc_topo, fsa_with_self_loops, treat_epsilons_specially=False
        )

        assert graph.requires_grad is False

        return graph, target_lens
