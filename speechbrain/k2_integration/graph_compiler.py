"""Graph compiler class to create, store, and use k2 decoding graphs in
speechbrain. The addition of a decoding graph, be it HL or HLG (with LM),
limits the output words to the ones in the lexicon. On top of that, a
bigger LM can be used to rescore the decoding graph and get better results.

This code is an extension, and therefore heavily inspired or taken from
icefall's (https://github.com/k2-fsa/icefall) graph compiler.

Authors:
  * Zeyu Zhao 2023
  * Georgios Karakasidis 2023
"""


import os
from pathlib import Path
from typing import List, Union, Optional

from . import k2 # import k2 from ./__init__.py

import torch
import logging

from . import lexicon, utils, lattice_decode


logger = logging.getLogger(__name__)


class CharCtcTrainingGraphCompiler(object):
    """This class is used to compile decoding graphs for CTC training.

    Arguments
    ---------
    lexicon: Lexicon
        It is built from `data/lang/lexicon.txt`.
    device: torch.device
        The device to use for operations compiling transcripts to FSAs.
    oov: str
        Out of vocabulary word. When a word in the transcript
        does not exist in the lexicon, it is replaced with `oov`.
    need_repeat_flag: bool
        If True, will add an attribute named `_is_repeat_token_` to ctc_topo
        indicating whether this token is a repeat token in ctc graph.
        This attribute is needed to implement delay-penalty for phone-based
        ctc loss. See https://github.com/k2-fsa/k2/pull/1086 for more
        details. Note: The above change MUST be included in k2 to enable this
        flag so make sure you have an up-to-date version.
    """

    # G_path: str
    #     Path to the language model FST to be used in the decoding-graph creation.
    #     If None, then we assume that the language model is not used.
    # rescoring_lm_path: Path | str
    #     Path to the language model FST to be used in the rescoring of the decoding
    #     graph. If None, then we assume that the language model is not used.
        # G_path: Union[Path, str, None] = None,
        # rescoring_lm_path: Union[Path, str, None] = None,


    def __init__(
        self,
        lexicon: lexicon.Lexicon,
        device: torch.device,
        oov: str = "<UNK>",
        need_repeat_flag: bool = False,
    ):
        L_inv = lexicon.L_inv.to(device)
        L = lexicon.L.to(device)
        self.lexicon = lexicon
        assert L_inv.requires_grad is False

        assert oov in lexicon.word_table

        self.L_inv = k2.arc_sort(L_inv)
        self.L = k2.arc_sort(L)
        self.oov_id = lexicon.word_table[oov]
        self.word_table = lexicon.word_table

        max_token_id = max(lexicon.tokens)
        ctc_topo = k2.ctc_topo(max_token_id, modified=False)

        self.ctc_topo = ctc_topo.to(device)

        if need_repeat_flag:
            self.ctc_topo._is_repeat_token_ = (
                self.ctc_topo.labels != self.ctc_topo.aux_labels
            )

        self.device = device

    def compile(self, texts: List[str]) -> k2.Fsa:
        """Build decoding graphs by composing ctc_topo with
        given transcripts.

        Arguments
        ---------
        texts: List[str]
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns
        -------
        decoding_graph:
            An FsaVec, the composition result of `self.ctc_topo` and the
            transcript FSA.
        """
        print(texts, self.word_table, self.oov_id)
        word_ids_list = utils.texts_to_ids(texts,
                                           self.word_table,
                                           self.oov_id)
        word_fsa = k2.linear_fsa(word_ids_list, self.device)

        word_fsa_with_self_loops = k2.add_epsilon_self_loops(word_fsa)

        fsa = k2.intersect(
            self.L_inv, word_fsa_with_self_loops, treat_epsilons_specially=False
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

        decoding_graph = k2.compose(
            self.ctc_topo, fsa_with_self_loops, treat_epsilons_specially=False
        )

        assert decoding_graph.requires_grad is False

        return decoding_graph

    def decode(
        self,
        log_probs: torch.Tensor,
        input_lens: torch.Tensor,
        search_beam=5,
        output_beam=5,
        ac_scale=1.0,
        min_active_states=300,
        max_active_states=1000,
        is_test: bool = True,
        decoding_method: str = "1best",
        lm_scale: Optional[float] = None,
        rescoring_lm_path: Optional[Path] = None,
    ) -> List[str]:
        """
        Decode the given log_probs with self.decoding_graph without language model.

        Arguments
        ---------
        log_probs: torch.Tensor
            It is an input tensor of shape (batch, seq_len, num_tokens).
        input_lens: torch.Tensor
            It is an int tensor of shape (batch,). It contains lengths of
            each sequence in `log_probs`.
        search_beam: int
            decoding beam size
        output_beam: int
            lattice beam size
        ac_scale: float
            acoustic scale applied to `log_probs`
        min_active_states: int
            minimum #states that are not pruned during decoding
        max_active_states: int
            maximum #active states that are kept during decoding
        is_test: bool
            if testing is performed then we won't log warning about <UNK>s.
        decoding_method: str
            one of 1best, whole-lattice-rescoring, or nbest.
        lm_scale: float
            cale factor for rescoring with an LM. Defaults to [0.4].
        rescoring_lm_path: Path
            path to the LM to be used for rescoring. If not provided
            and the decoding method is whole-lattice-rescoring, then you need to provide
            the `rescoring_lm_path` in the constructor of this class.

        Returns
        -------
        A list of strings, each of which is the decoding result of the
        corresponding utterance.
        """
        lm_scale = lm_scale or 0.4
        device = log_probs.device
        if self.decoding_graph is None:
            if is_test:
                pass
                # self.lexicon.log_unknown_warning = False
            if self.G_path is None:
                self.compile_HL()
            else:
                logger.info("Compiling HLG instead of HL")
                self.compile_HLG()
            if self.decoding_graph.device != device:
                self.decoding_graph = self.decoding_graph.to(device)
            if decoding_method == "whole-lattice-rescoring":

                logger.info(f"Loading rescoring LM: {path}")
                G = utils.load_G(rescoring_lm_path, device="cpu")
                del G.aux_labels
                G.labels[G.labels >= self.lexicon.word_table["#0"]] = 0
                G = utils.prepare_G(G, device)

                self.prepare_G(
                    rescoring_lm_path, device
                )
        input_lens = input_lens.to(device)

        input_lens = (input_lens * log_probs.shape[1]).round().int()
        # NOTE: low ac_scales may results in very big lattices and OOM errors.
        log_probs *= ac_scale

        with torch.no_grad():
            lattice = k2.get_lattice(
                log_probs,
                input_lens,
                self.decoding_graph,
                search_beam=search_beam,
                output_beam=output_beam,
                min_active_states=min_active_states,
                max_active_states=max_active_states,
            )
            if decoding_method == "1best":
                key = "no_rescore"
                best_path = {
                    key: lattice_decode.one_best_decoding(
                        lattice=lattice, use_double_scores=True
                    )
                }
                out = utils.lattice_to_text(best_path[key], self.word_table)
            elif decoding_method == "whole-lattice-rescoring":
                best_path = lattice_decode.rescore_with_whole_lattice(
                    lattice=lattice.to(self.device),
                    G_with_epsilon_loops=self.rescoring_graph,
                    lm_scale_list=[lm_scale],
                    use_double_scores=True,
                )
                out = utils.lattice_to_text(best_path[f"lm_scale_{lm_scale:.1f}"], self.word_table)
            else:
                raise ValueError(
                    f"Decoding method '{decoding_method}' not supported."
                )
            del lattice
            del best_path
            torch.cuda.empty_cache()

            return out


def ctc_compile_HL(H, L):
    """
    Compile the decoding graph by composing H with L.
    This is for decoding without language model.
    Usually, you don't need to call this function explicitly.
    """
    logger.info("Arc sorting L")
    L = k2.arc_sort(L).to("cpu")
    H = H.to("cpu")
    logger.info("Composing H and L")
    HL = k2.compose(H, L, inner_labels="tokens")

    logger.info("Connecting HL")
    HL = k2.connect(HL)

    logger.info("Arc sorting HL")
    logger.info("Done compiling HL")
    return k2.arc_sort(HL)


def compile_HLG(H, L, G):
    """
    Compile the decoding graph by composing H with LG.
    This is for decoding with language model.
    Usually, you don't need to call this function explicitly.
    """
    L = k2.arc_sort(L).to("cpu")
    G = k2.arc_sort(G).to("cpu")
    logger.debug("Intersecting L and G")
    LG = k2.compose(L, G)

    logger.debug("Connecting LG")
    LG = k2.connect(LG)

    logger.debug("Determinizing LG")
    LG = k2.determinize(LG)

    logger.debug("Connecting LG after k2.determinize")
    LG = k2.connect(LG)

    logger.debug("Removing disambiguation symbols on LG")
    # NOTE: We need to clone here since LG.labels is just a reference to a tensor
    #       and we will end up having issues with misversioned updates on fsa's
    #       properties.
    labels = LG.labels.clone()
    labels[labels >= first_token_disambig_id] = 0
    LG.labels = labels

    assert isinstance(LG.aux_labels, k2.RaggedTensor)
    LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

    LG = k2.remove_epsilon(LG)

    LG = k2.connect(LG)
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)
    logger.debug("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logger.debug("Composing H and LG")
    HLG = k2.compose(H, LG, inner_labels="tokens")

    logger.debug("Connecting HLG")
    HLG = k2.connect(HLG)

    logger.debug("Arc sorting HLG")
    return k2.arc_sort(HLG)
