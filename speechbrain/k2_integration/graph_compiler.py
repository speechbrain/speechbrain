"""Graph compiler class to create, store, and use k2 decoding graphs in
speechbrain. The addition of a decoding graph, be it HL or HLG (with LM),
limits the output words to the ones in the lexicon. On top of that, a
bigger LM can be used to rescore the decoding graph and get better results.

This code is an extension of icefall's (https://github.com/k2-fsa/icefall)
graph compiler.

Authors:
  * Zeyu Zhao 2023
  * Georgios Karakasidis 2023
"""


import os
from pathlib import Path
from typing import Dict, List, Union, Optional

try:
    import k2
except ImportError:
    MSG = "Please install k2 to use k2 training \n"
    MSG += "E.G. run: pip install k2\n"
    MSG += "or if the extra_requirements.txt file exists in your recipe \n"
    MSG += "install them with: pip install -r extra_requirements.txt"
    raise ImportError(MSG)
import torch
import logging

from speechbrain.k2_integration.lexicon import Lexicon
from speechbrain.k2_integration.utils import get_texts, one_best_decoding, rescore_with_whole_lattice


logger = logging.getLogger(__name__)


class CtcTrainingGraphCompiler(object):
    def __init__(
        self,
        lexicon: Lexicon,
        device: torch.device,
        oov: str = "<UNK>",
        need_repeat_flag: bool = False,
        G_path: Optional[str] = None,
        rescoring_lm_path: Union[Path, str, None] = None,
    ):
        """
        Args:
          lexicon:
            It is built from `data/lang/lexicon.txt`.
          device:
            The device to use for operations compiling transcripts to FSAs.
          oov:
            Out of vocabulary word. When a word in the transcript
            does not exist in the lexicon, it is replaced with `oov`.
          need_repeat_flag:
            If True, will add an attribute named `_is_repeat_token_` to ctc_topo
            indicating whether this token is a repeat token in ctc graph.
            This attribute is needed to implement delay-penalty for phone-based
            ctc loss. See https://github.com/k2-fsa/k2/pull/1086 for more
            details. Note: The above change MUST be included in k2 to enable this
            flag.
          G_path: str
            Path to the language model FST to be used in the decoding-graph creation.
            If None, then we assume that the language model is not used.
          rescoring_lm_path: Path | str
            Path to the language model FST to be used in the rescoring of the decoding
            graph. If None, then we assume that the language model is not used.
        """
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
        self.G_path: str = G_path
        self.rescoring_lm_path: Path = rescoring_lm_path
        self.decoding_graph: k2.Fsa = None  # HL or HLG
        self.rescoring_graph: k2.Fsa = None  # G (usually 4-gram LM)

    def get_G(self, path: str = None, save: bool = True) -> k2.Fsa:
        """Load a LM to be used in the decoding graph creation (or LM rescoring).
        Note that it doesn't load G into memory.

        Args:
            path: str, The path to an FST LM (ending with .fst.txt) or a k2-converted
                LM (in pytorch .pt format).
            save: bool, Whether or not to save the LM in .pt format (in the same dir).
        
        Returns:
            An FSA representing the LM. The device is the same as graph_compiler.device.
        """
        path = str(path or self.G_path)
        if os.path.exists(path.replace(".fst.txt", ".pt")):
            logger.info(f"NOTE: Loading G from its .pt format")
            path = path.replace(".fst.txt", ".pt")
        # If G_path is an fst.txt file then convert to .pt file
        if path.endswith(".fst.txt"):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File {path} not found. You need to run the kaldilm to get it.")
            with open(path) as f:
                G = k2.Fsa.from_openfst(f.read(), acceptor=False).to(self.device)
        elif path.endswith(".pt"):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File {path} not found.")
            d = torch.load(path, map_location=self.device)
            G = k2.Fsa.from_dict(d).to(self.device)
        else:
            raise ValueError(f"File {path} is not a .fst.txt or .pt file.")
        if save:
            torch.save(G.as_dict(), path[:-8] + ".pt")
        return G

    def get_rescoring_LM(self, path: str = None) -> k2.Fsa:
        """Load a LM with the purpose of using it for LM rescoring.
        For instance, in the librispeech recipe this is a 4-gram LM (while  a
        3gram LM is used for HLG construction).

        Args:
            path: str, The path to an FST LM (ending with .fst.txt) or a k2-converted
                LM (in pytorch .pt format).

        Returns:
            An FSA representing the LM. The device is the same as graph_compiler.device.
        """
        path = str(path or self.rescoring_lm_path)
        logger.info(f"Loading rescoring LM: {path}")
        G = self.get_G(path, save=False).to("cpu")
        del G.aux_labels
        G.labels[G.labels >= self.lexicon.word_table["#0"]] = 0
        G.__dict__["_properties"] = None
        G = k2.Fsa.from_fsas([G]).to('cpu')  # only used for decoding which is done in cpu
        G = k2.arc_sort(G)
        G = k2.add_epsilon_self_loops(G)
        G = k2.arc_sort(G)
        G = G.to(self.device)
        # G.lm_scores is used to replace HLG.lm_scores during
        # LM rescoring.
        if not hasattr(G, "lm_scores"):
            G.lm_scores = G.scores.clone()
        return G

    def compile(self, texts: List[str]) -> k2.Fsa:
        """Build decoding graphs by composing ctc_topo with
        given transcripts.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          An FsaVec, the composition result of `self.ctc_topo` and the
          transcript FSA.
        """
        transcript_fsa = self.convert_transcript_to_fsa(texts)

        # NOTE: k2.compose runs on CUDA only when treat_epsilons_specially
        # is False, so we add epsilon self-loops here
        fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(
            transcript_fsa)

        fsa_with_self_loops = k2.arc_sort(fsa_with_self_loops)

        decoding_graph = k2.compose(
            self.ctc_topo, fsa_with_self_loops, treat_epsilons_specially=False
        )

        assert decoding_graph.requires_grad is False

        return decoding_graph

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of word IDs.

        Args:
          texts:
            It is a list of strings. Each string consists of space(s)
            separated words. An example containing two strings is given below:

                ['HELLO ICEFALL', 'HELLO k2']
        Returns:
          Return a list-of-list of word IDs.
        """
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split():
                if word in self.word_table:
                    word_ids.append(self.word_table[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)
        return word_ids_list

    def convert_transcript_to_fsa(self, texts: List[str]) -> k2.Fsa:
        """Convert a list of transcript texts to an FsaVec.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          Return an FsaVec, whose `shape[0]` equals to `len(texts)`.
        """
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split():
                if word in self.word_table:
                    word_ids.append(self.word_table[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)

        word_fsa = k2.linear_fsa(word_ids_list, self.device)

        word_fsa_with_self_loops = k2.add_epsilon_self_loops(word_fsa)

        fsa = k2.intersect(
            self.L_inv, word_fsa_with_self_loops, treat_epsilons_specially=False
        )
        # fsa has word ID as labels and token ID as aux_labels, so
        # we need to invert it
        ans_fsa = fsa.invert_()
        return k2.arc_sort(ans_fsa)

    def decode(self,
               log_probs: torch.Tensor,
               input_lens: torch.Tensor,
               search_beam=5,
               output_beam=5,
               ac_scale=1.0,
               min_active_states=300,
               max_active_states=1000,
               is_test: bool = True,
               decoding_method: str = "1best",
               lm_scale_list: Optional[List[float]] = None,
               rescoring_lm_path: Optional[Path] = None
        ) -> Union[List[str], Dict[str, List[str]]]:
        """
        Decode the given log_probs with self.decoding_graph without language model.

        Args:
          log_probs:
            It is an input tensor of shape (batch, seq_len, num_tokens).
          input_lens:
            It is an int tensor of shape (batch,). It contains lengths of
            each sequence in `log_probs`.
          search_beam: int, decoding beam size
          output_beam: int, lattice beam size
          ac_scale: float, acoustic scale applied to `log_probs`
          min_active_states: int, minimum #states that are not pruned during decoding
          max_active_states: int, maximum #active states that are kept during decoding
          is_test: bool, if testing is performed then we won't log warning about <UNK>s.
          decoding_method: str, one of 1best, whole-lattice-rescoring, or nbest.
          lm_scale_list: List[float], a list of language model scale factors. Defaults to [0.6].
          rescoring_lm_path: Path, path to the LM to be used for rescoring. If not provided
            and the decoding method is whole-lattice-rescoring, then you need to provide
            the `rescoring_lm_path` in the constructor of this class.          

        Returns:
          If decoding_method==1best: a list of strings, each of which is the decoding 
            result of the corresponding utterance.
          If decoding_method==whole-lattice-rescoring: a dict of lists of strings, each of
            which is the decoding result of the corresponding utterance. The keys of the dict
            are the language model scale factors used for rescoring.
        """
        lm_scale_list = lm_scale_list or [0.6]
        device = log_probs.device
        if self.decoding_graph is None:
            if is_test:
                self.lexicon.log_unknown_warning = False
            if self.G_path is None:
                self.compile_HL()
            else:
                logger.info("Compiling HLG instead of HL")
                self.compile_HLG()
                # if not hasattr(self.decoding_graph, "lm_scores"):
                #     self.decoding_graph.lm_scores = self.decoding_graph.scores.clone()
            if self.decoding_graph.device != device:
                self.decoding_graph = self.decoding_graph.to(device)
            if decoding_method == "whole-lattice-rescoring":
                # fst_4gram_path = str(Path(self.G_path).parent / "G_4_gram.fst.txt")
                # fst_4gram_path = "lm/G_4_gram_withfullwords.fst.txt"
                self.rescoring_graph = self.get_rescoring_LM(rescoring_lm_path).to(self.device)
        input_lens = input_lens.to(device)

        input_lens = (input_lens * log_probs.shape[1]).round().int()
        # NOTE: low ac_scales may results in very big lattices and OOM errors.
        log_probs *= ac_scale

        def lattice2text(best_path: k2.Fsa) -> List[str]:
            """Convert the best path to a list of strings."""
            hyps: List[List[int]] = get_texts(best_path, return_ragged=False)
            texts = []
            for wids in hyps:
                texts.append(" ".join([self.word_table[wid]
                            for wid in wids]))
            return texts

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
                    key: one_best_decoding(
                        lattice=lattice, use_double_scores=True
                    )
                }
                out = lattice2text(best_path[key])
            elif decoding_method == "whole-lattice-rescoring":
                best_path = rescore_with_whole_lattice(
                    lattice=lattice.to(self.device),
                    G_with_epsilon_loops=self.rescoring_graph,
                    lm_scale_list=lm_scale_list,
                    use_double_scores=True,
                )
                out = {}
                for lm_scale in lm_scale_list:
                    key = f"lm_scale_{lm_scale:.1f}"
                    out[key] = lattice2text(best_path[key])
            else:
                raise ValueError(f"Decoding method '{decoding_method}' is not supported.")
            del lattice
            del best_path
            torch.cuda.empty_cache()

            return out

    def compile_HL(self):
        '''
        Compile the decoding graph by composing ctc_topo with L.
        This is for decoding without language model.
        Usually, you don't need to call this function explicitly.
        '''
        logger.info("Arc sorting L")
        L = k2.arc_sort(self.L).to("cpu")
        H = self.ctc_topo.to("cpu")
        logger.info("Composing H and L")
        HL = k2.compose(H, L, inner_labels="tokens")

        logger.info("Connecting HL")
        HL = k2.connect(HL)

        logger.info("Arc sorting HL")
        self.decoding_graph = k2.arc_sort(HL)
        logger.info("Done compiling HL")

    def compile_HLG(self):
        '''
        Compile the decoding graph by composing ctc_topo with LG.
        This is for decoding with language model (by default we assume a 3gram lm).
        Usually, you don't need to call this function explicitly.
        '''
        H = self.ctc_topo.to("cpu")
        G = self.get_G()
        G = G.to("cpu")
        L = self.lexicon.L_disambig.to("cpu")

        first_token_disambig_id = self.lexicon.token_table["#0"]
        first_word_disambig_id = self.lexicon.word_table["#0"]
        
        L = k2.arc_sort(L)
        G = k2.arc_sort(G)
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
        #       and we will end up having issues with misversioned updates on fsa's properties.
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
        HLG = k2.arc_sort(HLG)

        self.decoding_graph = HLG