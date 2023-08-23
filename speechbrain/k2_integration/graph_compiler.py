# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
# Modified 2023 the University of Edinburgh (Zeyu Zhao, Georgios Karakasidis)
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


import os
from pathlib import Path
from typing import List

import k2
import torch
import logging

from speechbrain.k2_integration.lexicon import Lexicon
from speechbrain.k2_integration.utils import get_texts, one_best_decoding, rescore_with_whole_lattice


class CtcTrainingGraphCompiler(object):
    def __init__(
        self,
        lexicon: Lexicon,
        device: torch.device,
        oov: str = "<UNK>",
        need_repeat_flag: bool = False,
        G_path: str = None,
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
            details. Note: The above change MUST be included in k2 to open this
            flag.
          G_path: str
            Path to the language model FST to be used in the decoding-graph creation.
            If None, then we assume that the language model is not used.
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
        self.decoding_graph: k2.Fsa = None  # HL or HLG
        self.rescoring_graph: k2.Fsa = None  # G (usually 4-gram LM)

    def get_G(self, path: str = None, save: bool = True) -> k2.Fsa:
        """Load a LM to be used in the decoding graph creation (or LM rescoring).
        Note that it doesn't load G into memory.

        Args:
            path: str, The path to an FST LM (ending with .fst.txt) or a k2-converted
                LM (in pytorch .pt format). E.g. for librispeech, you can get the
                .fst.txt file by passing your words.txt file to kaldilm along with the
                already available ARPA model (e.g. the 3-gram pruned one).
            save: bool, Whether or not to save the LM in .pt format (in the same dir).
        
        Returns:
            An FSA representing the LM. The device is the same as graph_compiler.device.
        """
        path = path or self.G_path
        # If G_path is an fst.txt file then convert to .pt file
        if path.endswith(".fst.txt"):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File {path} not found. You need to run the kaldilm to acquire it.")
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

    def get_rescoring_LM(self, path: str) -> k2.Fsa:
        """Load a LM with the purpose of using it for LM rescoring.
        In the librispeech recipe this is a 4-gram LM, in contrast to
        the 3-gram LM used for decoding. The 4-gram .fst.txt file can,
        once again, be acquired by running kaldilm on the ARPA 4-gram
        model which can be downloaded from the internet.

        Args:
            path: str, The path to an FST LM (ending with .fst.txt) or a k2-converted
                LM (in pytorch .pt format).

        Returns:
            An FSA representing the LM. The device is the same as graph_compiler.device.
        """
        # TODO: LG Could also be used instead of just G.
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
               decoding_method: str = "1best") -> List[str]:
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
          

        Returns:
          texts: a list of strings, each of which is the decoding result of the corresponding utterance.
        """
        device = log_probs.device
        if self.decoding_graph is None:
            if is_test:
                self.lexicon.log_unknown_warning = False
            if self.G_path is None:
                self.compile_HL()
            else:
                logging.info("Compiling HLG instead of HL")
                self.compile_HLG()
                # if not hasattr(self.decoding_graph, "lm_scores"):
                #     self.decoding_graph.lm_scores = self.decoding_graph.scores.clone()
            if self.decoding_graph.device != device:
                self.decoding_graph = self.decoding_graph.to(device)
            if decoding_method == "whole-lattice-rescoring":
                # fst_4gram_path = str(Path(self.G_path).parent / "G_4_gram.fst.txt")
                fst_4gram_path = "results/train_wav2vec2_char_k2/1112/lang/G_3_gram.fst.txt"
                self.rescoring_graph = self.get_rescoring_LM(fst_4gram_path).to(self.device)
        input_lens = input_lens.to(device)

        input_lens = (input_lens * log_probs.shape[1]).round().int()
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
                    key: one_best_decoding(
                        lattice=lattice, use_double_scores=True
                    )
                }
            elif decoding_method == "whole-lattice-rescoring":
                lm_scale_list = [0.01]  # [0.2, 0.4, 0.6, 0.8, 1.0]
                best_path = rescore_with_whole_lattice(
                    lattice=lattice.to(self.device),
                    G_with_epsilon_loops=self.rescoring_graph,
                    lm_scale_list=lm_scale_list,
                    use_double_scores=True,
                )
                # TODO: Allow multiple lm_scales instead of just 1.
                key = f"lm_scale_{lm_scale_list[0]:.1f}"
            else:
                raise ValueError(f"Decoding method '{decoding_method}' is not supported.")
            hyps: List[List[int]] = get_texts(best_path[key], return_ragged=False)
            # Convert to list of word strings (for each item in the batch)
            # texts: List[List[str]] = [
            #     [self.word_table[wid] for wid in ids] for ids in hyps
            # ]
            texts = []
            for wids in hyps:
                texts.append(" ".join([self.word_table[wid]
                            for wid in wids]))
            del lattice
            del best_path
            # out = {key: texts}
            torch.cuda.empty_cache()

            # TODO: Instead of list of strings we should return a dict
            #       with the keys being the different lm scales (if len >1)
            #       or just "no_rescore" if len == 1.
            return texts

    def compile_HL(self):
        '''
        Compile the decoding graph by composing ctc_topo with L.
        This is for decoding without language model.
        Usually, you don't need to call this function explicitly.
        '''
        logging.info("Arc sorting L")
        L = k2.arc_sort(self.L).to("cpu")
        H = self.ctc_topo.to("cpu")
        logging.info("Composing H and L")
        HL = k2.compose(H, L, inner_labels="tokens")

        logging.info("Connecting HL")
        HL = k2.connect(HL)

        logging.info("Arc sorting HL")
        self.decoding_graph = k2.arc_sort(HL)
        logging.info("Done compiling HL")

    def compile_HLG(self):
        '''
        Compile the decoding graph by composing ctc_topo with L and G.
        This is for decoding with language model (by default we assume a 3gram lm).
        Usually, you don't need to call this function explicitly.
        '''
        H = self.ctc_topo.to("cpu")
        G = self.get_G()
        G = G.to("cpu")
        L = self.L.to("cpu")

        first_token_disambig_id = self.lexicon.token_table["#0"]
        first_word_disambig_id = self.lexicon.word_table["#0"]
        logging.info("Arc sorting L")
        L = k2.arc_sort(L)
        G = k2.arc_sort(G)
        G = G.scores * 0.1
        logging.info("Intersecting L and G")
        LG = k2.compose(L, G)

        logging.info("Connecting LG")
        LG = k2.connect(LG)

        logging.info("Determinizing LG")
        LG = k2.determinize(LG)

        logging.info("Connecting LG after k2.determinize")
        LG = k2.connect(LG)

        logging.info("Removing disambiguation symbols on LG")
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
        logging.info("Arc sorting LG")
        LG = k2.arc_sort(LG)

        logging.info("Composing H and LG")
        # CAUTION: The name of the inner_labels is fixed
        # to `tokens`. If you want to change it, please
        # also change other places in icefall that are using
        # it.
        HLG = k2.compose(H, LG, inner_labels="tokens")

        logging.info("Connecting HLG")
        HLG = k2.connect(HLG)

        logging.info("Arc sorting HLG")
        HLG = k2.arc_sort(HLG)

        self.decoding_graph = HLG