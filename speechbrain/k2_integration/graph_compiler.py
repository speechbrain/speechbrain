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


from typing import List

import k2
import torch
import logging

from speechbrain.k2_integration.lexicon import Lexicon
from speechbrain.k2_integration.utils import get_texts


class CtcTrainingGraphCompiler(object):
    def __init__(
        self,
        lexicon: Lexicon,
        device: torch.device,
        oov: str = "<UNK>",
        need_repeat_flag: bool = False,
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
        self.HL = None

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
               min_active_states=30,
               max_active_states=1000) -> List[str]:
        """
        Decode the given log_probs with self.HL without language model.

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

        Returns:
        texts: a list of strings, each of which is the decoding result of the corresponding utterance.
        """
        if self.HL is None:
            self.compile_HL()
        device = log_probs.device
        if self.HL.device != device:
            self.HL = self.HL.to(device)
        input_lens = input_lens.to(device)

        input_lens = (input_lens * log_probs.shape[1]).round().int()
        log_probs *= ac_scale

        with torch.no_grad():
            lattice = k2.get_lattice(
                log_probs,
                input_lens,
                self.HL,
                search_beam=search_beam,
                output_beam=output_beam,
                min_active_states=min_active_states,
                max_active_states=max_active_states,
            )
            one_best = k2.shortest_path(lattice, use_double_scores=True)
            list_wids = get_texts(one_best, return_ragged=False)
            texts = []
            for wids in list_wids:
                texts.append(" ".join([self.word_table[wid]
                             for wid in wids]))
            del lattice
            del one_best
            torch.cuda.empty_cache()
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
        self.HL = k2.arc_sort(HL)
        logging.info("Done compiling HL")
