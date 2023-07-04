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


# Adjusted from: icefall.graph_compiler.CtcTrainingGraphCompiler

from typing import List

import k2
import torch

from icefall.lexicon import Lexicon


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
        assert L_inv.requires_grad is False

        assert oov in lexicon.word_table

        self.L_inv = k2.arc_sort(L_inv)
        self.oov_id = lexicon.word_table[oov]
        self.word_table = lexicon.word_table
        self.tokenizer = lexicon.tokenizer

        max_token_id = max(lexicon.tokens)
        ctc_topo = k2.ctc_topo(max_token_id, modified=False)
        try:
          self.ctc_topo = ctc_topo.to(device)
        except torch.cuda.OutOfMemoryError as e:
          print("ctc_topo.to(device={}) failed, try to use CPU instead. max_token_id =".format(device), max_token_id)
          raise e

        if need_repeat_flag:
            self.ctc_topo._is_repeat_token_ = (
                self.ctc_topo.labels != self.ctc_topo.aux_labels
            )

        self.device = device

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
        fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(transcript_fsa)

        fsa_with_self_loops = k2.arc_sort(fsa_with_self_loops)

        decoding_graph = k2.compose(
            self.ctc_topo, fsa_with_self_loops, treat_epsilons_specially=False
        )

        assert decoding_graph.requires_grad is False

        return decoding_graph
    
    def tokenize(self, text: str) -> List[int]:
        # Convert to list of list of ids and then flatten
        return [
                idx for sublist in self.tokenizer.encode(text.split(), out_type=int) \
                    for idx in sublist
        ]

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
        # word_ids_list = [self.tokenize(text) for text in texts]

        word_fsa = k2.linear_fsa(word_ids_list, self.device)

        word_fsa_with_self_loops = k2.add_epsilon_self_loops(word_fsa)

        fsa = k2.intersect(
            self.L_inv, word_fsa_with_self_loops, treat_epsilons_specially=False
        )
        # fsa has word ID as labels and token ID as aux_labels, so
        # we need to invert it
        ans_fsa = fsa.invert_()
        return k2.arc_sort(ans_fsa)
