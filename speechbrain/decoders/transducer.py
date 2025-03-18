"""Decoders and output normalization for Transducer sequence.

Author:
    Abdelwahab HEBA 2020
    Sung-Lin Yeh 2020
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import torch


@dataclass
class TransducerGreedySearcherStreamingContext(torch.nn.Module):
    """Simple wrapper for the hidden state of the transducer greedy searcher.
    Used by :meth:`~TransducerBeamSearcher.transducer_greedy_decode_streaming`.
    """

    hidden: Optional[Any] = None
    """Hidden state; typically a tensor or a tuple of tensors."""


class TransducerBeamSearcher(torch.nn.Module):
    """
    This class implements the beam-search algorithm for the transducer model.

    Arguments
    ---------
    decode_network_lst : list
        List of prediction network (PN) layers.
    tjoint: transducer_joint module
        This module perform the joint between TN and PN.
    classifier_network : list
        List of output layers (after performing joint between TN and PN)
        exp: (TN,PN) => joint => classifier_network_list [DNN block, Linear..] => chars prob
    blank_id : int
        The blank symbol/index.
    beam_size : int
        The width of beam. Greedy Search is used when beam_size = 1.
    nbest : int
        Number of hypotheses to keep.
    lm_module : torch.nn.ModuleList
        Neural networks modules for LM.
    lm_weight : float
        The weight of LM when performing beam search (λ).
        log P(y|x) + λ log P_LM(y). (default: 0.3)
    state_beam : float
        The threshold coefficient in log space to decide if hyps in A (process_hyps)
        is likely to compete with hyps in B (beam_hyps), if not, end the while loop.
        Reference: https://arxiv.org/pdf/1911.01629.pdf
    expand_beam : float
        The threshold coefficient to limit the number of expanded hypotheses
        that are added in A (process_hyp).
        Reference: https://arxiv.org/pdf/1911.01629.pdf
        Reference: https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/simple-decoder.cc (See PruneToks)

    Example
    -------
    searcher = TransducerBeamSearcher(
        decode_network_lst=[hparams["emb"], hparams["dec"]],
        tjoint=hparams["Tjoint"],
        classifier_network=[hparams["transducer_lin"]],
        blank_id=0,
        beam_size=hparams["beam_size"],
        nbest=hparams["nbest"],
        lm_module=hparams["lm_model"],
        lm_weight=hparams["lm_weight"],
        state_beam=2.3,
        expand_beam=2.3,
    )
    >>> from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
    >>> import speechbrain as sb
    >>> emb = sb.nnet.embedding.Embedding(
    ...     num_embeddings=35,
    ...     embedding_dim=3,
    ...     consider_as_one_hot=True,
    ...     blank_id=0
    ... )
    >>> dec = sb.nnet.RNN.GRU(
    ...     hidden_size=10, input_shape=(1, 40, 34), bidirectional=False
    ... )
    >>> lin = sb.nnet.linear.Linear(input_shape=(1, 40, 10), n_neurons=35)
    >>> joint_network= sb.nnet.linear.Linear(input_shape=(1, 1, 40, 35), n_neurons=35)
    >>> tjoint = Transducer_joint(joint_network, joint="sum")
    >>> searcher = TransducerBeamSearcher(
    ...     decode_network_lst=[emb, dec],
    ...     tjoint=tjoint,
    ...     classifier_network=[lin],
    ...     blank_id=0,
    ...     beam_size=1,
    ...     nbest=1,
    ...     lm_module=None,
    ...     lm_weight=0.0,
    ... )
    >>> enc = torch.rand([1, 20, 10])
    >>> hyps, _, _, _ = searcher(enc)
    """

    def __init__(
        self,
        decode_network_lst,
        tjoint,
        classifier_network,
        blank_id,
        beam_size=4,
        nbest=5,
        lm_module=None,
        lm_weight=0.0,
        state_beam=2.3,
        expand_beam=2.3,
    ):
        super().__init__()
        self.decode_network_lst = decode_network_lst
        self.tjoint = tjoint
        self.classifier_network = classifier_network
        self.blank_id = blank_id
        self.beam_size = beam_size
        self.nbest = nbest
        self.lm = lm_module
        self.lm_weight = lm_weight

        if lm_module is None and lm_weight > 0:
            raise ValueError("Language model is not provided.")

        self.state_beam = state_beam
        self.expand_beam = expand_beam
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        if self.beam_size <= 1:
            self.searcher = self.transducer_greedy_decode
        else:
            self.searcher = self.transducer_beam_search_decode

    def forward(self, tn_output):
        """
        Arguments
        ---------
        tn_output : torch.Tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        Topk hypotheses
        """

        hyps = self.searcher(tn_output)
        return hyps

    def transducer_greedy_decode(
        self, tn_output, hidden_state=None, return_hidden=False
    ):
        """Transducer greedy decoder is a greedy decoder over batch which apply Transducer rules:
            1- for each time step in the Transcription Network (TN) output:
                -> Update the ith utterance only if
                    the previous target != the new one (we save the hiddens and the target)
                -> otherwise:
                ---> keep the previous target prediction from the decoder

        Arguments
        ---------
        tn_output : torch.Tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].
        hidden_state : (torch.Tensor, torch.Tensor)
            Hidden state to initially feed the decode network with. This is
            useful in conjunction with `return_hidden` to be able to perform
            beam search in a streaming context, so that you can reuse the last
            hidden state as an initial state across calls.
        return_hidden : bool
            Whether the return tuple should contain an extra 5th element with
            the hidden state at of the last step. See `hidden_state`.

        Returns
        -------
        Tuple of 4 or 5 elements (if `return_hidden`).

        First element: List[List[int]]
            List of decoded tokens

        Second element: torch.Tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.

        Third element: None
            nbest; irrelevant for greedy decode

        Fourth element: None
            nbest scores; irrelevant for greedy decode

        Fifth element: Present if `return_hidden`, (torch.Tensor, torch.Tensor)
            Tuple representing the hidden state required to call
            `transducer_greedy_decode` where you left off in a streaming
            context.
        """
        hyp = {
            "prediction": [[] for _ in range(tn_output.size(0))],
            "logp_scores": [0.0 for _ in range(tn_output.size(0))],
        }
        # prepare BOS = Blank for the Prediction Network (PN)
        input_PN = (
            torch.ones(
                (tn_output.size(0), 1),
                device=tn_output.device,
                dtype=torch.int32,
            )
            * self.blank_id
        )

        if hidden_state is None:
            # First forward-pass on PN
            out_PN, hidden = self._forward_PN(input_PN, self.decode_network_lst)
        else:
            out_PN, hidden = hidden_state

        # For each time step
        for t_step in range(tn_output.size(1)):
            # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
            log_probs = self._joint_forward_step(
                tn_output[:, t_step, :].unsqueeze(1).unsqueeze(1),
                out_PN.unsqueeze(1),
            )
            # Sort outputs at time
            logp_targets, positions = torch.max(log_probs.squeeze(1).squeeze(1), dim=1)
            # Batch hidden update
            have_update_hyp = []
            for i in range(positions.size(0)):
                # Update hiddens only if
                # 1- current prediction is non blank
                if positions[i].item() != self.blank_id:
                    hyp["prediction"][i].append(positions[i].item())
                    hyp["logp_scores"][i] += logp_targets[i]
                    input_PN[i][0] = positions[i]
                    have_update_hyp.append(i)
            if len(have_update_hyp) > 0:
                # Select sentence to update
                # And do a forward steps + generated hidden
                (
                    selected_input_PN,
                    selected_hidden,
                ) = self._get_sentence_to_update(have_update_hyp, input_PN, hidden)
                selected_out_PN, selected_hidden = self._forward_PN(
                    selected_input_PN, self.decode_network_lst, selected_hidden
                )
                # update hiddens and out_PN
                out_PN[have_update_hyp] = selected_out_PN
                hidden = self._update_hiddens(have_update_hyp, selected_hidden, hidden)

        ret = (
            hyp["prediction"],
            torch.Tensor(hyp["logp_scores"]).exp().mean(),
            None,
            None,
        )

        if return_hidden:
            # append the `(out_PN, hidden)` tuple to ret
            ret += (
                (
                    out_PN,
                    hidden,
                ),
            )

        return ret

    def transducer_greedy_decode_streaming(
        self, x: torch.Tensor, context: TransducerGreedySearcherStreamingContext
    ):
        """Tiny wrapper for
        :meth:`~TransducerBeamSearcher.transducer_greedy_decode` with an API
        that makes it suitable to be passed as a `decoding_function` for
        streaming.

        Arguments
        ---------
        x : torch.Tensor
            Outputs of the prediction network (equivalent to `tn_output`)
        context : TransducerGreedySearcherStreamingContext
            Mutable streaming context object, which must be specified and reused
            across calls when streaming.
            You can obtain an initial context by initializing a default object.

        Returns
        -------
        hyp : torch.Tensor
        """
        (hyp, _scores, _, _, hidden) = self.transducer_greedy_decode(
            x, context.hidden, return_hidden=True
        )
        context.hidden = hidden
        return hyp

    def transducer_beam_search_decode(self, tn_output):
        """Transducer beam search decoder is a beam search decoder over batch which apply Transducer rules:
            1- for each utterance:
                2- for each time steps in the Transcription Network (TN) output:
                    -> Do forward on PN and Joint network
                    -> Select topK <= beam
                    -> Do a while loop extending the hyps until we reach blank
                        -> otherwise:
                        --> extend hyp by the new token

        Arguments
        ---------
        tn_output : torch.Tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        torch.Tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.
        """

        # min between beam and max_target_lent
        nbest_batch = []
        nbest_batch_score = []
        for i_batch in range(tn_output.size(0)):
            # if we use RNN LM keep there hiddens
            # prepare BOS = Blank for the Prediction Network (PN)
            # Prepare Blank prediction
            blank = (
                torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                * self.blank_id
            )
            input_PN = (
                torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                * self.blank_id
            )
            # First forward-pass on PN
            hyp = {
                "prediction": [self.blank_id],
                "logp_score": 0.0,
                "hidden_dec": None,
            }
            if self.lm_weight > 0:
                lm_dict = {"hidden_lm": None}
                hyp.update(lm_dict)
            beam_hyps = [hyp]

            # For each time step
            for t_step in range(tn_output.size(1)):
                # get hyps for extension
                process_hyps = beam_hyps
                beam_hyps = []
                while True:
                    if len(beam_hyps) >= self.beam_size:
                        break
                    # Add norm score
                    a_best_hyp = max(
                        process_hyps,
                        key=partial(get_transducer_key),
                    )

                    # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                    if len(beam_hyps) > 0:
                        b_best_hyp = max(
                            beam_hyps,
                            key=partial(get_transducer_key),
                        )
                        a_best_prob = a_best_hyp["logp_score"]
                        b_best_prob = b_best_hyp["logp_score"]
                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    # remove best hyp from process_hyps
                    process_hyps.remove(a_best_hyp)

                    # forward PN
                    input_PN[0, 0] = a_best_hyp["prediction"][-1]
                    out_PN, hidden = self._forward_PN(
                        input_PN,
                        self.decode_network_lst,
                        a_best_hyp["hidden_dec"],
                    )
                    # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
                    log_probs = self._joint_forward_step(
                        tn_output[i_batch, t_step, :]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                        out_PN.unsqueeze(0),
                    )

                    if self.lm_weight > 0:
                        log_probs_lm, hidden_lm = self._lm_forward_step(
                            input_PN, a_best_hyp["hidden_lm"]
                        )

                    # Sort outputs at time
                    logp_targets, positions = torch.topk(
                        log_probs.view(-1), k=self.beam_size, dim=-1
                    )
                    best_logp = (
                        logp_targets[0] if positions[0] != blank else logp_targets[1]
                    )

                    # Extend hyp by  selection
                    for j in range(logp_targets.size(0)):
                        # hyp
                        topk_hyp = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"] + logp_targets[j],
                            "hidden_dec": a_best_hyp["hidden_dec"],
                        }

                        if positions[j] == self.blank_id:
                            beam_hyps.append(topk_hyp)
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = a_best_hyp["hidden_lm"]
                            continue

                        if logp_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp["prediction"].append(positions[j].item())
                            topk_hyp["hidden_dec"] = hidden
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = hidden_lm
                                topk_hyp["logp_score"] += (
                                    self.lm_weight * log_probs_lm[0, 0, positions[j]]
                                )
                            process_hyps.append(topk_hyp)
            # Add norm score
            nbest_hyps = sorted(
                beam_hyps,
                key=partial(get_transducer_key),
                reverse=True,
            )[: self.nbest]
            all_predictions = []
            all_scores = []
            for hyp in nbest_hyps:
                all_predictions.append(hyp["prediction"][1:])
                all_scores.append(hyp["logp_score"] / len(hyp["prediction"]))
            nbest_batch.append(all_predictions)
            nbest_batch_score.append(all_scores)
        return (
            [nbest_utt[0] for nbest_utt in nbest_batch],
            torch.Tensor([nbest_utt_score[0] for nbest_utt_score in nbest_batch_score])
            .exp()
            .mean(),
            nbest_batch,
            nbest_batch_score,
        )

    def _joint_forward_step(self, h_i, out_PN):
        """Join predictions (TN & PN)."""

        with torch.no_grad():
            # the output would be a tensor of [B,T,U, oneof[sum,concat](Hidden_TN,Hidden_PN)]
            out = self.tjoint(
                h_i,
                out_PN,
            )
            # forward the output layers + activation + save logits
            out = self._forward_after_joint(out, self.classifier_network)
            log_probs = self.softmax(out)
        return log_probs

    def _lm_forward_step(self, inp_tokens, memory):
        """This method should implement one step of
        forwarding operation for language model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The memory variables input for this timestep.
            (e.g., RNN hidden states).

        Return
        ------
        log_probs : torch.Tensor
            Log-probabilities of the current timestep output.
        hs : No limit
            The memory variables are generated in this timestep.
            (e.g., RNN hidden states).
        """
        with torch.no_grad():
            logits, hs = self.lm(inp_tokens, hx=memory)
            log_probs = self.softmax(logits)
        return log_probs, hs

    def _get_sentence_to_update(self, selected_sentences, output_PN, hidden):
        """Select and return the updated hiddens and output
        from the Prediction Network.

        Arguments
        ---------
        selected_sentences : list
            List of updated sentences (indexes).
        output_PN: torch.Tensor
            Output tensor from prediction network (PN).
        hidden : torch.Tensor
            Optional: None, hidden tensor to be used for
            recurrent layers in the prediction network.

        Returns
        -------
        selected_output_PN: torch.Tensor
            Outputs a logits tensor [B_selected,U, hiddens].
        hidden_update_hyp: torch.Tensor
            Selected hiddens tensor.
        """

        selected_output_PN = output_PN[selected_sentences, :]
        # for LSTM hiddens (hn, hc)
        if isinstance(hidden, tuple):
            hidden0_update_hyp = hidden[0][:, selected_sentences, :]
            hidden1_update_hyp = hidden[1][:, selected_sentences, :]
            hidden_update_hyp = (hidden0_update_hyp, hidden1_update_hyp)
        else:
            hidden_update_hyp = hidden[:, selected_sentences, :]
        return selected_output_PN, hidden_update_hyp

    def _update_hiddens(self, selected_sentences, updated_hidden, hidden):
        """Update hidden tensor by a subset of hidden tensor (updated ones).

        Arguments
        ---------
        selected_sentences : list
            List of index to be updated.
        updated_hidden : torch.Tensor
            Hidden tensor of the selected sentences for update.
        hidden : torch.Tensor
            Hidden tensor to be updated.

        Returns
        -------
        torch.Tensor
            Updated hidden tensor.
        """

        if isinstance(hidden, tuple):
            hidden[0][:, selected_sentences, :] = updated_hidden[0]
            hidden[1][:, selected_sentences, :] = updated_hidden[1]
        else:
            hidden[:, selected_sentences, :] = updated_hidden
        return hidden

    def _forward_PN(self, out_PN, decode_network_lst, hidden=None):
        """Compute forward-pass through a list of prediction network (PN) layers.

        Arguments
        ---------
        out_PN : torch.Tensor
            Input sequence from prediction network with shape
            [batch, target_seq_lens].
        decode_network_lst: list
            List of prediction network (PN) layers.
        hidden : torch.Tensor
            Optional: None, hidden tensor to be used for
                recurrent layers in the prediction network

        Returns
        -------
        out_PN : torch.Tensor
            Outputs a logits tensor [B,U, hiddens].
        hidden : torch.Tensor
            Hidden tensor to be used for the next step
            by recurrent layers in prediction network.
        """

        for layer in decode_network_lst:
            if layer.__class__.__name__ in [
                "RNN",
                "LSTM",
                "GRU",
                "LiGRU",
                "LiGRU_Layer",
            ]:
                out_PN, hidden = layer(out_PN, hidden)
            else:
                out_PN = layer(out_PN)
        return out_PN, hidden

    def _forward_after_joint(self, out, classifier_network):
        """Compute forward-pass through a list of classifier neural network.

        Arguments
        ---------
        out : torch.Tensor
            Output from joint network with shape
            [batch, target_len, time_len, hiddens]
        classifier_network : list
            List of output layers (after performing joint between TN and PN)
            exp: (TN,PN) => joint => classifier_network_list [DNN block, Linear..] => chars prob

        Returns
        -------
        torch.Tensor
            Outputs a logits tensor [B, U,T, Output_Dim];
        """

        for layer in classifier_network:
            out = layer(out)
        return out


def get_transducer_key(x):
    """Argument function to customize the sort order (in sorted & max).
    To be used as `key=partial(get_transducer_key)`.

    Arguments
    ---------
    x : dict
        one of the items under comparison

    Returns
    -------
    float
        Normalized log-score.
    """
    logp_key = x["logp_score"] / len(x["prediction"])
    return logp_key
