"""
Decoders and output normalization for Transducer sequence

Author:
    Abdelwahab HEBA 2020
    Sung-Lin Yeh 2020
"""
import torch


class TransducerBeamSearcher(torch.nn.Module):
    """
    This class implements the beam-search algorithm for transducer model.

    Parameters
    ----------
    decode_network_lst : list
        list of prediction netowrk (PN) layers
    tjoint: transducer_joint module
        this module perform the joint between TN and PN
    classifier_network : list
        list of output layers (after performing joint between TN and PN)
        exp: (TN,PN) => joint => classifier_network_list [DNN bloc, Linear..] => chars prob
    blank_id : int,
        The blank symbol/index.
    beam : int
        The width of beam. Greedy Search is used when beam = 1.
    nbest : int
        Number of hypothesis to keep.
    lm_module: torch.nn.ModuleList
        neural networks modules for LM.
    lm_weight: float
        Default: 0.3
        The weight of LM when performing beam search (λ).
        log P(y|x) + λ log P_LM(y)
    state_beam: float
        The threshold coefficient in log space to decide if hyps in A (process_hyps)
        is likely to compete with hyps in B (beam_hyps), if not, end the while loop.
        Reference: https://arxiv.org/abs/1904.02619
    expand_beam: float
        The threshold coefficient to limit number of expanded hypothesises that are added in A (process_hyp).
        Reference: https://arxiv.org/abs/1904.02619

    Author:
        Abdelwahab HEBA 2020
        Sung-Lin Yeh 2020
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
        super(TransducerBeamSearcher, self).__init__()
        self.decode_network_lst = decode_network_lst
        self.tjoint = tjoint
        self.classifier_network = classifier_network
        self.blank_id = blank_id
        self.beam_size = beam_size
        self.nbest = nbest
        self.lm = lm_module
        self.lm_weight = lm_weight

        if lm_module is not None and lm_weight == 0:
            raise ValueError(
                "Language model is provided. \
                             Set lm_weigt != 0"
            )

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
        ----------
        tn_output : torch.tensor
            output from transcription network with shape
            [batch, time_len, hiddens]
        Returns
        -------
        Topk hypotheses
        """

        hyps = self.searcher(tn_output)
        return hyps

    def transducer_greedy_decode(self, tn_output):
        """
        transducer greedy decoder is a greedy decoder over batch which apply Transducer rules:
            1- for each time stemps in the Transcription Network (TN) output:
                -> Update the ith utterance only if
                    the previous target != the new one (we save the hiddens and the target)
                -> otherwise:
                ---> keep the previous target prediction from the decoder

        Arguments
        ----------
        tn_output : torch.tensor
            output from transcription network with shape
            [batch, time_len, hiddens]

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.

        Author:
            Abdelwahab HEBA 2020
        """
        hyp = {
            "prediction": [[] for _ in range(tn_output.size(0))],
            "logp_scores": [0.0 for _ in range(tn_output.size(0))],
        }
        # prepare BOS = Blank for the Prediction Network (PN)
        hidden = None
        input_PN = (
            torch.ones(
                (tn_output.size(0), 1),
                device=tn_output.device,
                dtype=torch.int32,
            )
            * self.blank_id
        )
        # First forward-pass on PN
        out_PN, hidden = self._forward_PN(input_PN, self.decode_network_lst)
        # For each time step
        for t_step in range(tn_output.size(1)):
            # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
            log_probs = self._joint_forward_step(
                tn_output[:, t_step, :].unsqueeze(1).unsqueeze(1),
                out_PN.unsqueeze(1),
            )
            # Sort outputs at time
            logp_targets, positions = torch.max(
                self.softmax(log_probs).squeeze(1).squeeze(1), dim=1
            )
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
                ) = self._get_sentence_to_update(
                    have_update_hyp, input_PN, hidden
                )
                selected_out_PN, selected_hidden = self._forward_PN(
                    selected_input_PN, self.decode_network_lst, selected_hidden
                )
                # update hiddens and out_PN
                out_PN[have_update_hyp] = selected_out_PN
                hidden = self._update_hiddens(
                    have_update_hyp, selected_hidden, hidden
                )

        return (
            hyp["prediction"],
            torch.Tensor(hyp["logp_scores"]).exp().mean(),
            None,
            None,
        )

    def transducer_beam_search_decode(self, tn_output):
        """
        transducer beam search decoder is a beam search decoder over batch which apply Transducer rules:
            1- for each utterance:
                2- for each time stemps in the Transcription Network (TN) output:
                    -> Do forward on PN and Joint network
                    -> Select topK <= beam
                    -> Do a while loop extending the hyps until we reach blank
                        -> otherwise:
                        --> extend hyp by the new token

        Arguments
        ----------
        tn_output : torch.tensor
            output from transcription network with shape
            [batch, time_len, hiddens]

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.

        Author:
            Abdelwahab HEBA 2020
            Sung-Lin Yeh 2020
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
                        key=lambda x: x["logp_score"] / len(x["prediction"]),
                    )

                    # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                    if len(beam_hyps) > 0:
                        b_best_hyp = max(
                            beam_hyps,
                            key=lambda x: x["logp_score"]
                            / len(x["prediction"]),
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
                        log_probs.view(-1)[1:], k=self.beam_size, dim=-1
                    )
                    best_logp = logp_targets[0]

                    # concat blank_id
                    logp_targets = torch.cat(
                        (logp_targets, log_probs.view(-1)[0:1])
                    )
                    positions = torch.cat((positions + 1, blank.squeeze(1)))

                    # Extend hyp by  selection
                    for j in range(logp_targets.size(0)):

                        # hyp
                        topk_hyp = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"]
                            + logp_targets[j],
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
                                    self.lm_weight
                                    * log_probs_lm[0, 0, positions[j]]
                                )
                            process_hyps.append(topk_hyp)
            # Add norm score
            nbest_hyps = sorted(
                beam_hyps,
                key=lambda x: x["logp_score"] / len(x["prediction"]),
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
            torch.Tensor(
                [nbest_utt_score[0] for nbest_utt_score in nbest_batch_score]
            )
            .exp()
            .mean(),
            nbest_batch,
            nbest_batch_score,
        )

    def _joint_forward_step(self, h_i, out_PN):
        """
        Join predictions (TN & PN)
        """
        with torch.no_grad():
            # the output would be a tensor of [B,T,U, oneof[sum,concat](Hidden_TN,Hidden_PN)]
            out = self.tjoint(h_i, out_PN,)
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
            The input tensor of current timestep.
        memory : No limit
            The momory variables input for this timestep.
            (ex. RNN hidden states).

        Return
        ------
        log_probs : torch.Tensor
            Log-probilities of the current timestep output.
        hs : No limit
            The memory variables generated in this timestep.
            (ex. RNN hidden states).
        """
        with torch.no_grad():
            logits, hs = self.lm(inp_tokens, hx=memory)
            log_probs = self.softmax(logits)
        return log_probs, hs

    def _get_sentence_to_update(self, selected_sentences, output_PN, hidden):
        """
        Select and return the updated hiddens and output
        from the Prediction Network

        Arguments
        ----------
        selected_sentences : list
            list of updated sentences (indexes)
        output_PN: torch.tensor
            output tensor from prediction netowrk (PN)
        hidden : torch.tensor
            Optional: None, hidden tensor to be used for
            reccurent layers in the prediction network

        Returns
        -------
        selected_output_PN: torch.tensor
            Outputs a logits tensor [B_selected,U, hiddens]
        hidden_update_hyp: torch.tensor
            selected hiddens tensor

        Author:
            Abdelwahab HEBA 2020
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
        """
        Update hidden tensor by a subset of hidden tensor (updated ones)

        Arguments
        ----------
        selected_sentences : list
            list of index to be updated
        updated_hidden: torch.tensor
            hidden tensor of the selected sentences for update
        hidden: torch.tensor
            hidden tensor to be updated

        Returns
        -------
        torch.tensor
            updated hidden tensor

        Author:
            Abdelwahab HEBA 2020
        """
        if isinstance(hidden, tuple):
            hidden[0][:, selected_sentences, :] = updated_hidden[0]
            hidden[1][:, selected_sentences, :] = updated_hidden[1]
        else:
            hidden[:, selected_sentences, :] = updated_hidden
        return hidden

    def _forward_PN(self, out_PN, decode_network_lst, hidden=None):
        """
        Compute forward-pass through a list of prediction network (PN) layers

        Arguments
        ----------
        out_PN : torch.tensor
            input sequence from prediction network with shape
            [batch, target_seq_lens]
        decode_network_lst: list
            list of prediction netowrk (PN) layers
        hinne : torch.tensor
            Optional: None, hidden tensor to be used for
                reccurent layers in the prediction network

        Returns
        -------
        out_PN: torch.tensor
            Outputs a logits tensor [B,U, hiddens]
        hidden: torch.tensor
            Hidden tensor to be used for the next step
            by reccurent layers in prediction network

        Author:
            Abdelwahab HEBA 2020
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
        """
        Compute forward-pass through a list of classifier neural network

        Arguments
        ----------
        out: torch.tensor
            output from joint network with shape
            [batch, target_len, time_len, hiddens]
        classifier_network: list
            list of output layers (after performing joint between TN and PN)
            exp: (TN,PN) => joint => classifier_network_list [DNN bloc, Linear..] => chars prob

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B, U,T, Output_Dim];

        Author:
            Abdelwahab HEBA 2020
        """
        for layer in classifier_network:
            out = layer(out)
        return out
