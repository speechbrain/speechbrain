"""
Decoders and output normalization for Transducer sequence

Author:
    Abdelwahab HEBA 2020
"""
import torch


def _get_sentence_to_update(selected_sentences, output_PN, hidden):
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

    Example
    -------
        >>> import torch
        >>> from speechbrain.decoders.transducer import _get_sentence_to_update
        >>> from speechbrain.nnet.RNN import GRU
        >>> from speechbrain.nnet.embedding import Embedding
        >>> blank_id = 34
        >>> PN_emb = Embedding(num_embeddings=35, consider_as_one_hot=True, blank_id=blank_id)
        >>> test_emb = PN_emb(torch.Tensor([[1],[2],[10],[6]]).long())
        >>> PN = GRU(hidden_size=5, input_shape=test_emb.shape)
        >>> test_PN, hidden = PN(test_emb)
        >>> selected_sentences = [1,3]
        >>> selected_output_PN, selected_hidden = _get_sentence_to_update(selected_sentences, test_PN, hidden)

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


def _update_hiddens(selected_sentences, updated_hidden, hidden):
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

    Example
    -------
        >>> import torch
        >>> from speechbrain.decoders.transducer import _update_hiddens
        >>> from speechbrain.nnet.RNN import GRU
        >>> from speechbrain.nnet.embedding import Embedding
        >>> blank_id = 34
        >>> PN_emb = Embedding(num_embeddings=35, consider_as_one_hot=True, blank_id=blank_id)
        >>> test_emb = PN_emb(torch.Tensor([[1],[2],[10],[6]]).long())
        >>> PN = GRU(hidden_size=5, input_shape=test_emb.shape)
        >>> test_PN, hidden = PN(test_emb)
        >>> selected_sentences = [1,3]
        >>> updated_hidden = torch.ones((1,2,5))
        >>> hidden = _update_hiddens(selected_sentences, updated_hidden, hidden)

    Author:
        Abdelwahab HEBA 2020
    """
    if isinstance(hidden, tuple):
        hidden[0][:, selected_sentences, :] = updated_hidden[0]
        hidden[1][:, selected_sentences, :] = updated_hidden[1]
    else:
        hidden[:, selected_sentences, :] = updated_hidden
    return hidden


def _forward_PN(out_PN, decode_network_lst, hidden=None):
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

    Example
    -------
        >>> import torch
        >>> from speechbrain.decoders.transducer import _forward_PN
        >>> from speechbrain.nnet.RNN import GRU
        >>> from speechbrain.nnet.embedding import Embedding
        >>> blank_id = 34
        >>> PN_emb = Embedding(num_embeddings=35, consider_as_one_hot=True, blank_id=blank_id)
        >>> test_emb = PN_emb(torch.Tensor([[1]]).long())
        >>> PN = GRU(hidden_size=5, input_shape=test_emb.shape)
        >>> test_PN, hidden = PN(test_emb)
        >>> out_PN, hidden = _forward_PN(torch.Tensor([[1]]).long(), [PN_emb, PN], hidden)

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


def _forward_after_joint(out, classifier_network):
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

    Example
    -------
        >>> import torch
        >>> from speechbrain.decoders.transducer import _forward_after_joint
        >>> from speechbrain.nnet.linear import Linear
        >>> inputs = torch.rand(3, 5, 10, 5)
        >>> Out_lin1 = Linear(input_shape=(3, 5, 10, 5), n_neurons=10)
        >>> Out_lin2 = Linear(input_shape=(3, 5, 10, 10), n_neurons=15)
        >>> out = Out_lin1(inputs)
        >>> out = Out_lin2(out)
        >>> logits = _forward_after_joint(inputs, [Out_lin1, Out_lin2])

    Author:
        Abdelwahab HEBA 2020
    """
    for layer in classifier_network:
        out = layer(out)
    return out


def transducer_greedy_decode(
    tn_output, decode_network_lst, tjoint, classifier_network, blank_id
):
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
    decode_network_lst: list
        list of prediction netowrk (PN) layers
    tjoint: transducer_joint module
        this module perform the joint between TN and PN
    classifier_network: list
        list of output layers (after performing joint between TN and PN)
        exp: (TN,PN) => joint => classifier_network_list [DNN bloc, Linear..] => chars prob
    blank_id : int, string
        The blank symbol/index. Default: -1. If a negative number is given,
        it is assumed to mean counting down from the maximum possible index,
        so that -1 refers to the maximum possible index.

    Returns
    -------
    torch.tensor
        Outputs a logits tensor [B,T,1,Output_Dim]; padding
        has not been removed.

    Example
    -------
        >>> import torch
        >>> from speechbrain.decoders.transducer import transducer_greedy_decode
        >>> from speechbrain.nnet.RNN import GRU
        >>> from speechbrain.nnet.activations import Softmax
        >>> from speechbrain.nnet.embedding import Embedding
        >>> from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
        >>> from speechbrain.nnet.linear import Linear
        >>> inputs = torch.rand(3, 40, 35)
        >>> TN = GRU(hidden_size=5, input_shape=(3, 40, 35), bidirectional=True)
        >>> TN_lin = Linear(input_shape=(3, 40, 10), n_neurons=35)
        >>> log_softmax = Softmax(apply_log=False)
        >>> TN_out, hidden = TN(inputs)
        >>> TN_out = TN_lin(TN_out)
        >>> # Initialize modules...
        >>> blank_id = 34
        >>> PN_emb = Embedding(num_embeddings=35, consider_as_one_hot=True, blank_id=blank_id)
        >>> test_emb = PN_emb(torch.Tensor([[1]]).long())
        >>> PN = GRU(hidden_size=5, input_shape=test_emb.shape)
        >>> test_PN, _ = PN(test_emb)
        >>> PN_lin = Linear(input_shape=test_PN.shape, n_neurons=35)
        >>> test_PN = PN_lin(test_PN)
        >>> # init tjoint
        >>> joint_network= Linear(input_shape=TN_out.unsqueeze(1).shape, n_neurons=35)
        >>> tjoint = Transducer_joint(joint_network, joint="sum")
        >>> joint_tensor = tjoint(TN_out.unsqueeze(1), test_PN.unsqueeze(2))
        >>> Out_lin = Linear(input_shape=joint_tensor.shape, n_neurons=35)
        >>> out = Out_lin(joint_tensor)
        >>> best_hyps, scores = transducer_greedy_decode(TN_out, [PN_emb,PN,PN_lin], tjoint, [Out_lin], blank_id)

    Author:
        Abdelwahab HEBA 2020
    """
    hyp = {
        "prediction": [[] for _ in range(tn_output.size(0))],
        "logp_scores": [0.0 for _ in range(tn_output.size(0))],
    }
    # prepare BOS= Blank for the Prediction Network (PN)
    hidden = None
    # Prepare Blank prediction
    input_PN = (
        torch.ones(
            (tn_output.size(0), 1), device=tn_output.device, dtype=torch.int64
        )
        * blank_id
    )
    # First forward-pass on PN
    out_PN, hidden = _forward_PN(input_PN, decode_network_lst)
    # For each time step
    for t_step in range(tn_output.size(1)):
        # Join predictions (TN & PN)
        # tjoint must be have a 4 dim [B,T,U,Hidden]
        # so do unsqueeze over
        # the output would be a tensor of [B,T,U, oneof[sum,concat](Hidden_TN,Hidden_PN)]
        out = tjoint(
            tn_output[:, t_step, :].unsqueeze(1).unsqueeze(1),
            out_PN.unsqueeze(1),
        )
        # forward the output layers + activation + save logits
        out = _forward_after_joint(out, classifier_network)
        # Sort outputs at time
        logp_targets, positions = torch.max(
            out.log_softmax(dim=-1).squeeze(1).squeeze(1), dim=1
        )
        # Batch hidden update
        have_update_hyp = []
        for i in range(positions.size(0)):
            # Update hiddens only if
            # 1- current prediction is non blank
            if positions[i] != blank_id:
                hyp["prediction"][i].append(positions[i].item())
                hyp["logp_scores"][i] += logp_targets[i]
                input_PN[i][0] = positions[i]
                have_update_hyp.append(i)
        if len(have_update_hyp) > 0:
            # Select sentence to update
            # And do a forward steps + generated hidden
            selected_input_PN, selected_hidden = _get_sentence_to_update(
                have_update_hyp, input_PN, hidden
            )
            selected_out_PN, selected_hidden = _forward_PN(
                selected_input_PN, decode_network_lst, selected_hidden
            )
            # update hiddens and out_PN
            out_PN[have_update_hyp] = selected_out_PN
            hidden = _update_hiddens(have_update_hyp, selected_hidden, hidden)

    return hyp["prediction"], torch.Tensor(hyp["logp_scores"]).exp().mean()


def transducer_beam_search_decode(
    tn_output,
    decode_network_lst,
    tjoint,
    classifier_network,
    blank_id=-1,
    beam=4,
    nbest=5,
    lm_module=None,
    lm_weight=0.3,
    state_beam=2.0,
    expand_beam=2.0,
):
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
    decode_network_lst : list
        list of prediction netowrk (PN) layers
    tjoint: transducer_joint module
        this module perform the joint between TN and PN
    classifier_network : list
        list of output layers (after performing joint between TN and PN)
        exp: (TN,PN) => joint => classifier_network_list [DNN bloc, Linear..] => chars prob
    blank_id : int, string
        The blank symbol/index. Default: -1. If a negative number is given,
        it is assumed to mean counting down from the maximum possible index,
        so that -1 refers to the maximum possible index.
    beam : int
        The width of beam.
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
    Returns
    -------
    torch.tensor
        Outputs a logits tensor [B,T,1,Output_Dim]; padding
        has not been removed.

    Example
    -------
        >>> import torch
        >>> from speechbrain.decoders.transducer import transducer_beam_search_decode
        >>> from speechbrain.nnet.RNN import GRU
        >>> from speechbrain.nnet.activations import Softmax
        >>> from speechbrain.nnet.embedding import Embedding
        >>> from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
        >>> from speechbrain.nnet.linear import Linear
        >>> inputs = torch.rand(3, 40, 35)
        >>> TN = GRU(hidden_size=5, input_shape=(3, 40, 35))
        >>> TN_lin = Linear(input_shape=(3, 40, 5), n_neurons=35)
        >>> blank_id = 34
        >>> log_softmax = Softmax(apply_log=False)
        >>> TN_out, _ = TN(inputs)
        >>> TN_out = TN_lin(TN_out)
        >>> # Initialize modules...
        >>> PN_emb = Embedding(num_embeddings=35, consider_as_one_hot=True, blank_id=blank_id)
        >>> test_emb = PN_emb(torch.Tensor([[1]]).long())
        >>> PN = GRU(hidden_size=5, input_shape=test_emb.shape)
        >>> test_PN, _ = PN(test_emb)
        >>> PN_lin = Linear(input_shape=test_PN.shape, n_neurons=35)
        >>> test_PN = PN_lin(test_PN)
        >>> # init tjoint
        >>> joint_network= Linear(input_shape=TN_out.unsqueeze(1).shape, n_neurons=35)
        >>> tjoint = Transducer_joint(joint_network, joint="sum")
        >>> joint_tensor = tjoint(TN_out.unsqueeze(1), test_PN.unsqueeze(2))
        >>> Out_lin = Linear(input_shape=joint_tensor.shape, n_neurons=35)
        >>> out = Out_lin(joint_tensor)
        >>> # out_decode = transducer_beam_search_decode(TN_out, [PN_emb,PN,PN_lin], tjoint, [Out_lin], blank_id, beam=2, nbest=5)

    Author:
        Abdelwahab HEBA 2020
    """
    # min between beam and max_target_lent
    nbest_batch = []
    nbest_batch_score = []
    for i_batch in range(tn_output.size(0)):
        # if we use RNN LM keep there hiddens
        # prepare BOS = Blank for the Prediction Network (PN)
        # Prepare Blank prediction
        input_PN = (
            torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
            * blank_id
        )
        # First forward-pass on PN
        out_PN, hidden = _forward_PN(input_PN, decode_network_lst)
        hyp = {
            "prediction": [],
            "logp_score": 0.0,
            "hidden_dec": hidden,
            "out_PN": out_PN,
        }
        if lm_module:
            lm_dict = {"hidden_lm": None}
            hyp.update(lm_dict)
        beam_hyps = [hyp]

        # For each time step
        for t_step in range(tn_output.size(1)):
            # get hyps for extension
            process_hyps = beam_hyps
            beam_hyps = []
            while True:
                if len(beam_hyps) >= beam:
                    break
                # pondéré la proba
                a_best_hyp = max(process_hyps, key=lambda x: x["logp_score"])

                # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                if len(beam_hyps) > 0:
                    b_best_hyp = max(beam_hyps, key=lambda x: x["logp_score"])
                    a_best_prob = a_best_hyp["logp_score"]
                    b_best_prob = b_best_hyp["logp_score"]
                    if b_best_prob >= state_beam + a_best_prob:
                        break

                # remove best hyp from process_hyps
                process_hyps.remove(a_best_hyp)
                out_PN = a_best_hyp["out_PN"]

                # Join predictions (TN & PN)
                # tjoint must be have a 4 dim [B,T,U,Hidden]
                # so do unsqueeze over
                # the output would be a tensor of [B,T,U, oneof[sum,concat](Hidden_TN,Hidden_PN)]
                out = tjoint(
                    tn_output[i_batch, t_step, :]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(0),
                    out_PN.unsqueeze(0),
                )
                # forward the output layers + activation + save logits
                out = _forward_after_joint(out, classifier_network)
                out = out.log_softmax(dim=-1)

                if lm_module:
                    # print(input_PN)
                    logits, hidden_lm = lm_module(
                        input_PN, hx=a_best_hyp["hidden_lm"]
                    )
                    log_probs_lm = logits.log_softmax(dim=-1)

                # Sort outputs at time
                logp_targets, positions = torch.topk(
                    out.view(-1)[1:], k=beam, dim=-1
                )
                best_logp = logp_targets[0]

                # concat blank_id
                logp_targets = torch.cat((logp_targets, out.view(-1)[0:1]))
                positions = torch.cat(
                    (
                        positions + 1,
                        torch.zeros(
                            (1), device=tn_output.device, dtype=torch.int32
                        ),
                    )
                )

                # Extend hyp by  selection
                for j in range(logp_targets.size(0)):

                    # hyp
                    topk_hyp = {
                        "prediction": a_best_hyp["prediction"],
                        "logp_score": a_best_hyp["logp_score"]
                        + logp_targets[j],
                        "hidden_dec": a_best_hyp["hidden_dec"],
                        "out_PN": a_best_hyp["out_PN"],
                    }

                    if positions[j] == blank_id:
                        if lm_module:
                            topk_hyp["hidden_lm"] = a_best_hyp["hidden_lm"]
                        beam_hyps.append(topk_hyp)
                        continue

                    if logp_targets[j] >= best_logp - expand_beam:
                        input_PN[0, 0] = positions[j]
                        out_PN, hidden = _forward_PN(
                            input_PN,
                            decode_network_lst,
                            a_best_hyp["hidden_dec"],
                        )
                        topk_hyp["prediction"].append(positions[j].item())
                        topk_hyp["hidden_dec"] = hidden
                        topk_hyp["out_PN"] = out_PN
                        if lm_module:
                            topk_hyp["hidden_lm"] = hidden_lm
                            topk_hyp["logp_score"] += (
                                lm_weight * log_probs_lm[0, 0, positions[j]]
                            )
                        process_hyps.append(topk_hyp)
        # Add norm score
        nbest_hyps = sorted(
            beam_hyps,
            key=lambda x: x["logp_score"] / len(x["prediction"]),
            reverse=True,
        )[:nbest]
        all_predictions = []
        all_scores = []
        for hyp in nbest_hyps:
            all_predictions.append(hyp["prediction"])
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
