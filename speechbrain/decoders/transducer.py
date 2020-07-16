"""
Decoders and output normalization for Transducer sequence

Author:
    Abdelwahab HEBA 2020
"""
import torch


def decode_batch(
    tn_output, decode_network_lst, tjoint, classifier_network, blank_id
):
    """
    Batch greedy decoder is a greedy decoder over batch which apply Transducer rules:
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
        >>> from speechbrain.decoders.transducer import decode_batch
        >>> from speechbrain.nnet.RNN import GRU
        >>> from speechbrain.nnet.activations import Softmax
        >>> from speechbrain.nnet.embedding import Embedding
        >>> from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
        >>> from speechbrain.nnet.linear import Linear
        >>> TN = GRU(hidden_size=5, num_layers=1, bidirectional=True)
        >>> TN_lin = Linear(n_neurons=35, bias=True)
        >>> blank_id = 34
        >>> PN_emb = Embedding(num_embeddings=35, consider_as_one_hot=True, blank_id=blank_id)
        >>> PN = GRU(hidden_size=5, num_layers=1, bidirectional=False, return_hidden=True)
        >>> PN_lin = Linear(n_neurons=35, bias=True)
        >>> joint_network= Linear(n_neurons=35, bias=True)
        >>> tjoint = Transducer_joint(joint_network, joint="sum")
        >>> Out_lin = Linear(n_neurons=35)
        >>> log_softmax = Softmax(apply_log=False)
        >>> inputs = torch.randn((3,40,35))
        >>> TN_out = TN(inputs, init_params=True)
        >>> TN_out = TN_lin(TN_out, init_params=True)
        >>> # Initialize modules...
        >>> test_emb = PN_emb(torch.Tensor([[1]]).long(), init_params=True)
        >>> test_PN, _ = PN(test_emb, init_params=True)
        >>> test_PN = PN_lin(test_PN, init_params=True)
        >>> # init tjoint
        >>> joint_tensor = tjoint(TN_out.unsqueeze(1), test_PN.unsqueeze(2), init_params=True)
        >>> out = Out_lin(joint_tensor, init_params=True)
        >>> out_decode = decode_batch(TN_out, [PN_emb,PN,PN_lin], tjoint, [Out_lin], blank_id)

    Author:
        Abdelwahab HEBA 2020
    """
    # prepare BOS= Blank for the Prediction Network (PN)
    hidden = None
    list_outs = []
    # Prepare Blank prediction
    input_PN = (
        torch.ones(
            (tn_output.size(0), 1), device=tn_output.device, dtype=torch.int64
        )
        * blank_id
    )
    out_PN = input_PN
    # First forward-pass on PN
    hidden = None
    for layer in decode_network_lst:
        if layer.__class__.__name__ in [
            "RNN",
            "LSTM",
            "GRU",
            "LiGRU",
            "LiGRU_Layer",
        ]:
            out_PN, hidden = layer(out_PN)
        else:
            out_PN = layer(out_PN)
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
        for layer in classifier_network:
            out = layer(out)

        list_outs.append(out)
        # Sort outputs at time
        prob_targets, positions = torch.max(
            out.log_softmax(dim=-1).squeeze(1).squeeze(1), dim=1
        )
        # Batch hidden update
        have_update_hyp = []
        for i in range(positions.size(0)):
            # Update hiddens only if
            # 1- current prediction is non blank
            # 2- current prediction is different from the previous
            if positions[i] != blank_id and positions[i] != input_PN[i][0]:
                input_PN[i][0] = positions[i]
                have_update_hyp.append(i)
        if len(have_update_hyp) > 0:
            out_update_hyp = input_PN[have_update_hyp, :]
            # for LSTM hiddens (hn, hc)
            if isinstance(hidden, tuple):
                hidden0_update_hyp = hidden[0][:, have_update_hyp, :]
                hidden1_update_hyp = hidden[1][:, have_update_hyp, :]
                hidden_update_hyp = (hidden0_update_hyp, hidden1_update_hyp)
            else:
                hidden_update_hyp = hidden[:, have_update_hyp, :]

            for layer in decode_network_lst:
                if layer.__class__.__name__ in [
                    "RNN",
                    "LSTM",
                    "GRU",
                    "LiGRU",
                    "LiGRU_Layer",
                ]:
                    out_update_hyp, hidden_update_hyp = layer(
                        out_update_hyp, hidden_update_hyp, init_params=False
                    )
                else:
                    out_update_hyp = layer(out_update_hyp)

            out_PN[have_update_hyp] = out_update_hyp
            # for LSTM hiddens (hn, hc)
            if isinstance(hidden, tuple):
                hidden[0][:, have_update_hyp, :] = hidden_update_hyp[0]
                hidden[1][:, have_update_hyp, :] = hidden_update_hyp[1]
            else:
                hidden[:, have_update_hyp, :] = hidden_update_hyp
    return torch.cat(list_outs, dim=1)
