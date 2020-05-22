"""
Decoders and output normalization for Transducer sequence

Author:
    Abdelwahab HEBA 2020
"""
import torch


def decode_batch(F, decode_network_lst, Tjoint, classif_network_lst, blank_id):
    """
    Batch greedy decoder of the probabilities and apply Transducer rules

    Arguments
    ----------
    F : torch.tensor
        output from transcription network with shape
        [batch, time_len, hiddens]
    decode_network_lst: list
        list of prediction layers
    Tjoint: transducer_joint module
        this module perform the joint between TN and PN
    classif_network_lst: list
        list of output layers (after performing joint between TN and PN)
    blank_id : int, string
        The blank symbol/index. Default: -1. If a negative number is given,
        it is assumed to mean counting down from the maximum possible index,
        so that -1 refers to the maximum possible index.

    Returns
    -------
    list
        Outputs as Python list of lists, with "ragged" dimensions; padding
        has been removed.

    Example
    -------
        >>> import torch
        >>> from speechbrain.decoders.transducer import decode_batch
        >>> from speechbrain.nnet.RNN import RNN
        >>> from speechbrain.nnet.activations import Softmax
        >>> from speechbrain.nnet.embedding import Embedding
        >>> from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
        >>> from speechbrain.nnet.linear import Linear
        >>> TN = RNN(rnn_type="gru", n_neurons=5, num_layers=1, bidirectional=True)
        >>> TN_lin = Linear(n_neurons=35, bias=True)
        >>> blank_id = 35
        >>> PN_emb = Embedding(embeddings_dim=35, consider_as_one_hot=True, blank_id=blank_id)
        >>> PN = RNN(rnn_type="gru", n_neurons=5, num_layers=1, bidirectional=False, return_hidden=True)
        >>> PN_lin = Linear(n_neurons=35, bias=True)
        >>> Tjoint = Transducer_joint(joint="sum")
        >>> Out_lin = Linear(n_neurons=35)
        >>> log_softmax = Softmax(apply_log=False)
        >>> inputs = torch.randn((3,40,35))
        >>> TN_out = TN(inputs, init_params=True)
        >>> TN_out = TN_lin(TN_out, init_params=True)
        >>> # Initialize modules...
        >>> test_emb = PN_emb(torch.Tensor([[1]]).long(), init_params=True)
        >>> test_PN, _ = PN(test_emb, init_params=True)
        >>> test_PN = PN_lin(test_PN, init_params=True)
        >>> joint_tensor = Tjoint(TN_out, test_PN)
        >>> out = Out_lin(joint_tensor, init_params=True)
        >>> out_decode = decode_batch(TN_out, [PN_emb,PN,PN_lin], Tjoint, [Out_lin,log_softmax], blank_id)

    Author:
        Abdelwahab HEBA 2020
    """
    hidden = None
    list_outs = []
    # Prepare Blank prediction
    input_PN = (
        torch.ones((F.size(0), 1), device=F.device, dtype=torch.int64)
        * blank_id
    )
    out_PN = input_PN
    # First forward on PN
    hidden = None
    for layer in decode_network_lst:
        if layer.__class__.__name__ == "RNN":
            out_PN, hidden = layer(out_PN, init_params=False)
        else:
            if layer.__class__.__name__ == "Embedding":
                out_PN = layer(out_PN, init_params=False)
            else:
                out_PN = layer(out_PN, init_params=False)
    # For each time step
    for t_step in range(F.size(1)):
        # Join predictions (TN & PN)
        out = Tjoint(
            F[:, t_step, :].unsqueeze(1).unsqueeze(1), out_PN.unsqueeze(1)
        )
        # forward the output layers + activation + save logits
        activation = False
        for layer in classif_network_lst:
            out = layer(out)
            if layer.__class__.__name__ == "Softmax":
                activation = True
                list_outs.append(out)
                out = layer(out)
        # Sort outputs at time T
        if activation:
            prob_targets, positions = torch.max(
                out.squeeze(1).squeeze(1), dim=1
            )
        else:
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
                if layer.__class__.__name__ == "RNN":
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
