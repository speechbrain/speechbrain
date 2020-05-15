"""
Decoders and output normalization for Transducer sequence

Author:
    Abdelwahab HEBA 2020
"""
import torch
from itertools import groupby


def filter_ctc_output(string_pred, blank_id=-1):
    """Apply CTC output merge and filter rules.

    Removes the blank symbol and output repetitions.

    Parameters
    ----------
    string_pred : list
        a list containing the output strings/ints predicted by the CTC system
    blank_id : int, string
        the id of the blank

    Returns
    ------
    list
          The output predicted by CTC without the blank symbol and
          the repetitions

    Example
    -------
        >>> string_pred = ['a','a','blank','b','b','blank','c']
        >>> string_out = filter_ctc_output(string_pred, blank_id='blank')
        >>> print(string_out)
        ['a', 'b', 'c']

    Author
    ------
        Mirco Ravanelli 2020
    """

    if isinstance(string_pred, list):
        # Filter the repetitions
        string_out = [
            v
            for i, v in enumerate(string_pred)
            if i == 0 or v != string_pred[i - 1]
        ]

        # Remove duplicates
        string_out = [i[0] for i in groupby(string_out)]

        # Filter the blank symbol
        string_out = list(filter(lambda elem: elem != blank_id, string_out))
    else:
        raise ValueError("filter_ctc_out can only filter python lists")
    return string_out

def decode_batch(F, decode_network_lst, Tjoint, classif_network_lst, blank_id):
    """
    Greedy decode a batch of probabilities with Transducer rules

    Parameters
    ----------
    probabilities : torch.tensor
        Output probabilities (or log-probabilities) from network with shape
        [batch, probabilities, time]
    seq_lens : torch.tensor
        Relative true sequence lengths (to deal with padded inputs),
        longest sequence has length 1.0, others a value betwee zero and one
        shape [batch, lengths]
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
        >>> probs = torch.tensor([[[0.3, 0.7], [0.0, 0.0]],
        ...                       [[0.2, 0.8], [0.9, 0.1]]])
        >>> lens = torch.tensor([0.51, 1.0])
        >>> blank_id = 0
        >>> ctc_greedy_decode(probs, lens, blank_id)
        [[1], [1]]

    Author:
        Abdelwahab HEBA 2020
    """
    hidden=None
    list_outs=[]
    #### Prepare Blank prediction
    input_PN=torch.ones((F.size(0),1),device=F.device,dtype=torch.int64)*blank_id
    out_PN=input_PN
    hidden=None
    for layer in decode_network_lst:
        if layer.__class__.__name__ == "RNN":
            out_PN, hidden = layer(out_PN)
        else:
            if layer.__class__.__name__ == "Embedding":
                out_PN = layer(out_PN)
            else:
                out_PN = layer(out_PN)
    #### For each time step
    for t_step in range(F.size(1)):
        # Join predictions (TN & PN)
        out = Tjoint(F[:,t_step,:].unsqueeze(1),out_PN)
        # Classifiers layers
        for layer in classif_network_lst:
            out = layer(out)
        list_outs.append(out)
        prob_targets, positions = torch.max(out.log_softmax(dim=-1).squeeze(1),dim=1)
        #print(positions)
        #print(positions.shape)
        have_update_hyp=[]
        for i in range(positions.size(0)):
            if positions[i] != blank_id and positions[i] != input_PN[i][0]:
                input_PN[i][0]=positions[i]
                #out_PN=input_PN
                have_update_hyp.append(i)
        if len(have_update_hyp)>0:
            out_update_hyp=input_PN[have_update_hyp,:]
            # for LSTM Hidden
            if len(hidden) > 1:
                hidden0_update_hyp=hidden[0][:,have_update_hyp,:]
                hidden1_update_hyp=hidden[1][:,have_update_hyp,:]
                hidden_update_hyp=(hidden0_update_hyp,hidden1_update_hyp)
            else:
                hidden_update_hype=hidden[:,have_update_hyp,:]
            
            for layer in decode_network_lst:
                if layer.__class__.__name__ == "RNN":
                    out_update_hyp, hidden_update_hyp = layer(out_update_hyp,hidden_update_hyp)
                else:
                    out_update_hyp = layer(out_update_hyp)
            
            out_PN[have_update_hyp]=out_update_hyp
            if len(hidden) > 1:
                hidden[0][:,have_update_hyp,:]=hidden_update_hyp[0]
                hidden[1][:,have_update_hyp,:]=hidden_update_hyp[1]
            else:
                hidden[:,have_update_hyp,:]=hidden_update_hyp
    return torch.cat(list_outs,dim=1)
