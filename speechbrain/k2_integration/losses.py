# Copyright      2023 the University of Edinburgh (Zeyu Zhao)
import k2
import torch


def ctc_k2(log_probs,
           input_lens,
           graph_compiler,
           texts,
           reduction="mean",
           beam_size=10,
           use_double_scores=True,
           is_training=True,
           ):
    input_lens = (input_lens * log_probs.shape[1]).round().int()

    batch_size = log_probs.shape[0]

    supervision_segments = torch.tensor(
        [[i, 0, input_lens[i]] for i in range(batch_size)],
        device="cpu",
        dtype=torch.int32,
    )

    decoding_graph = graph_compiler.compile(texts)
    tids = graph_compiler.lexicon.texts2tids(texts)
    target_lens = torch.tensor([len(t) for t in tids], device=log_probs.device, dtype=torch.long)

    dense_fsa_vec = k2.DenseFsaVec(
        log_probs,
        supervision_segments,
    )
    loss = k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec,
        target_lengths=target_lens,
        output_beam=beam_size,
        reduction=reduction,
        use_double_scores=use_double_scores,
    )

    assert loss.requires_grad == is_training

    return loss


def k2_ctc(log_probs, targets, input_lens, target_lens, reduction="mean"):
    """CTC loss implemented with k2. Please make sure that k2 has been installed properly.
    Note that the blank index must be 0 in this implementation.

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape [batch, time, chars].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len]
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    reduction : str
        What reduction to apply to the output. 'mean', 'sum', 'none'.
        See k2.ctc_loss for 'mean', 'sum', 'none'.
    """
    input_lens = (input_lens * log_probs.shape[1]).round().int()
    target_lens = (target_lens * targets.shape[1]).round().int()
    batch_size = log_probs.shape[0]

    max_token_id = log_probs.shape[-1] - 1
    ctc_topo = k2.ctc_topo(max_token_id, modified=False,
                           device=log_probs.device)

    # convert targets to k2.FsaVec
    labels = [targets[i, :target_lens[i]].tolist()
              for i in range(len(target_lens))]
    label_fsas = k2.linear_fsa(labels, device=log_probs.device)

    labels_fsas_with_self_loops = k2.remove_epsilon_and_add_self_loops(
        label_fsas)

    labels_fsas_with_self_loops = k2.arc_sort(labels_fsas_with_self_loops)

    graph = k2.compose(ctc_topo, labels_fsas_with_self_loops,
                       treat_epsilons_specially=False)

    assert graph.requires_grad is False

    supervision_segments = torch.tensor(
        [[i, 0, input_lens[i]] for i in range(batch_size)],
        device="cpu",
        dtype=torch.int32,
    )

    dense_fsa_vec = k2.DenseFsaVec(
        log_probs,
        supervision_segments,
    )

    loss = k2.ctc_loss(
        decoding_graph=graph,
        dense_fsa_vec=dense_fsa_vec,
        output_beam=10,
        reduction=reduction,
        target_lengths=target_lens,
    )
    return loss
