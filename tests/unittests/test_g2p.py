import torch
from torch.nn import functional as F


def _fake_probs(idx, count):
    result = torch.zeros(count)
    result[idx] = 2.0
    return F.softmax(result, dim=-1)


def _batch_fake_probs(indexes, count):
    p_seq = torch.zeros(indexes.shape + (count,))

    for batch_idx in range(len(indexes)):
        for item_idx in range(indexes.size(1)):
            p_seq[batch_idx, item_idx, :] = _fake_probs(
                indexes[batch_idx, item_idx], count
            )
    return p_seq


def test_subsequence_loss():
    from speechbrain.lobes.models.g2p.homograph import SubsequenceLoss
    from speechbrain.nnet.losses import nll_loss

    phn_dim = 4
    phns = torch.tensor(
        [
            [1, 2, 3, 0, 3, 1, 2, 1, 0, 3, 1, 2, 0, 0],
            [1, 2, 3, 1, 0, 3, 2, 1, 0, 1, 3, 2, 0, 0],
            [1, 2, 3, 1, 2, 3, 0, 1, 3, 1, 3, 2, 0, 1],
        ]
    )
    phn_lens = torch.tensor([12, 12, 14])

    preds = torch.tensor(
        [
            [1, 3, 3, 0, 3, 1, 2, 1, 0, 3, 1, 2],
            [1, 1, 2, 1, 0, 3, 2, 1, 0, 1, 3, 2],
            [3, 2, 1, 1, 2, 3, 0, 1, 3, 2, 3, 3],
        ]
    )

    p_seq = _batch_fake_probs(preds, phn_dim)

    start = torch.tensor([0, 5, 7])
    end = torch.tensor([3, 8, 12])

    word_phns_pred = torch.tensor(
        [[1, 3, 3, 0, 0], [3, 2, 1, 0, 0], [1, 3, 2, 3, 3]]
    )
    word_phns_ref = torch.tensor(
        [[1, 2, 3, 0, 0], [3, 2, 1, 0, 0], [1, 3, 1, 3, 2]]
    )
    word_p_seq = _batch_fake_probs(word_phns_pred, phn_dim)
    word_lengths = torch.tensor([3, 3, 5]) / 5

    loss = SubsequenceLoss(seq_cost=nll_loss, word_separator=0)
    loss_value = loss.forward(phns, phn_lens, p_seq.log(), start, end)
    loss_value_ref = nll_loss(word_p_seq.log(), word_phns_ref, word_lengths)
    assert loss_value == loss_value_ref


def test_extract_hyps():
    from speechbrain.lobes.models.g2p.homograph import SubsequenceExtractor

    phns = torch.tensor(
        [
            [1, 2, 3, 0, 3, 1, 2, 1, 0, 3, 1, 2, 0, 0],
            [1, 2, 3, 1, 0, 3, 2, 1, 0, 1, 3, 2, 0, 0],
            [1, 2, 3, 1, 2, 3, 0, 1, 3, 1, 3, 2, 0, 1],
        ]
    )
    hyps = [
        [1, 2, 3, 2, 0, 3, 1, 2, 1, 0, 3, 1, 2],
        [1, 2, 3, 0, 3, 2, 1, 0, 1, 3, 2],
        [1, 2, 3, 1, 2, 3, 0, 1, 3, 1, 3, 2, 0, 1],
    ]
    subsequence_phn_start = torch.tensor([4, 0, 7])
    ref_hyps = [[3, 1, 2, 1], [1, 2, 3], [1, 3, 1, 3, 2]]

    extractor = SubsequenceExtractor(word_separator=0)
    subsequence_hyps = extractor.extract_hyps(
        ref_seq=phns, hyps=hyps, subsequence_phn_start=subsequence_phn_start
    )
    assert subsequence_hyps == ref_hyps
