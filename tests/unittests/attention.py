import torch


def test_rel_pos_MHA():

    from speechbrain.nnet.attention import RelPosMultiHeadAttention

    bsz = 2
    emb_dim = 4
    k_len = [12, 10]
    q_len = [10, 12]
    bias = [True, False]
    head_dim = [3, None]

    for kl in k_len:
        for ql in q_len:
            for b in bias:
                for h in head_dim:
                    relpos = RelPosMultiHeadAttention(
                        emb_dim, 2, bias=b, head_dim=h
                    )
                    q = torch.rand((bsz, ql, emb_dim))
                    k = torch.rand((bsz, kl, emb_dim))
                    relpos(q, k, k)
