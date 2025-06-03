import torch
import torch.nn


def test_make_transformer_src_tgt_masks(device):

    from numpy import inf

    from speechbrain.lobes.models.transformer.TransformerASR import (
        make_transformer_src_tgt_masks,
    )
    from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

    config = DynChunkTrainConfig(chunk_size=4, left_context_size=3)

    x = torch.rand(1, 18)
    tgt = torch.rand(18, 18)
    tgt[:, 15:] = 0

    (
        _,
        tgt_key_padding_mask,
        src_mask,
        tgt_mask,
    ) = make_transformer_src_tgt_masks(x, tgt, dynchunktrain_config=config)

    # fmt: off
    # flake8: noqa
    expected_src_mask = torch.tensor(
        [[False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,],[True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,],[True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,],]
    )
    expected_key_padding_mask = torch.tensor(
        [[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,],]
    )

    expected_tgt_mask = torch.tensor(
        [[0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf,],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],]
    )
    # fmt: on

    assert torch.all(torch.eq(src_mask, expected_src_mask))
    assert torch.all(torch.eq(tgt_key_padding_mask, expected_key_padding_mask))
    assert torch.all(torch.eq(tgt_mask, expected_tgt_mask))


def test_make_transformer_src_mask(device):

    from speechbrain.lobes.models.transformer.TransformerASR import (
        make_transformer_src_mask,
    )
    from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

    x = torch.rand(1, 18)

    config = DynChunkTrainConfig(chunk_size=4, left_context_size=3)

    src_mask = make_transformer_src_mask(x, False, config)

    # fmt: off
    # flake8: noqa
    expected = torch.tensor(
        [[False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,],[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,],[True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,],[True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,],]
    )
    # fmt: on

    assert torch.all(torch.eq(src_mask, expected))


def test_get_lookahead_mask(device):

    from numpy import inf

    from speechbrain.lobes.models.transformer.Transformer import (
        get_lookahead_mask,
    )

    # fmt: off
    # flake8: noqa
    x = torch.LongTensor([[1, 1, 0], [2, 3, 0], [4, 5, 0]])

    out = get_lookahead_mask(x)

    expected = torch.tensor(
        [[0.0, -inf, -inf], [0.0, 0.0, -inf], [0.0, 0.0, 0.0]]
    )
    # fmt: on

    assert torch.all(torch.eq(out, expected))


def test_get_key_padding_mask(device):

    from speechbrain.lobes.models.transformer.Transformer import (
        get_key_padding_mask,
    )

    # fmt: off
    # flake8: noqa
    x = torch.LongTensor([[1, 1, 0], [2, 3, 0], [4, 5, 0]])

    out = get_key_padding_mask(x, 0)

    expected = torch.tensor(
        [[False, False, True], [False, False, True], [False, False, True]]
    )
    # fmt: on

    assert torch.all(torch.eq(out, expected))
