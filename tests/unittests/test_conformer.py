import torch


@torch.no_grad
def test_streaming_conformer_layer(device):
    """Test whether the Conformer encoder layer masking code path (used at train
    time) is equivalent to a real streaming scenario.
    """
    from speechbrain.lobes.models.transformer.Conformer import (
        ConformerEncoderLayer,
    )
    from speechbrain.lobes.models.transformer.TransformerASR import (
        make_transformer_src_mask,
    )
    from speechbrain.nnet.attention import RelPosEncXL
    from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

    TOLERATED_MEAN_ERROR = 1.0e-6

    bs, seq_len, num_feats = input_shape = 1, 24, 16
    config = DynChunkTrainConfig(chunk_size=8, left_context_size=1)

    assert (
        seq_len % config.chunk_size == 0
    ), "For this test, we assume the sequence length can evenly be divided"
    num_chunks = seq_len // config.chunk_size

    torch.manual_seed(1337)

    module = ConformerEncoderLayer(
        d_model=num_feats, d_ffn=num_feats * 2, nhead=1, kernel_size=5
    ).to(device=device)
    module.eval()

    pos_encoder = RelPosEncXL(num_feats).to(device=device)

    # build inputs
    test_input = torch.randn(input_shape, device=device)

    test_input_chunked = test_input.unfold(
        1, size=config.chunk_size, step=config.chunk_size
    )
    test_input_chunked = test_input_chunked.transpose(1, 3)
    assert test_input_chunked.shape == (
        bs,
        config.chunk_size,
        num_feats,
        num_chunks,
    ), "Test bug: incorrect shape for the chunked input?"

    # build the transformer mask for masked inference (dynchunktrain_config does
    # not suffice)
    src_mask = make_transformer_src_mask(
        test_input, dynchunktrain_config=config
    )

    # masked inference
    pos_embs_full = pos_encoder(test_input)
    out_mask_path, _out_attn = module(
        test_input,
        src_mask=src_mask,
        pos_embs=pos_embs_full,
        dynchunktrain_config=config,
    )

    # streaming inference
    mutable_ctx = module.make_streaming_context(
        config.left_context_size * config.chunk_size
    )
    output_chunks = []

    for i in range(num_chunks):
        chunk_in = test_input_chunked[..., i]

        # HACK due to pos embeddings
        pos_embs_dummy_input = chunk_in
        if mutable_ctx.mha_left_context is not None:
            pos_embs_dummy_input = torch.empty(
                (
                    bs,
                    config.chunk_size + mutable_ctx.mha_left_context.size(1),
                    num_feats,
                ),
                device=device,
            )

        pos_embs_chunk = pos_encoder(pos_embs_dummy_input)
        chunk_out, _chunk_attn = module.forward_streaming(
            chunk_in, mutable_ctx, pos_embs=pos_embs_chunk
        )
        output_chunks.append(chunk_out)

    out_stream_path = torch.cat(output_chunks, dim=1)

    # check output embedding differences
    abs_diff = (out_mask_path - out_stream_path).abs()

    assert torch.mean(abs_diff).item() < TOLERATED_MEAN_ERROR
