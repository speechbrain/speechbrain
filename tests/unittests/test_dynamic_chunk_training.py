def test_dynchunktrain_sampler():
    from speechbrain.core import Stage
    from speechbrain.utils.dynamic_chunk_training import (
        DynChunkTrainConfig,
        DynChunkTrainConfigRandomSampler,
    )

    # sanity check and cover for the random sampler

    valid_cfg = DynChunkTrainConfig(16, 32)
    test_cfg = DynChunkTrainConfig(16, 32)

    sampler = DynChunkTrainConfigRandomSampler(
        chunkwise_prob=1.0,
        chunk_size_min=8,
        chunk_size_max=8,
        limited_left_context_prob=1.0,
        left_context_chunks_min=16,
        left_context_chunks_max=16,
        test_config=valid_cfg,
        valid_config=test_cfg,
    )

    sampled_train_config = sampler(Stage.TRAIN)
    assert sampled_train_config.chunk_size == 8
    assert sampled_train_config.left_context_size == 16

    assert sampler(Stage.VALID) == valid_cfg
    assert sampler(Stage.TEST) == test_cfg


def test_dynchunktrain():
    from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

    assert DynChunkTrainConfig(chunk_size=16).is_infinite_left_context()
    assert not DynChunkTrainConfig(
        chunk_size=16, left_context_size=4
    ).is_infinite_left_context()
