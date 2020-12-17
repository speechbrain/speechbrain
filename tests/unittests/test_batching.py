def test_paddedbatch():
    from speechbrain.data_io.batch import PaddedBatch
    import torch

    batch = PaddedBatch(
        [
            {
                "id": "ex1",
                "foo": torch.Tensor([1.0]),
                "bar": torch.Tensor([1.0, 2.0, 3.0]),
            },
            {
                "id": "ex2",
                "foo": torch.Tensor([2.0, 1.0]),
                "bar": torch.Tensor([2.0]),
            },
        ]
    )
    batch.to(dtype=torch.half)
    assert batch.foo.data.dtype == torch.half
    assert batch["foo"][1].dtype == torch.half
    assert batch.bar.lengths.dtype == torch.half
    assert batch.foo.data.shape == torch.Size([2, 2])
    assert batch.bar.data.shape == torch.Size([2, 3])
    ids, foos, bars = batch
    assert ids == ["ex1", "ex2"]
