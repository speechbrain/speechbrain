def test_pretrainer(tmpdir):
    import torch
    from torch.nn import Linear

    # save a model in tmpdir/original/model.ckpt
    first_model = Linear(32, 32)
    pretrained_dir = tmpdir / "original"
    pretrained_dir.mkdir()
    with open(pretrained_dir / "model.ckpt", "wb") as fo:
        torch.save(first_model.state_dict(), fo)

    # Make a new model and Pretrainer
    pretrained_model = Linear(32, 32)
    assert not torch.all(torch.eq(pretrained_model.weight, first_model.weight))
    from speechbrain.utils.parameter_transfer import Pretrainer

    pt = Pretrainer(
        collect_in=tmpdir / "reused", loadables={"model": pretrained_model}
    )
    pt.collect_files(default_source=pretrained_dir)
    pt.load_collected()
    assert torch.all(torch.eq(pretrained_model.weight, first_model.weight))
