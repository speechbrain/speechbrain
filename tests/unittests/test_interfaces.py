def test_load_audio(tmpdir):
    from speechbrain.pretrained.interfaces import Pretrained
    verification = Pretrained.from_hparams(
        source = "speechbrain/spkrec-ecapa-voxceleb",
        savedir = tmpdir.mkdir("savedir") )
    audio1 = verification.load_audio("samples/voxceleb_samples/wav/id10002/xTV-jFAUKcw/00001.wav")
    audio2 = verification.load_audio("samples/voxceleb_samples/wav/id10001/1zcIwhmdeo4/00001.wav")
    assert not (audio1 == audio2).all()