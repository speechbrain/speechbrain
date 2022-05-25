from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml


def test_load_audio(tmpdir):
    from speechbrain.pretrained.interfaces import Pretrained

    hparams_file = (
        Path("tests")
        / "integration"
        / "neural_networks"
        / "ASR_seq2seq"
        / "hyperparams.yaml"
    )
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    pretrained = Pretrained(hparams["modules"], hparams)
    audio1 = pretrained.load_audio(
        "samples/voxceleb_samples/wav/id10002/xTV-jFAUKcw/00001.wav"
    )
    audio2 = pretrained.load_audio(
        "samples/voxceleb_samples/wav/id10001/1zcIwhmdeo4/00001.wav"
    )
    assert not torch.equal(audio1, audio2)
