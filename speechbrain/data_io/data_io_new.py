import torchaudio
import torch


def read_audio_example(example):
    """
    example with {"supervision": {"start" : x "stop": y ... etc}
                 {"waveforms": {"files": , channels,
    """

    waveforms = []
    for f in example["waveforms"]["files"]:
        if (
            "start" not in example["supervision"].keys()
            or "stop" not in example["supervision"].keys()
        ):
            tmp = torchaudio.load(f)
            waveforms.append(tmp)
        else:
            num_frames = (
                example["supervision"]["stop"] - example["supervision"]["start"]
            )
            offset = example["supervision"]["start"]
            tmp, fs = torchaudio.load(f, num_frames=num_frames, offset=offset)
            waveforms.append(tmp)

    # we pad to same length here or on dataloader ?

    return torch.cat(waveforms, 0)  # torch.cat(waveforms, 0)
