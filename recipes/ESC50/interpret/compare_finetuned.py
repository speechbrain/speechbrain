"""
To run:
    python compare_finetuned.py $PRETRAINED/reconstructions $FINETUNED/reconstructions
"""


from pathlib import Path
import matplotlib.pyplot as plt
import torchvision
import torchaudio
import torch
import sys
import os

if __name__ == "__main__":
    path_pretrained = Path(sys.argv[1])
    path_finetuned = Path(sys.argv[2])
    out_folder = "comparison/"

    os.makedirs(out_folder, exist_ok=True)
    for ut in sorted(path_pretrained.iterdir()):
        tmp = os.path.join(out_folder, ut.name)
        os.makedirs(tmp, exist_ok=True)

        ppath = path_pretrained.joinpath(ut.name, "interpretation.wav")
        fpath = path_finetuned.joinpath(ut.name, "interpretation.wav")
        opath = path_finetuned.joinpath(ut.name, "original.wav")

        pwav, sr = torchaudio.load(ppath)
        fwav, _ = torchaudio.load(fpath)
        owav, _ = torchaudio.load(opath)

        # dump waveform for this sample
        torchaudio.save(f"{os.path.join(tmp, 'finetuned.wav')}", fwav, sr)
        torchaudio.save(f"{os.path.join(tmp, 'no_finetuning.wav')}", pwav, sr)
        torchaudio.save(f"{os.path.join(tmp, 'original.wav')}", owav, sr)

        ppath = path_pretrained.joinpath(ut.name, "reconstructions.png")
        fpath = path_finetuned.joinpath(ut.name, "reconstructions.png")

        pspec = torchvision.io.read_image(str(ppath)).float() / 255
        fspec = torchvision.io.read_image(str(fpath)).float() / 255

        cat_ = torch.cat((pspec, fspec), dim=1)
        torchvision.utils.save_image(cat_, str(os.path.join(tmp, "spec.png")))


