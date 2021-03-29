import os
import torch
import torchaudio
from tqdm import tqdm

rir_path = "/network/tmp1/subakany/whamr/wav8k/min/tr/rirs"
save_path = "/miniscratch/subakany/rir_wavs"

files = os.listdir(rir_path)

for fl in tqdm(files):
    rirs = torch.load(os.path.join(rir_path, fl))

    for i, mics in enumerate(rirs):
        for j, source in enumerate(mics):

            torchaudio.save(
                os.path.join(save_path, "{}_{}_".format(i, j) + fl[:-2]),
                source,
                8000,
            )
