import os
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

np.random.seed(1234)


class WHAMDataset(IterableDataset):
    def __init__(self, data_dir, target_length=4, sample_rate=22050):
        self.data_dir = data_dir
        self.target_length = target_length
        self.sample_rate = sample_rate

        # Get a list of all WAV files in the WHAM data directory
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".wav")]

    def generate(self):
        while True:
            idx = np.random.choice([i for i in range(len(self.file_list))])
            file_path = os.path.join(self.data_dir, self.file_list[idx])

            waveform, sr = torchaudio.load(file_path)
            waveform = waveform.mean(0, keepdim=True)

            # Resample if needed
            if self.sample_rate != sr:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(
                    waveform
                )

            # Cut audio to the target length
            if waveform.shape[1] > self.target_length * self.sample_rate:
                start = 0
                end = int(self.target_length * self.sample_rate)
                waveform = waveform[:, start:end]

            zeros = (
                int(self.target_length * self.sample_rate) - waveform.shape[1]
            )
            waveform = F.pad(waveform, (0, zeros))

            yield waveform

    def __iter__(self):
        return iter(self.generate())


def combine_batches(clean, noise_loader):
    batch_size = clean.shape[0]

    noise = []
    for _ in range(batch_size):
        noise.append(next(noise_loader))
    noise = torch.stack(noise).to(clean.device)

    if noise.ndim == 3:
        noise = noise.squeeze(1)
    elif noise.ndim == 1:
        noise = noise[None]

    max_amplitude = torch.max(torch.abs(torch.cat([clean, noise], dim=0)))
    clean_l2 = (clean ** 2).sum(-1) ** 0.5
    noise_l2 = (noise ** 2).sum(-1) ** 0.5

    # Combine the batches at 0dB
    combined_batch = clean / clean_l2[..., None] + noise / noise_l2[..., None]
    combined_batch = combined_batch / torch.max(combined_batch, dim=1, keepdim=True).values

    # torchaudio.save("clean.wav", clean[0][None].cpu(), 16000)
    # torchaudio.save("noise.wav", noise[0][None].cpu(), 16000)
    # torchaudio.save("combined.wav", combined_batch[0][None].cpu(), 16000)
    # breakpoint()

    return combined_batch
