import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import IterableDataset

from speechbrain.utils.fetching import fetch

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

    clean_l2 = (clean**2).sum(-1) ** 0.5
    noise_l2 = (noise**2).sum(-1) ** 0.5

    # Combine the batches at 0dB
    combined_batch = clean / clean_l2[..., None] + noise / noise_l2[..., None]
    combined_batch = (
        combined_batch / torch.max(combined_batch, dim=1, keepdim=True).values
    )

    return combined_batch


def download_wham(wham_path: str):
    if len(os.listdir(wham_path)) != 0:
        return

    print("WHAM! is missing. Downloading WHAM!. This will take a while...")
    os.makedirs(wham_path, exist_ok=True)

    temp_path = os.path.join(wham_path, "temp_download_wham")

    # download the data
    fetch(
        "wham_noise.zip",
        "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com",
        savedir=temp_path,
    )

    # unpack the .zip file
    shutil.unpack_archive(os.path.join(temp_path, "wham_noise.zip"), wham_path)

    files = os.listdir(os.path.join(wham_path, "WHAM", "wham_noise"))
    for fl in files:
        shutil.move(
            os.path.join(wham_path, "WHAM", "wham_noise", fl), wham_path
        )

    # remove the unused datapath
    shutil.rmtree(temp_path)
    shutil.rmtree(os.path.join(wham_path, "WHAM"))

    print(f"WHAM! is downloaded in {wham_path}")


def prepare_wham(hparams):
    """Creates WHAM! dataset when needed."""
    if "wham_folder" not in hparams:
        return None

    if hparams["wham_folder"] is None:
        if hparams["add_wham_noise"]:
            raise Exception("You should specify wham_folder to add noise.")
        return None

    if hparams["add_wham_noise"]:
        # download WHAM! in specified folder
        download_wham(hparams["wham_folder"])

        config_sample_rate = hparams["sample_rate"]
        config_target_length = hparams["signal_length_s"]
        data_audio_folder = hparams["wham_audio_folder"]
        dataset = WHAMDataset(
            data_dir=data_audio_folder,
            target_length=config_target_length,
            sample_rate=config_sample_rate,
        )

        return dataset

    return None


if __name__ == "__main__":
    download_wham("wham_here")
