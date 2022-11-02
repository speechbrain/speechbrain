"""Implementation of dynamic mixing for speech separation

Authors
    * Samuele Cornell 2021
    * Cem Subakan 2021
    * Martin Kocour 2022
"""

from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils import data_pipeline

import torch
import torchaudio
import numpy as np


class DynamicMixingDataset(torch.utils.data.Dataset):
    """Dataset which creates mixtures from single-talker dataset

    Example
    -------
    >>> data = DynamicItemDataset.from_csv(csv_path)
    ... data = [
    ...     {
    ...         'wav_file': '/example/path/src1.wav',
    ...         'spkr': 'Gandalf',
    ...     },
    ...     {   'wav_file': '/example/path/src2.wav',
    ...         'spkr': 'Frodo',
    ...     }
    ... ]
    >>> dm_dataset = DynamicMixixingDataset.from_didataset(data, "wav_file", "spkr")

    Arguments
    ---------
    spkr_files : dict
    num_spkrs : Union[list, range, int], optional
    overlap_ratio : Union[list, range, int], optional
    """

    def __init__(
        self, spkr_files, num_spkrs=2, overlap_ratio=1.0, normalize_audio=True
    ):
        if isinstance(num_spkrs, int):
            num_spkrs = [num_spkrs]

        if len(spkr_files.keys) < max(num_spkrs):
            raise ValueError(
                f"Expected at least {num_spkrs} spkrs in spkr_files"
            )

        self.num_spkrs = num_spkrs
        self.overlap_ratio = overlap_ratio
        self.normalize_audio = normalize_audio

    @classmethod
    def from_didataset(cls, dataset, wav_key=None, spkr_key=None, **kwargs):
        if wav_key is None:
            raise ValueError("Provide valid wav_key for dataset item")

        if spkr_key is None:
            files = [d[wav_key] for d in dataset]
            return cls.from_wavs(files, **kwargs)
        else:
            spkr_files = {}
            for d in dataset:
                spkr_files[d[spkr_key]] = spkr_files.get(
                    d[spkr_key], []
                ).append(d[wav_key])

            return cls(spkr_files, **kwargs)

    @classmethod
    def from_wavs(cls, wav_file_list, **kwargs):
        spkr_files = {}
        spkr = 0
        # we assume that each wav is coming from different spkr
        for wavfile in wav_file_list:
            spkr_files[f"spkr{spkr}"] = wavfile
            spkr += 1

        return cls(spkr_files, **kwargs)

    def generate(self, wavfile=None):
        n_spkrs = np.random.choice(self.num_spkrs)
        mix_spkrs = np.random.choice(self.spkr_files.keys(), n_spkrs)

        sources = []
        fs = None
        for spkr in mix_spkrs:
            src_file = np.random.choice(self.spkr_files[spkr])
            src_audio, fs = torchaudio.load(src_file)
            if self.normalize_audio:
                src_audio = self.normalize(src_audio)

            sources.append(src_audio)

        if len(sources) == 0:
            raise NotImplementedError("Expect at least 1 source")

        sources = sorted(sources, key=len, reverse=True)
        mixture = sources[0]  # longest audio
        for i in range(1, len(sources)):
            src = sources[i]
            ratio = np.random.choice(self.overlap_ratios)
            overlap_samples = int(len(src) * ratio)
            mixture = mix(mixture, src, overlap_samples)

        # TODO: do some post-porcessing of the mixture, e.g. replace zeros with small noise
        if wavfile:
            torchaudio.save(mixture, wavfile)
        return mixture


def normalize(audio):
    raise NotImplementedError("Normalization is not supported yet")


def mix(longer_src, shorter_src, overlap_samples):
    mixture = None
    n_long = len(longer_src)
    n_short = len(shorter_src)
    n_diff = n_long - n_short
    assert n_diff >= 0

    if overlap_samples >= n_short:
        # full overlap
        #
        # short |              ++++++++++++              |
        # long  |----------------------------------------|
        # sum   |--------------++++++++++++--------------|
        #        <--offset---->            <---padding-->
        #
        offset = np.random.choice(range(n_diff))
        padding_len = n_diff - offset
        tmp = torch.cat(
            torch.zeros((1, offset)),
            shorter_src,
            torch.zeros((1, padding_len)),
        )
        return torch.sum(tmp, longer_src)
    elif overlap_samples > 0:
        # partial overlap
        start_short = np.random.choice([True, False])  # start with short
        n_total = n_long + n_short - overlap_samples
        if start_short:
            src1 = torch.cat(
                shorter_src,
                torch.zeros((1, n_total - n_short)),
            )
            src2 = torch.cat(torch.zeros((1, n_total - n_long)), longer_src)
            return torch.sum(src1, src2)
        else:
            src1 = torch.cat(longer_src, torch.zeros((1, n_total - n_long)))
            src2 = torch.cat(
                torch.zeros((1, n_total - n_short)),
                shorter_src,
            )
            return torch.sum(src1, src2)
    else:
        # no-overlap
        sil_between = torch.zeros((1, torch.abs(overlap_samples)))
        start_short = np.random.choice([True, False])  # start with short
        if start_short:
            return torch.cat(shorter_src, sil_between, longer_src)
        else:
            return torch.cat(longer_src, sil_between, shorter_src)
    return mixture
