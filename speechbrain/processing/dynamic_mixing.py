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
import numbers
import random
import warnings
import pyloudnorm # WARNING: External dependency


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
    >>> mixture, spkrs, ratios, sources = dm_dataset.generate()

    Arguments
    ---------
    spkr_files : dict
    num_spkrs : Union[list, range, int], optional
    overlap_ratio : Union[list, range, int], optional
    normalize_audio: bool, optional
    """

    def __init__(
        self, spkr_files, num_spkrs=2, overlap_ratio=1.0, normalize_audio=True
    ):
        if isinstance(num_spkrs, int):
            num_spkrs = [num_spkrs]

        if isinstance(overlap_ratio, numbers.Real):
            overlap_ratio = [overlap_ratio]

        if len(spkr_files.keys()) < max(num_spkrs):
            raise ValueError(
                f"Expected at least {num_spkrs} spkrs in spkr_files"
            )

        self.num_spkrs = num_spkrs
        self.overlap_ratio = overlap_ratio
        self.normalize_audio = normalize_audio
        self.spkr_files = spkr_files

        tmp_file = next(iter(spkr_files.values()))[0]
        self.sampling_rate = torchaudio.info(tmp_file).sample_rate

        self.meter = None
        if self.normalize_audio:
            self.meter = pyloudnorm.Meter(self.sampling_rate)

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
            spkr_files[f"spkr{spkr}"] = [wavfile]
            spkr += 1

        return cls(spkr_files, **kwargs)

    def generate(self, wavfile=None):
        n_spkrs = np.random.choice(self.num_spkrs)
        if n_spkrs <= 0:
            # TODO: how long mixture?
            raise NotImplementedError("Expect at least 1 source")

        mix_spkrs = np.random.choice(list(self.spkr_files.keys()), n_spkrs)

        sources = []
        fs = None
        for spkr in mix_spkrs:
            src_file = np.random.choice(self.spkr_files[spkr])
            src_audio, fs = torchaudio.load(src_file)

            if fs != self.sampling_rate:
                raise RuntimeError(
                    f"{self.sampling_rate} Hz sampling rate expected, but found {fs}"
                )

            if self.normalize_audio:
                src_audio = normalize(src_audio.squeeze().numpy(), self.meter)
                src_audio = src_audio.unsqueeze(0)

            sources.append(src_audio)

        sources = sorted(sources, key=lambda x: x.size(1), reverse=True)
        mixture = sources[0]  # longest audio
        overlap_ratios = []
        padded_sources = []
        for i in range(1, len(sources)):
            src = sources[i]
            ratio = np.random.choice(self.overlap_ratio)
            overlap_samples = int(src.size(1) * ratio)
            mixture, padded_tmp = mix(src, mixture, overlap_samples)
            overlap_ratios.append((ratio, overlap_samples))
            if len(padded_sources) == 0:
                padded_sources.append(padded_tmp[1].unsqueeze(0))
            padded_sources.append(
                padded_tmp[0].unsqueeze(0)
            )  # padded sources are returned in same order

        # TODO: do some post-porcessing of the mixture, e.g. replace zeros with small noise
        if wavfile:
            torchaudio.save(mixture, wavfile)
        return mixture, mix_spkrs, overlap_ratios, padded_sources


def normalize(audio, meter, MIN_LOUDNESS=-33, MAX_LOUDNESS=-25, MAX_AMP=0.9):
    """This function normalizes the audio signals for loudness"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_loudness = meter.integrated_loudness(audio)
        # if is_noise:
        #    target_loudness = random.uniform(
        #        MIN_LOUDNESS - 5, MAX_LOUDNESS - 5
        #    )
        # else:
        #    target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        signal = pyloudnorm.normalize.loudness(
            audio, c_loudness, target_loudness
        )

        # check for clipping
        if np.max(np.abs(signal)) >= 1:
            signal = signal * MAX_AMP / np.max(np.abs(signal))

    return torch.from_numpy(signal)


def mix(src1, src2, overlap_samples, channel_first=True):
    """Mix two audio samples"""
    src1, src2 = src1.squeeze(), src2.squeeze()
    n_diff = len(src1) - len(src2)
    swapped = False
    if n_diff >= 0:
        longer_src = src1
        shorter_src = src2
        swapped = True
    else:
        longer_src = src2
        shorter_src = src1
        n_diff = abs(n_diff)
    n_long = len(longer_src)
    n_short = len(shorter_src)

    if overlap_samples >= n_short:
        # full overlap
        #
        # short |              ++++++++++++              |
        # long  |----------------------------------------|
        # sum   |--------------++++++++++++--------------|
        #        <--offset---->            <---padding-->
        #
        offset = np.random.choice(range(n_diff)) if n_diff > 0 else 0
        padding_len = n_diff - offset
        src1 = torch.cat(
            (torch.zeros(offset), shorter_src, torch.zeros(padding_len))
        )
        src2 = longer_src
    elif overlap_samples > 0:
        # partial overlap
        #
        # short | +++++++             |
        # long  |    -----------------|
        # sum   | +++++++-------------|
        #
        start_short = np.random.choice([True, False])  # start with short
        n_total = n_long + n_short - overlap_samples
        if start_short:
            src1 = torch.cat((shorter_src, torch.zeros(n_total - n_short)))
            src2 = torch.cat((torch.zeros(n_total - n_long), longer_src))
        else:
            src1 = torch.cat((torch.zeros(n_total - n_short), shorter_src))
            src2 = torch.cat((longer_src, torch.zeros(n_total - n_long)))
    else:
        # no-overlap
        #
        # short | +++++            |
        # long  |         ---------|
        # sum   | +++++   ---------|
        #
        sil_between = abs(overlap_samples)
        start_short = np.random.choice([True, False])  # start with short
        if start_short:
            src1 = torch.cat((shorter_src, torch.zeros(sil_between + n_long)))
            src2 = torch.cat((torch.zeros(sil_between + n_short), longer_src))
        else:
            src1 = torch.cat((torch.zeros(sil_between + n_long), shorter_src))
            src2 = torch.cat((longer_src, torch.zeros(sil_between + n_short)))
    try:
        sources = (
            torch.stack((src2, src1)) if swapped else torch.stack((src1, src2))
        )
        mixture = torch.sum(sources, dim=0)
    except Exception as e:
        print(e)
        import pdb

        pdb.set_trace()

    if channel_first:
        mixture = mixture.unsqueeze(0)
    return mixture, sources
