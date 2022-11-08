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
import uuid
import pyloudnorm  # WARNING: External dependency

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(kw_only=True)
class DynamicMixingConfig:
    num_spkrs: Union[int, list] = 2
    overlap_ratio: Union[int, list] = 1.0
    audio_norm: bool = True
    audio_min_loudness: float = -33.0
    audio_max_loudness: float = -25.0
    audio_max_amp: float = 0.9

    @classmethod
    def from_hparams(cls, hparams):
        config = {}
        for fld in fields(cls):
            config[fld.name] = hparams.get(fld.name, fld.default)
        return cls(**config)

    def __post_init__(self):
        if isinstance(self.num_spkrs, int):
            self.num_spkrs = [self.num_spkrs]

        if isinstance(self.overlap_ratio, numbers.Real):
            self.overlap_ratio = [self.overlap_ratio]


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
    >>> config = DynamicMixingConfig.from_hparams(hparams)
    >>> dm_dataset = DynamicMixixingDataset.from_didataset(data, config, "wav_file", "spkr")
    >>> mixture, spkrs, ratios, sources = dm_dataset.generate()

    Arguments
    ---------
    spkr_files : dict
    config: DynamicMixingConfig
    """

    def __init__(self, spkr_files, config):
        if len(spkr_files.keys()) < max(config.num_spkrs):
            raise ValueError(
                f"Expected at least {num_spkrs} spkrs in spkr_files"
            )

        self.num_spkrs = config.num_spkrs
        self.overlap_ratio = config.overlap_ratio
        self.normalize_audio = config.audio_norm
        self.spkr_files = spkr_files

        tmp_file = next(iter(spkr_files.values()))[0]
        self.sampling_rate = torchaudio.info(tmp_file).sample_rate

        self.meter = None
        if self.normalize_audio:
            self.meter = pyloudnorm.Meter(self.sampling_rate)

        if min(config.num_spkrs) <= 0:
            lengths = map(torchaudio.info, iter(self._all_files()))

        self.config = config

    @classmethod
    def from_didataset(cls, dataset, config, wav_key=None, spkr_key=None):
        if wav_key is None:
            raise ValueError("Provide valid wav_key for dataset item")

        if spkr_key is None:
            files = [d[wav_key] for d in dataset]
            return cls.from_wavs(files, config)
        else:
            spkr_files = {}
            for d in dataset:
                spkr_files[d[spkr_key]] = spkr_files.get(
                    d[spkr_key], []
                ).append(d[wav_key])

            return cls(spkr_files, config)

    @classmethod
    def from_wavs(cls, wav_file_list, config):
        spkr_files = {}
        spkr = 0
        # we assume that each wav is coming from different spkr
        for wavfile in wav_file_list:
            spkr_files[f"spkr{spkr}"] = [wavfile]
            spkr += 1

        return cls(spkr_files, config)

    def generate(self, wavfile=None):
        n_spkrs = np.random.choice(self.config.num_spkrs)
        if n_spkrs <= 0:
            # TODO: how long mixture?
            raise NotImplementedError("Expect at least 1 source")

        mix_spkrs = np.random.choice(list(self.spkr_files.keys()), n_spkrs)

        sources = []
        fs = None
        for spkr in mix_spkrs:
            src_file = np.random.choice(self.spkr_files[spkr])
            src_audio, fs = torchaudio.load(src_file)
            src_audio = src_audio[0]  # Support only single channel

            if fs != self.sampling_rate:
                raise RuntimeError(
                    f"{self.sampling_rate} Hz sampling rate expected, but found {fs}"
                )

            src_audio = self.__prepare_source__(src_audio)
            sources.append(src_audio)

        sources = sorted(sources, key=lambda x: x.size(0), reverse=True)
        mixture = sources[0]  # longest audio
        overlap_ratios = []
        padded_sources = []
        for i in range(1, len(sources)):
            src = sources[i]
            ratio = np.random.choice(self.config.overlap_ratio)
            overlap_samples = int(src.size(0) * ratio)

            mixture, padded_tmp, paddings = mix(src, mixture, overlap_samples)
            # padded sources are returned in same order
            overlap_ratios.append((ratio, overlap_samples))

            # for sources>2 previous padded_sources are shorter
            padded_sources = __pad_sources__(
                padded_sources,
                [paddings[1] for _ in range(len(padded_sources))],
            )
            if len(padded_sources) == 0:
                padded_sources.append(padded_tmp[1])

            padded_sources.append(padded_tmp[0])
        mixture, padded_source = self.__postprocess__(mixture, padded_sources)
        if wavfile:
            torchaudio.save(mixture, wavfile)
        return mixture, mix_spkrs, overlap_ratios, padded_sources

    def __prepare_source__(self, source):
        if self.normalize_audio:
            source = normalize(
                source,
                self.meter,
                self.config.audio_min_loudness,
                self.config.audio_max_loudness,
                self.config.audio_max_amp,
            )
        # TODO: Random gain
        # TODO: add reverb
        # TODO: add noise
        return source

    def __postprocess__(self, mixture, sources):
        # modify gain
        mix_max_amp = mixture.abs().max().item()
        gain = 1.0
        if mix_max_amp > self.config.audio_max_amp:
            gain = self.config.audio_max_amp / mix_max_amp

        mixture = gain * mixture
        sources = map(lambda src: gain * src, sources)
        # TODO: replace zeros with small noise
        return mixture, sources

    def __len__(self):
        return sum(map(len, self.spkr_files.values()))  # dict of lists

    def __getitem__(self, idx):
        # TODO: Refactor completly
        mix, spkrs, ratios, srcs = self.generate()
        if len(srcs) != 2:
            raise NotImplementedError("getitem supports exactly 2 sources")

        if idx is None:
            idx = uuid.uuid4()
        mix_id = (
            str(idx)
            + "_"
            + "-".join(spkrs)
            + "_overlap"
            + "-".join(map(lambda x: f"{x[0]:.2f}", ratios))
        )
        # "id", "mix_sig", "s1_sig", "s2_sig", "s3_sig", "noise_sig"
        return (
            mix_id,
            mix,
            srcs[0],
            srcs[1],
            torch.zeros(mix.size(0)),
            torch.zeros(mix.size(0)),
        )


def normalize(audio, meter, min_loudness=-33, max_loudness=-25, max_amp=0.9):
    """This function normalizes the loudness of audio signal"""
    audio = audio.numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_loudness = meter.integrated_loudness(audio)
        # if is_noise:
        #    target_loudness = random.uniform(
        #        MIN_LOUDNESS - 5, MAX_LOUDNESS - 5
        #    )
        # else:
        #    target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        target_loudness = random.uniform(min_loudness, max_loudness)
        signal = pyloudnorm.normalize.loudness(
            audio, c_loudness, target_loudness
        )

        # check for clipping
        if np.max(np.abs(signal)) >= 1:
            signal = signal * max_amp / np.max(np.abs(signal))

    return torch.from_numpy(signal)


def mix(src1, src2, overlap_samples):
    """Mix two audio samples"""
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

    paddings = []
    if overlap_samples >= n_short:
        # full overlap
        #
        # short |        ++++++        |
        # long  |----------------------|
        # sum   |--------++++++--------|
        #        <-lpad->      <-rpad->
        #
        lpad = np.random.choice(range(n_diff)) if n_diff > 0 else 0
        rpad = n_diff - offset
        paddings = [(lpad, rpad), (0, 0)]
    elif overlap_samples > 0:
        # partial overlap
        #
        # short | +++++++       |
        # long  |    -----------|
        # sum   | +++++++-------|
        #
        start_short = np.random.choice([True, False])  # start with short
        n_total = n_long + n_short - overlap_samples
        if start_short:
            paddings = [(0, n_total - n_short), (n_total - n_long, 0)]
        else:
            paddings = [(n_total - n_short, 0), (0, n_total - n_long)]
    else:
        # no-overlap
        #
        # short | +++++         |
        # long  |         ------|
        # sum   | +++++   ------|
        #
        sil_between = abs(overlap_samples)
        start_short = np.random.choice([True, False])  # start with short
        if start_short:
            paddings = [(0, sil_between + n_long), (sil_between + n_short, 0)]
        else:
            paddings = [(sil_between + n_long, 0), (0, sil_between + n_short)]

    assert len(paddings) == 2
    src1, src2 = __pad_sources__([shorter_src, longer_src], paddings)
    sources = (
        torch.stack((src2, src1)) if swapped else torch.stack((src1, src2))
    )
    if swapped:
        paddings.reverse()

    mixture = torch.sum(sources, dim=0)
    return mixture, sources, paddings


def __pad_sources__(sources, paddings):
    result = []
    for src, (lpad, rpad) in zip(sources, paddings):
        nsrc = torch.cat((torch.zeros(lpad), src, torch.zeros(rpad)))
        result.append(nsrc)
    return result
