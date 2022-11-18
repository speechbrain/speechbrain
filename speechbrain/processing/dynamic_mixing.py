"""Implementation of dynamic mixing for speech separation

Authors
    * Samuele Cornell 2021
    * Cem Subakan 2021
    * Martin Kocour 2022
"""

from speechbrain.processing.signal_processing import reverberate

import torch
import torchaudio
import numpy as np
import numbers
import random
import warnings
import uuid
import logging
import os
import pyloudnorm  # WARNING: External dependency

from dataclasses import dataclass, fields
from typing import Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DynamicMixingConfig:
    num_spkrs: Union[int, list] = 2
    num_spkrs_prob: Optional[list] = None  # default: uniform distribution
    overlap_ratio: Union[int, list] = 1.0
    overlap_prob: Optional[list] = None  # default: uniform distribution
    audio_norm: bool = True  # normalize loudness of sources
    audio_min_loudness: float = -33.0  # dB
    audio_max_loudness: float = -25.0  # dB
    audio_max_amp: float = 0.9  # max amplitude in mixture and sources
    noise_add: bool = False
    noise_prob: float = 1.0
    noise_files: Optional[
        Union[str, list]
    ] = None  # list or path to list of files separated by newline
    # noise_snr: float = 20.0 # dB TODO
    noise_min_loudness: float = -33.0 - 5
    noise_max_loudness: float = -25.0 - 5
    white_noise_add: bool = True
    white_noise_mu: float = 0.0
    white_noise_var: float = 1e-7
    rir_add: bool = False
    rir_prob: float = 1.0
    rir_files: Optional[
        Union[str, list]
    ] = None  # list or path to list of files separated by newline
    sample_rate: int = 16000
    min_source_len: int = 16000
    max_source_len: int = 320000

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

        if self.num_spkrs_prob is None:
            self.num_spkrs_prob = [
                1 / len(self.num_spkrs) for _ in range(len(self.num_spkrs))
            ]

        if self.overlap_prob is None:
            self.overlap_prob = [
                1 / len(self.overlap_ratio)
                for _ in range(len(self.overlap_ratio))
            ]

        if isinstance(self.noise_files, list):
            paths = []
            try:
                for path_lst in self.noise_files:
                    paths.extend(parse_paths(path_lst))
                self.noise_files = paths
            except AssertionError:
                logger.info("Assuming that noise files are actual files")

        if isinstance(self.noise_files, str):
            self.noise_files = parse_paths(self.noise_files)

        if isinstance(self.rir_files, list):
            paths = []
            try:
                for path_lst in self.rir_files:
                    paths.extend(parse_paths(path_lst))
                self.rir_files = paths
            except AssertionError:
                logger.info("Assuming that RIR files are actual files")

        if isinstance(self.rir_files, str):
            self.rir_files = parse_paths(self.rir_files)

        assert len(self.num_spkrs) == len(self.num_spkrs_prob)
        assert len(self.overlap_ratio) == len(self.overlap_prob)
        assert (self.noise_add and self.noise_files) or not self.noise_add
        assert (self.rir_add and self.rir_files) or not self.rir_add


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
                f"Expected at least {config.num_spkrs} spkrs in spkr_files"
            )

        self.spkr_files = spkr_files
        self.config = config

        total = sum(map(len, self.spkr_files.values()))
        self.spkr_names = [x for x in spkr_files.keys()]
        self.spkr_weights = [
            len(spkr_files[x]) / total for x in self.spkr_names
        ]

        tmp_file, _ = next(iter(spkr_files.values()))[0]
        self.orig_sr = torchaudio.info(tmp_file).sample_rate
        if self.orig_sr != config.sample_rate:
            self.resampler = torchaudio.transforms.Resample(
                self.orig_sr, self.config.sample_rate
            )
            logger.warning(
                "Audio files has different sampling rate (%d != %d)!",
                self.orig_sr,
                self.config.sample_rate,
            )
        else:
            self.resampler = lambda x: x

        self.meter = None
        if self.config.audio_norm:
            self.meter = pyloudnorm.Meter(self.config.sample_rate)

        self.dataset = None  # used for inner database

    @classmethod
    def from_didataset(cls, dataset, config, wav_key=None, spkr_key=None):
        if wav_key is None:
            raise ValueError("Provide valid wav_key for dataset item")

        if spkr_key is None:
            files = [(d[wav_key], idx) for idx, d in enumerate(dataset)]
            dmdataset = cls.from_wavs(files, config)
        else:
            spkr_files = {}
            for idx, d in enumerate(dataset):
                spkr_files[d[spkr_key]] = spkr_files.get(d[spkr_key], [])
                spkr_files[d[spkr_key]].append((d[wav_key], idx))
            dmdataset = cls(spkr_files, config)
        dmdataset.set_dataset(dataset)
        return dmdataset

    @classmethod
    def from_wavs(cls, wav_file_list, config):
        spkr_files = {}
        spkr = 0
        # we assume that each wav is coming from different spkr
        for wavfile in wav_file_list:
            spkr_files[f"spkr{spkr}"] = [(wavfile, None)]
            spkr += 1

        return cls(spkr_files, config)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def generate(self):
        """Generate new audio mixture

        Returns:
          - mixture
          - mixed spkrs
          - used overlap ratios
          - padded sources
          - noise
          - data
        """

        mix_info = {
            "num_spkrs": 0,
            "speakers": [],
            "sources": [],
            "source_lengths": [],
            "overlap_ratios_paddings": [],
            "noise": None,
            "rir": None,
            "data": [],
            "duration": 0.0,
        }

        mix_info["num_spkrs"] = np.random.choice(
            self.config.num_spkrs, p=self.config.num_spkrs_prob
        ).item()

        if mix_info["num_spkrs"] <= 0:
            length = random.randint(
                self.config.min_source_len, self.config.max_source_len
            )
            sources = [torch.zeros(length)]
            mixture, sources, noise = self.__postprocess__(
                sources[0], sources, mix_info=mix_info
            )
            mix_info["speakers"].append("no-spkr")
            # return mixture, ["noise"], [(0.0, [(0, 0)])], sources, noise, []
            return mixture, sources, noise, mix_info

        mix_info["speakers"] = np.random.choice(
            self.spkr_names,
            mix_info["num_spkrs"],
            replace=False,
            p=self.spkr_weights,
        )
        mix_info["speakers"] = list(mix_info["speakers"])

        rir = None
        if self.config.rir_add and random.uniform(0, 1) < self.config.rir_prob:
            mix_info["rir"] = np.random.choice(self.config.rir_files)
            rir, fs = torchaudio.load(mix_info["rir"])
            assert (
                fs == self.orig_sr
            ), f"{self.orig_sr} Hz sampling rate expected, but found {fs}"
            rir = self.resampler(rir)
            rir = rir[0]

        sources = []
        source_idxs = []
        fs = None
        for spkr in mix_info["speakers"]:
            spkr_idx = random.randint(0, len(self.spkr_files[spkr]) - 1)
            src_file, src_idx = self.spkr_files[spkr][spkr_idx]
            mix_info["sources"].append(src_file)
            src_audio, fs = torchaudio.load(src_file)
            assert (
                fs == self.orig_sr
            ), f"{self.orig_sr} Hz sampling rate expected, but found {fs}"
            src_audio = self.resampler(src_audio)
            src_audio = src_audio[0]  # Support only single channel
            # use same RIR for all sources
            src_audio = self.__prepare_source__(
                src_audio, rir, mix_info=mix_info
            )
            sources.append(src_audio)
            source_idxs.append(src_idx)

        sources, source_idxs, src_files, src_lens = zip(
            *sorted(
                zip(sources, source_idxs, mix_info['sources'], mix_info["source_lengths"]),
                key=lambda x: x[0].size(0),
                reverse=True,
            )
        )
        mix_info['sources'] = list(src_files)
        mix_info['source_lengths'] = list(src_lens)

        mixture = sources[0].detach().clone()  # longest audio
        padded_sources = [sources[0]]
        for i in range(1, len(sources)):
            src = sources[i]
            ratio = np.random.choice(
                self.config.overlap_ratio, p=self.config.overlap_prob
            ).item()
            overlap_samples = int(src.size(0) * ratio)

            mixture, padded_tmp, paddings = mix(src, mixture, overlap_samples)
            # padded sources are returned in same order
            mix_info["overlap_ratios_paddings"].append((ratio, paddings))

            # previous padded_sources are shorter
            padded_sources = __pad_sources__(
                padded_sources,
                [paddings[1] for _ in range(len(padded_sources))],
            )
            padded_sources.append(padded_tmp[0])
        mixture, padded_source, noise = self.__postprocess__(
            mixture, padded_sources, mix_info=mix_info
        )

        if self.dataset is not None:
            mix_info["data"] = [self.dataset[idx] for idx in source_idxs]

        return mixture, padded_sources, noise, mix_info

    def __prepare_source__(self, source, rir, is_noise=False, mix_info={}):

        # cut the source to a random length
        length = random.randint(
            self.config.min_source_len, self.config.max_source_len
        )
        # TODO: length shortening is huge problem for ASR

        if not is_noise:
            # noise is automatically adjusted to the mixture size
            source = source[:length]
            mix_info["source_lengths"] = mix_info.get("source_lengths", [])
            mix_info["source_lengths"].append(length)

        if self.config.audio_norm:
            # normalize loudness and apply random gain
            source = normalize(
                source,
                self.meter,
                self.config.audio_min_loudness
                if not is_noise
                else self.config.noise_min_loudness,
                self.config.audio_max_loudness
                if not is_noise
                else self.config.noise_max_loudness,
                self.config.audio_max_amp,
            )

        # add reverb
        if not is_noise and rir is not None:
            # noise is not reverberated
            reverberate(source, rir)
        return source

    def __postprocess__(self, mixture, sources, mix_info={}):
        # add noise
        noise = None
        if (
            self.config.noise_add
            and random.uniform(0, 1) < self.config.noise_prob
        ):
            mix_info["noise"] = np.random.choice(self.config.noise_files)
            noise, fs = torchaudio.load(mix_info["noise"])
            assert (
                fs == self.orig_sr
            ), f"{self.orig_sr} Hz sampling rate expected, but found {fs}"
            noise = self.resampler(noise)
            noise = self.__prepare_source__(
                noise[0], None, is_noise=True, mix_info=mix_info
            )
            noise = noise.repeat(
                mixture.size(0) // noise.size(0) + 1
            )  # extend the noise

            noise = noise[: mixture.size(0)]
            mixture += noise

        # replace zeros with small gaussian noise
        if self.config.white_noise_add:
            white_noise = np.random.normal(
                self.config.white_noise_mu,
                self.config.white_noise_var,
                size=mixture.size(0),
            )
            white_noise = torch.from_numpy(white_noise)
            mixture += white_noise

        # normalize gain
        # this should be the final step
        mix_max_amp = mixture.abs().max().item()
        gain = 1.0
        if mix_max_amp > self.config.audio_max_amp:
            gain = self.config.audio_max_amp / mix_max_amp

        mixture = gain * mixture
        sources = list(map(lambda src: gain * src, sources))
        if noise is not None:
            noise = gain * noise

        mix_info["duration"] = mixture.size(0) / self.config.sample_rate
        return mixture, sources, noise

    def __len__(self):
        return sum(map(len, self.spkr_files.values()))  # dict of lists

    def __getitem__(self, idx):
        # TODO: Refactor completly
        mix, srcs, noise, mix_info = self.generate()

        for i in range(3 - len(srcs)):
            srcs.append(torch.zeros(mix.size(0)))

        if idx is None:
            idx = uuid.uuid4()
        mix_id = (
            str(idx)
            + "_"
            + "-".join(mix_info["speakers"])
            + "_overlap"
            + "-".join(map(lambda x: f"{x[0]:.2f}", mix_info["overlap_ratios_paddings"]))
        )
        # "id", "mix_sig", "s1_sig", "s2_sig", "s3_sig", "noise_sig"
        dct = {
            "id": mix_id,
            "mix_sig": mix,
            "s1_sig": srcs[0],
            "s2_sig": srcs[1],
            "s3_sig": srcs[2],
            "noise_sig": noise if noise is not None else torch.zeros(mix.size(0)),
            "data": mix_info["data"],
        }

        return dct


def normalize(audio, meter, min_loudness=-33, max_loudness=-25, max_amp=0.9):
    """This function normalizes the loudness of audio signal"""
    audio = audio.numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_loudness = meter.integrated_loudness(audio)
        target_loudness = random.uniform(min_loudness, max_loudness)
        # TODO: pyloudnorm.normalize.loudness could be replaced by rescale from SB
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
        lpad = np.random.choice(range(n_diff)).item() if n_diff > 0 else 0
        rpad = n_diff - lpad
        paddings = [(lpad, rpad), (0, 0)]
    elif overlap_samples > 0:
        # partial overlap
        start_short = np.random.choice([True, False])  # start with short
        n_total = n_long + n_short - overlap_samples
        if start_short:
            paddings = [(0, n_total - n_short), (n_total - n_long, 0)]
        else:
            paddings = [(n_total - n_short, 0), (0, n_total - n_long)]
    else:
        # no-overlap
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


def parse_paths(file):
    paths = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            assert len(line.split()) == 1
            if not line.startswith("/"):
                # relative paths
                line = os.path.join(os.path.dirname(file), line)
            paths.append(line)
    return paths
