"""Implementation of dynamic mixing for speech separation

Authors
    * Samuele Cornell 2021
    * Cem Subakan 2021
    * Martin Kocour 2022
"""

from speechbrain.processing.signal_processing import reverberate
from speechbrain.utils.data_pipeline import DataPipeline

import torch
import torchaudio
import numpy as np
import numbers
import random
import warnings
import uuid
import logging
import re
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
    noise_min_loudness: float = -33.0  # dB
    noise_max_loudness: float = -43.0  # dB
    reverb: bool = False
    reverb_sources: bool = True
    reverb_prob: float = 1.0
    white_noise_add: bool = True
    white_noise_std: float = 1e-4  # should be close to 0
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

        assert len(self.num_spkrs) == len(self.num_spkrs_prob)
        assert len(self.overlap_ratio) == len(self.overlap_prob)


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

    def __init__(
        self,
        spkr_files,
        config,
        noise_flist=None,
        rir_flist=None,
        replacements={},
        length=None,
    ):
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

        if self.config.noise_add:
            assert (
                noise_flist is not None
            ), "Provide `noise_flist` or use `config.noise_add=False`"
            self.noise_files = parse_noises(noise_flist, replacements)
            assert len(self.noise_files) > 0

        if self.config.reverb:
            assert (
                rir_flist is not None
            ), "Provide `rir_flist` or use `config.reverb=False`"
            self.rir_files = parse_noises(rir_flist, replacements)
            assert len(self.rir_files) > 0

        self.length = length
        self.dataset = None  # used for inner database
        self.pipeline = DataPipeline(
            [
                "id",
                "mixture",
                "sources",
                "noise",
                "rir",
                "original_data",
                "mix_info",
            ]
        )

    @classmethod
    def from_didataset(
        cls, dataset, config, wav_key=None, spkr_key=None, **kwargs
    ):
        if wav_key is None:
            raise ValueError("Provide valid wav_key for dataset item")

        if spkr_key is None:
            files = [(d[wav_key], idx) for idx, d in enumerate(dataset)]
            dmdataset = cls.from_wavs(files, config, **kwargs)
        else:
            spkr_files = {}
            for idx, d in enumerate(dataset):
                spkr_files[d[spkr_key]] = spkr_files.get(d[spkr_key], [])
                spkr_files[d[spkr_key]].append((d[wav_key], idx))
            dmdataset = cls(spkr_files, config, **kwargs)
        dmdataset.set_dataset(dataset)
        return dmdataset

    @classmethod
    def from_wavs(cls, wav_file_list, config, **kwargs):
        spkr_files = {}
        spkr = 0
        # we assume that each wav is coming from different spkr
        for wavfile in wav_file_list:
            spkr_files[f"spkr{spkr}"] = [(wavfile, None)]
            spkr += 1

        return cls(spkr_files, config, **kwargs)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def add_dynamic_item(self, func, takes=None, provides=None):
        self.pipeline.add_dynamic_item(func, takes, provides)

    def set_output_keys(self, keys):
        self.pipeline.set_output_keys(keys)

    def generate(self):
        """Generate new audio mixture"""

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
            mixture, _, noise, rir = self.__postprocess__(
                torch.zeros(length), [], mix_info=mix_info
            )
            mix_info["speakers"].append("no-spkr")
            return mixture, [], noise, rir, mix_info

        mix_info["speakers"] = np.random.choice(
            self.spkr_names,
            mix_info["num_spkrs"],
            replace=False,
            p=self.spkr_weights,
        )
        mix_info["speakers"] = list(mix_info["speakers"])

        sources = []
        source_idxs = []
        for spkr in mix_info["speakers"]:
            spkr_idx = random.randint(0, len(self.spkr_files[spkr]) - 1)
            src_file, src_idx = self.spkr_files[spkr][spkr_idx]
            mix_info["sources"].append(src_file)
            src_audio = self.__prepare_source__(src_file, mix_info=mix_info)
            sources.append(src_audio)
            source_idxs.append(src_idx)

        sources, source_idxs, src_files, src_lens = zip(
            *sorted(
                zip(
                    sources,
                    source_idxs,
                    mix_info["sources"],
                    mix_info["source_lengths"],
                ),
                key=lambda x: x[0].size(0),
                reverse=True,
            )
        )
        mix_info["sources"] = list(src_files)
        mix_info["source_lengths"] = list(src_lens)

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
        mixture, padded_sources, noise, rir = self.__postprocess__(
            mixture, padded_sources, mix_info=mix_info
        )

        if self.dataset is not None:
            mix_info["data"] = [self.dataset[idx] for idx in source_idxs]

        return mixture, padded_sources, noise, rir, mix_info

    def __prepare_source__(self, source_file, is_noise=False, mix_info={}):
        source, fs = torchaudio.load(source_file)
        assert (
            fs == self.orig_sr
        ), f"{self.orig_sr} Hz sampling rate expected, but found {fs}"
        source = self.resampler(source)
        source = source[0]  # Support only single channel

        if not is_noise:
            # cut the source to a random length
            length = random.randint(
                self.config.min_source_len, self.config.max_source_len
            )
            # TODO: length shortening is huge problem for ASR
            offset = random.randint(0, max(len(source) - length, 0))

            # noise is automatically adjusted to the mixture size
            source = source[offset : offset + length]
            mix_info["source_lengths"] = mix_info.get("source_lengths", [])
            mix_info["source_lengths"].append(length)

        if self.config.audio_norm:
            # normalize loudness and apply random gain
            mix_info["source_loudness"] = mix_info.get("source_loudness", [])
            if is_noise:
                loudness = random.uniform(
                    self.config.noise_min_loudness,
                    self.config.noise_max_loudness,
                )
                mix_info["noise_loudness"] = loudness
            else:
                loudness = random.uniform(
                    self.config.audio_min_loudness,
                    self.config.audio_max_loudness,
                )
                mix_info["source_loudness"].append(loudness)

            source = normalize(
                source,
                self.meter,
                loudness,
                self.config.audio_max_amp,
            )
        return source

    def __postprocess__(self, mixture, sources, mix_info={}):
        # reverberate
        rir = None
        if (
            self.config.reverb
            and random.uniform(0, 1) < self.config.reverb_prob
        ):
            mix_info["rir"] = np.random.choice(self.rir_files)
            rir, fs = torchaudio.load(mix_info["rir"])
            assert (
                fs == self.orig_sr
            ), f"{self.orig_sr} Hz sampling rate expected, but found {fs}"
            rir = self.resampler(rir)
            rir = rir[0]  # pick 1st channel

            mixture = reverberate(mixture, rir)
            if self.config.reverb_sources:
                sources = [reverberate(x, rir) for x in sources]

        # add noise
        noise = None
        if (
            self.config.noise_add
            and random.uniform(0, 1) < self.config.noise_prob
        ):
            mix_info["noise"] = np.random.choice(self.noise_files)
            noise = self.__prepare_source__(
                mix_info["noise"], is_noise=True, mix_info=mix_info
            )
            noise = noise.repeat(
                mixture.size(0) // noise.size(0) + 1
            )  # extend the noise

            noise = noise[: mixture.size(0)]
            mixture += noise

        # replace zeros with small gaussian noise
        if self.config.white_noise_add:
            mixture += torch.randn(mixture.shape) * self.config.white_noise_std
            sources = [
                src + torch.randn(mixture.shape) * self.config.white_noise_std
                for src in sources
            ]

        # normalize gain
        # this should be the final step
        mix_max_amp = mixture.abs().max().item()
        gain = 1.0
        if mix_max_amp > self.config.audio_max_amp:
            gain = self.config.audio_max_amp / mix_max_amp

        mixture = gain * mixture
        sources = [src * gain for src in sources]
        if noise is not None:
            noise = gain * noise

        mix_info["duration"] = mixture.size(0) / self.config.sample_rate
        return mixture, sources, noise, rir

    def __len__(self):
        if hasattr(self, "length") and self.length is not None:
            return self.length
        self.length = sum(map(len, self.spkr_files.values()))  # dict of lists
        return self.length

    def __getitem__(self, idx):
        mix, srcs, noise, rir, mix_info = self.generate()

        if idx is None:
            idx = uuid.uuid4()

        mix_id = (
            str(idx)
            + "_"
            + "-".join(mix_info["speakers"])
            + "_overlap"
            + "-".join(
                map(
                    lambda x: f"{x[0]:.2f}", mix_info["overlap_ratios_paddings"]
                )
            )
            + ("_rir" + mix_info["rir"])
            if mix_info["rir"]
            else "" + ("_noise" + mix_info["noise"])
            if mix_info["noise"]
            else ""
        )

        original_data = mix_info["data"]
        del mix_info["data"]

        dct = {
            "id": mix_id,
            "mixture": mix,
            "sources": srcs,
            "noise": noise,
            "rir": rir,
            "original_data": original_data,
            "mix_info": mix_info,
        }
        return self.pipeline(dct)


def normalize(audio, meter, loudness=-25, max_amp=0.9):
    """This function normalizes the loudness of audio signal"""
    audio = audio.numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_loudness = meter.integrated_loudness(audio)
        # TODO: pyloudnorm.normalize.loudness could be replaced by rescale from SB
        signal = pyloudnorm.normalize.loudness(audio, c_loudness, loudness)

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


def parse_noises(file, replacements={}):
    paths = []
    with open(file) as f:
        for line in f:
            path, room_id = parse_line(line)
            for ptrn, replacement in replacements.items():
                path = re.sub(ptrn, replacement, path)
            paths.append(path)
    return paths


def parse_line(line):
    args = line.strip().replace("--", "").split()
    path = args[-1]
    room_id = None
    if "room-id" in args:
        room_id = args[args.index("room-id") + 1]
    return path, room_id
