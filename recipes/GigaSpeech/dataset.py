# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MODIFIED BY: Adel Moumen 2024
"""
GigaSpeech is an evolving, multi-domain English speech recognition corpus with 10,000 hours of high quality
labeled audio suitable for supervised training, and 40,000 hours of total audio suitable for semi-supervised
and unsupervised training. Around 40,000 hours of transcribed audio is first collected from audiobooks, podcasts
and YouTube, covering both read and spontaneous speaking styles, and a variety of topics, such as arts, science,
sports, etc. A new forced alignment and segmentation pipeline is proposed to create sentence segments suitable
for speech recognition training, and to filter out segments with low-quality transcription. For system training,
GigaSpeech provides five subsets of different sizes, 10h, 250h, 1000h, 2500h, and 10000h.
For our 10,000-hour XL training subset, we cap the word error rate at 4% during the filtering/validation stage,
and for all our other smaller training subsets, we cap it at 0%. The DEV and TEST evaluation sets, on the other hand,
are re-processed by professional human transcribers to ensure high transcription quality.
"""

import csv
import os

import datasets

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

_CITATION = """\
@article{DBLP:journals/corr/abs-2106-06909,
  author    = {Guoguo Chen and
               Shuzhou Chai and
               Guanbo Wang and
               Jiayu Du and
               Wei{-}Qiang Zhang and
               Chao Weng and
               Dan Su and
               Daniel Povey and
               Jan Trmal and
               Junbo Zhang and
               Mingjie Jin and
               Sanjeev Khudanpur and
               Shinji Watanabe and
               Shuaijiang Zhao and
               Wei Zou and
               Xiangang Li and
               Xuchen Yao and
               Yongqing Wang and
               Yujun Wang and
               Zhao You and
               Zhiyong Yan},
  title     = {GigaSpeech: An Evolving, Multi-domain {ASR} Corpus with 10, 000 Hours
               of Transcribed Audio},
  journal   = {CoRR},
  volume    = {abs/2106.06909},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.06909},
  eprinttype = {arXiv},
  eprint    = {2106.06909},
  timestamp = {Wed, 29 Dec 2021 14:29:26 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2106-06909.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
GigaSpeech is an evolving, multi-domain English speech recognition corpus with 10,000 hours of high quality
labeled audio suitable for supervised training, and 40,000 hours of total audio suitable for semi-supervised
and unsupervised training. Around 40,000 hours of transcribed audio is first collected from audiobooks, podcasts
and YouTube, covering both read and spontaneous speaking styles, and a variety of topics, such as arts, science,
sports, etc. A new forced alignment and segmentation pipeline is proposed to create sentence segments suitable
for speech recognition training, and to filter out segments with low-quality transcription. For system training,
GigaSpeech provides five subsets of different sizes, 10h, 250h, 1000h, 2500h, and 10000h.
For our 10,000-hour XL training subset, we cap the word error rate at 4% during the filtering/validation stage,
and for all our other smaller training subsets, we cap it at 0%. The DEV and TEST evaluation sets, on the other hand,
are re-processed by professional human transcribers to ensure high transcription quality.
"""

_HOMEPAGE = "https://github.com/SpeechColab/GigaSpeech"

_LICENSE = "Apache License 2.0"

_CATEGORIES = (
    "People  and  Blogs",
    "Business",
    "Nonprofits  and  Activism",
    "Crime",
    "History",
    "Pets  and  Animals",
    "News and Politics",
    "Travel and Events",
    "Kids and Family",
    "Leisure",
    "N/A",
    "Comedy",
    "News  and  Politics",
    "Sports",
    "Arts",
    "Science  and  Technology",
    "Autos  and  Vehicles",
    "Science and Technology",
    "People and Blogs",
    "Music",
    "Society and Culture",
    "Education",
    "Howto  and  Style",
    "Film  and  Animation",
    "Gaming",
    "Entertainment",
    "Travel  and  Events",
    "Health and Fitness",
    "audiobook",
)

_SOURCES = ("audiobook", "podcast", "youtube")

_SUBSETS = ("xs", "s", "m", "l", "xl")

_BASE_DATA_URL = (
    "https://huggingface.co/datasets/speechcolab/gigaspeech/resolve/main/data/"
)

_AUDIO_ARCHIVE_URL = (
    _BASE_DATA_URL
    + "audio/{subset}_files{is_additional}/{subset}_chunks_{archive_id:04}.tar.gz"
)

_META_URL = (
    _BASE_DATA_URL
    + "metadata/{subset}_metadata{is_additional}/{subset}_chunks_{archive_id:04}_metadata.csv"
)

_N_ARCHIVES_URL = _BASE_DATA_URL + "{subset}_n_archives{is_additional}.txt"

logger = datasets.utils.logging.get_logger(__name__)


class GigaspeechConfig(datasets.BuilderConfig):
    """BuilderConfig for Gigaspeech."""

    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        # larger subsets are supersets of smaller subsets,
        # if we want to download "m", we need to download "xs" and "s" data too.
        # so if name == "m", self.subsets_to_download will be ("xs", "s", "m")
        if name not in {"dev", "test"}:
            self.subsets_to_download = _SUBSETS[: _SUBSETS.index(name) + 1]
        else:
            self.subsets_to_download = (name,)


class Gigaspeech(datasets.GeneratorBasedBuilder):
    """
    GigaSpeech is an evolving, multi-domain English speech recognition corpus with 10,000 hours of high quality
    labeled audio suitable for supervised training, and 40,000 hours of total audio suitable for semi-supervised
    and unsupervised training (this implementation contains only labelled data for now).
    Around 40,000 hours of transcribed audio is first collected from audiobooks, podcasts
    and YouTube, covering both read and spontaneous speaking styles, and a variety of topics, such as arts, science,
    sports, etc. A new forced alignment and segmentation pipeline is proposed to create sentence segments suitable
    for speech recognition training, and to filter out segments with low-quality transcription. For system training,
    GigaSpeech provides five subsets of different sizes, 10h, 250h, 1000h, 2500h, and 10000h.
    For our 10,000-hour XL training subset, we cap the word error rate at 4% during the filtering/validation stage,
    and for all our other smaller training subsets, we cap it at 0%. The DEV and TEST evaluation sets, on the other hand,
    are re-processed by professional human transcribers to ensure high transcription quality.
    """

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        GigaspeechConfig(name=subset) for subset in _SUBSETS + ("dev", "test")
    ]

    DEFAULT_WRITER_BATCH_SIZE = 128

    def _info(self):
        features = datasets.Features(
            {
                "segment_id": datasets.Value("string"),
                "speaker": datasets.Value("string"),
                "text": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000, decode=False),
                "begin_time": datasets.Value("float32"),
                "end_time": datasets.Value("float32"),
                "audio_id": datasets.Value("string"),
                "title": datasets.Value("string"),
                "url": datasets.Value("string"),
                "source": datasets.ClassLabel(names=_SOURCES),
                "category": datasets.ClassLabel(names=_CATEGORIES),
                "original_full_path": datasets.Value(
                    "string"
                ),  # relative path to full audio in original data dirs
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _is_additional_data(self, name):
        if name in {"s", "m", "l", "xl"}:
            return "_additional"
        return ""

    @property
    def _splits_to_subsets(self):
        return {
            "train": self.config.subsets_to_download,
            "dev": ["dev"],
            "test": ["test"],
        }

    def _read_n_archives(self, n_archives_path):
        with open(n_archives_path, encoding="utf-8") as f:
            return int(f.read().strip())

    def _split_generators(self, dl_manager):
        splits_to_subsets = self._splits_to_subsets
        if self.config.name in {"dev", "test"}:
            splits = (self.config.name,)
        else:
            splits = ("train", "dev", "test")

        # 1. get number of archives (shards) in each subset
        n_archives_links = {
            split: {
                subset: _N_ARCHIVES_URL.format(
                    subset=subset,
                    is_additional=self._is_additional_data(subset),
                )
                for subset in splits_to_subsets[split]
            }
            for split in splits
        }
        logger.info("Downloading the data. It may take a while.")
        paths = dl_manager.download(n_archives_links)
        logger.info("Extracting the data. It may take a while.")
        n_archives_paths = dl_manager.extract(paths)
        n_archives = {
            # mapping from a subset to a single number - number of audio archives (shards) in a subset
            split: {
                subset: self._read_n_archives(n_archives_paths[split][subset])
                for subset in splits_to_subsets[split]
            }
            for split in splits
        }

        # 2. prepare sharded archives with audio files
        audio_archives_urls = {
            split: {
                subset: [
                    _AUDIO_ARCHIVE_URL.format(
                        subset=subset,
                        is_additional=self._is_additional_data(subset),
                        archive_id=i,
                    )
                    for i in range(n_archives[split][subset])
                ]
                for subset in splits_to_subsets[split]
            }
            for split in splits
        }
        audio_archives_paths = dl_manager.download(audio_archives_urls)
        # flatten archives paths from
        # {"train": {"xs": [path1, path2,], "s": [path3], "m": [path5, path5]}, "dev": {"dev": [path6,...]}, "test": {"test": [...]}}
        # to {"train": [path1, path2, path3, path4, path5], "dev": [path6, ...], "test": [...]}
        audio_archives_paths = _flatten_nested_dict(audio_archives_paths)
        local_audio_archives_paths = (
            dl_manager.extract(audio_archives_paths)
            if not dl_manager.is_streaming
            else None
        )

        # 3. prepare sharded metadata csv files
        meta_urls = {
            split: {
                subset: [
                    _META_URL.format(
                        subset=subset,
                        is_additional=self._is_additional_data(subset),
                        archive_id=i,
                    )
                    for i in range(n_archives[split][subset])
                ]
                for subset in splits_to_subsets[split]
            }
            for split in splits
        }
        meta_paths = dl_manager.download_and_extract(meta_urls)
        meta_paths = _flatten_nested_dict(meta_paths)

        if self.config.name not in {"dev", "test"}:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "audio_archives_iterators": [
                            dl_manager.iter_archive(archive_path)
                            for archive_path in audio_archives_paths["train"]
                        ],
                        "local_audio_archives_paths": (
                            local_audio_archives_paths["train"]
                            if local_audio_archives_paths
                            else None
                        ),
                        "meta_paths": meta_paths["train"],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "audio_archives_iterators": [
                            dl_manager.iter_archive(archive_path)
                            for archive_path in audio_archives_paths["dev"]
                        ],
                        "local_audio_archives_paths": (
                            local_audio_archives_paths["dev"]
                            if local_audio_archives_paths
                            else None
                        ),
                        "meta_paths": meta_paths["dev"],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "audio_archives_iterators": [
                            dl_manager.iter_archive(archive_path)
                            for archive_path in audio_archives_paths["test"]
                        ],
                        "local_audio_archives_paths": (
                            local_audio_archives_paths["test"]
                            if local_audio_archives_paths
                            else None
                        ),
                        "meta_paths": meta_paths["test"],
                    },
                ),
            ]

        if self.config.name == "dev":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "audio_archives_iterators": [
                            dl_manager.iter_archive(archive_path)
                            for archive_path in audio_archives_paths["dev"]
                        ],
                        "local_audio_archives_paths": (
                            local_audio_archives_paths["dev"]
                            if local_audio_archives_paths
                            else None
                        ),
                        "meta_paths": meta_paths["dev"],
                    },
                ),
            ]

        if self.config.name == "test":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "audio_archives_iterators": [
                            dl_manager.iter_archive(archive_path)
                            for archive_path in audio_archives_paths["test"]
                        ],
                        "local_audio_archives_paths": (
                            local_audio_archives_paths["test"]
                            if local_audio_archives_paths
                            else None
                        ),
                        "meta_paths": meta_paths["test"],
                    },
                ),
            ]

    def _generate_examples(
        self, audio_archives_iterators, local_audio_archives_paths, meta_paths
    ):
        assert len(audio_archives_iterators) == len(meta_paths)
        if local_audio_archives_paths:
            assert len(audio_archives_iterators) == len(
                local_audio_archives_paths
            )

        for i, (meta_path, audio_archive_iterator) in enumerate(
            zip(meta_paths, audio_archives_iterators)
        ):
            meta_dict = dict()
            with open(meta_path, encoding="utf-8") as csvfile:
                meta_csv = csv.DictReader(csvfile)
                for line in meta_csv:
                    meta_dict[line["sid"]] = line

            for audio_path_in_archive, audio_file in audio_archive_iterator:
                # `audio_path_in_archive` is like "dev_chunks_0000/YOU1000000029_S0000095.wav"
                audio_filename = os.path.split(audio_path_in_archive)[1]
                audio_id = audio_filename.split(".wav")[0]
                audio_meta = meta_dict[audio_id]
                audio_meta["segment_id"] = audio_meta.pop("sid")
                audio_meta["original_full_path"] = audio_meta.pop("path")
                audio_meta["text"] = audio_meta.pop("text_tn")
                audio_meta["audio_id"] = audio_meta.pop("aid")
                if not audio_meta["category"]:
                    audio_meta["category"] = "N/A"

                path = (
                    os.path.join(
                        local_audio_archives_paths[i], audio_path_in_archive
                    )
                    if local_audio_archives_paths
                    else audio_path_in_archive
                )

                yield audio_id, {
                    "audio": {"path": path, "bytes": audio_file.read()},
                    **{
                        feature: value
                        for feature, value in audio_meta.items()
                        if feature in self.info.features
                    },
                }


def _flatten_nested_dict(nested_dict):
    return {
        key: [
            inner_list_element
            for inner_list in value_to_lists.values()
            for inner_list_element in inner_list
        ]
        for key, value_to_lists in nested_dict.items()
    }
