################################################################################
#
# Converts the uncompressed DNS folder
# {french,german,...}_speech/../<*.wav>
# structure of DNS into a WebDataset format
#
# Author(s): Tanel Alumäe, Nik Vaessen, Sangeet Sagar (2023)
################################################################################

import argparse
import json
import os
import pathlib
import random
from collections import defaultdict

import librosa
import torch
import torchaudio
import webdataset as wds
from tqdm import tqdm

################################################################################
# methods for writing the shards

ID_SEPARATOR = "&"


def load_audio(audio_file_path: pathlib.Path) -> torch.Tensor:
    t, sr = torchaudio.load(audio_file_path)

    return t


def write_shards(
    dns_folder_path: pathlib.Path,
    shards_path: pathlib.Path,
    seed: int,
    samples_per_shard: int,
    min_dur: float,
):
    """
    Arguments
    ---------
    dns_folder_path: pathlib.Path
        folder where extracted DNS data is located
    shards_path: pathlib.Path
        folder to write shards of data to
    seed: int
        random seed used to initially shuffle data into shards
    samples_per_shard: int
        number of data samples to store in each shards.
    min_dur: float
        Smallest possible duration.
    """
    # make sure output folder exist
    shards_path.mkdir(parents=True, exist_ok=True)

    # find all audio files
    audio_files = sorted([f for f in dns_folder_path.rglob("*.wav")])

    # create tuples (unique_sample_id, language_id, path_to_audio_file, duration)
    data_tuples = []

    # track statistics on data
    all_language_ids = set()
    sample_keys_per_language = defaultdict(list)

    if "clean" in dns_folder_path.as_posix():
        delim = "clean_fullband/"
    elif "noise" in dns_folder_path.as_posix():
        delim = "noise_fullband/"
        lang = "noise"
    elif "dev_testset" in dns_folder_path.as_posix():
        delim = "dev_testset/"
        lang = "baseline_noisytestset"
    else:
        delim = os.path.basename(dns_folder_path.as_posix())
        lang = delim

    for f in tqdm(audio_files):
        # path should be
        # {french,german,...}_speech/../<*.wav>
        sub_path = f.as_posix().split(delim)[1]

        loc = f.as_posix()
        key = os.path.splitext(os.path.basename(sub_path))[0]
        if "clean_fullband" in dns_folder_path.as_posix():
            lang = key.split("_speech")[0]

        dur = librosa.get_duration(path=loc)

        # Period is not allowed in a WebDataset key name
        key = key.replace(".", "_")
        if dur > min_dur:
            # store statistics
            all_language_ids.add(lang)
            sample_keys_per_language[lang].append(key)
            t = (key, lang, loc, dur)
            data_tuples.append(t)

    all_language_ids = sorted(all_language_ids)

    # write a meta.json file which contains statistics on the data
    # which will be written to shards
    meta_dict = {
        "language_ids": list(all_language_ids),
        "sample_keys_per_language": sample_keys_per_language,
        "num_data_samples": len(data_tuples),
    }

    with (shards_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=4)

    # shuffle the tuples so that each shard has a large variety in languages
    random.seed(seed)
    random.shuffle(data_tuples)

    # write shards
    all_keys = set()
    shards_path.mkdir(exist_ok=True, parents=True)
    pattern = str(shards_path / "shard") + "-%06d.tar"

    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
        for key, language_id, f, duration in data_tuples:
            # load the audio tensor
            tensor = load_audio(f)

            # verify key is unique
            assert key not in all_keys
            all_keys.add(key)

            # create sample to write
            sample = {
                "__key__": key,
                "audio.pth": tensor,
                "language_id": language_id,
            }

            # write sample to sink
            sink.write(sample)


################################################################################
# define CLI

parser = argparse.ArgumentParser(
    description="Convert DNS-4 to WebDataset shards"
)

parser.add_argument(
    "dns_decompressed_path",
    type=pathlib.Path,
    help="directory containing the (decompressed) DNS dataset",
)
parser.add_argument(
    "shards_path", type=pathlib.Path, help="directory to write shards to"
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="random seed used for shuffling data before writing to shard",
)
parser.add_argument(
    "--samples_per_shard",
    type=int,
    default=5000,
    help="the maximum amount of samples placed in each shard. The last shard "
    "will most likely contain fewer samples.",
)
parser.add_argument(
    "--min-duration",
    type=float,
    default=3.0,
    help="Minimum duration of the audio",
)


################################################################################
# execute script

if __name__ == "__main__":
    args = parser.parse_args()

    write_shards(
        args.dns_decompressed_path,
        args.shards_path,
        args.seed,
        args.samples_per_shard,
        args.min_duration,
    )
