"""
Source: https://github.com/microsoft/DNS-Challenge
Ownership: Microsoft

This script will attempt to use each clean and noise
webdataset shards to synthesize clean-noisy pairs of
audio. The output is again stored in webdataset shards.

* Author
    chkarada

* Further modified
    Sangeet Sagar (2023)
"""

# Note: This single process audio synthesizer will attempt to use each clean
# speech sourcefile once (from the webdataset shards), as it does not
# randomly sample from these files

import sys
import os
from pathlib import Path
import random
import time

import numpy as np
from scipy import signal
from scipy.io import wavfile

import librosa

import utils
from audiolib import (
    segmental_snr_mixer,
    activitydetector,
    is_clipped,
)

import pandas as pd
import json
from functools import partial
from typing import Dict
from collections import defaultdict


import speechbrain as sb
import webdataset as wds
from hyperpyyaml import load_hyperpyyaml
import torch

np.random.seed(5)
random.seed(5)

MAXTRIES = 50
MAXFILELEN = 100

start = time.time()


def add_pyreverb(clean_speech, rir):
    """
    Add reverb to cean signal
    """
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")

    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0 : clean_speech.shape[0]]

    return reverb_speech


def build_audio(is_clean, params, index, audio_samples_length=-1):
    """Construct an audio signal from source files"""

    fs_output = params["fs"]
    silence_length = params["silence_length"]
    if audio_samples_length == -1:
        audio_samples_length = int(params["audio_length"] * params["fs"])

    output_audio = np.zeros(0)
    remaining_length = audio_samples_length
    files_used = []
    clipped_files = []

    if is_clean:
        data_iterator = iter(params["clean_data"])
        idx = index
    else:
        data_iterator = iter(params["noise_data"])
        idx = index

    # initialize silence
    silence = np.zeros(int(fs_output * silence_length))

    # iterate through multiple clips until we have a long enough signal
    tries_left = MAXTRIES
    while remaining_length > 0 and tries_left > 0:
        # read next audio file and resample if necessary
        fs_input = params["fs_input"]
        batch = next(data_iterator)
        input_audio = batch["sig"].numpy()

        if input_audio is None:
            sys.stderr.write(
                "\nWARNING: Cannot read file: %s\n" % batch["__key__"]
            )
            continue
        if fs_input != fs_output:
            input_audio = librosa.resample(
                input_audio, orig_sr=fs_input, target_sr=fs_output
            )

        # if current file is longer than remaining desired length, and this is
        # noise generation or this is training set, subsample it randomly
        if len(input_audio) > remaining_length and (
            not is_clean or not params["is_test_set"]
        ):
            idx_seg = np.random.randint(0, len(input_audio) - remaining_length)
            input_audio = input_audio[idx_seg : idx_seg + remaining_length]

        # check for clipping, and if found move onto next file
        if is_clipped(input_audio):
            clipped_files.append(batch["__key__"])
            tries_left -= 1
            continue

        # concatenate current input audio to output audio stream
        files_used.append(batch["__key__"])
        output_audio = np.append(output_audio, input_audio)
        remaining_length -= len(input_audio)

        # add some silence if we have not reached desired audio length
        if remaining_length > 0:
            silence_len = min(remaining_length, len(silence))
            output_audio = np.append(output_audio, silence[:silence_len])
            remaining_length -= silence_len

    if tries_left == 0 and not is_clean and "noise_data" in params.keys():
        print(
            "There are not enough non-clipped files in the "
            + "given noise directory to complete the audio build"
        )
        return [], [], clipped_files, idx

    return output_audio, files_used, clipped_files, idx


def gen_audio(is_clean, params, index, audio_samples_length=-1):
    """Calls build_audio() to get an audio signal, and verify that it meets the
    activity threshold"""

    clipped_files = []
    low_activity_files = []
    if audio_samples_length == -1:
        audio_samples_length = int(params["audio_length"] * params["fs"])
    if is_clean:
        activity_threshold = params["clean_activity_threshold"]
    else:
        activity_threshold = params["noise_activity_threshold"]

    while True:
        audio, source_files, new_clipped_files, index = build_audio(
            is_clean, params, index, audio_samples_length
        )

        clipped_files += new_clipped_files
        if len(audio) < audio_samples_length:
            continue

        if activity_threshold == 0.0:
            break

        percactive = activitydetector(audio=audio)
        if percactive > activity_threshold:
            break
        else:
            low_activity_files += source_files

    return audio, source_files, clipped_files, low_activity_files, index


def main_gen(params):
    """Calls gen_audio() to generate the audio signals, verifies that they meet
    the requirements, and writes the files to storage"""

    clean_source_files = []
    clean_clipped_files = []
    clean_low_activity_files = []
    noise_source_files = []
    noise_clipped_files = []
    noise_low_activity_files = []

    clean_index = 0
    noise_index = 0

    # write shards
    train_shards_path = Path(params["train_shard_destination"])
    train_shards_path.mkdir(exist_ok=True, parents=True)
    valid_shards_path = Path(params["valid_shard_destination"])
    valid_shards_path.mkdir(exist_ok=True, parents=True)

    all_keys = set()
    train_pattern = str(train_shards_path / "shard") + "-%06d.tar"
    valid_pattern = str(valid_shards_path / "shard") + "-%06d.tar"
    samples_per_shard = params["samples_per_shard"]

    # track statistics on data
    train_sample_keys = defaultdict(list)
    valid_sample_keys = defaultdict(list)

    # Define the percentage of data to be used for validation
    validation_percentage = 0.05

    # Calculate the number of samples for training and validation
    total_samples = params["fileindex_end"] - params["fileindex_start"] + 1
    num_validation_samples = int(total_samples * validation_percentage)

    # Define separate ShardWriters for training and validation data
    train_writer = wds.ShardWriter(train_pattern, maxcount=samples_per_shard)
    valid_writer = wds.ShardWriter(valid_pattern, maxcount=samples_per_shard)

    # Initialize counters and data lists for statistics
    file_num = params["fileindex_start"]
    train_data_tuples = []
    valid_data_tuples = []

    while file_num <= params["fileindex_end"]:
        print(
            "\rFiles synthesized {:4d}/{:4d}".format(
                file_num, params["fileindex_end"]
            ),
            end="",
        )
        # CLEAN SPEECH GENERATION
        clean, clean_sf, clean_cf, clean_laf, clean_index = gen_audio(
            True, params, clean_index
        )

        # add reverb with selected RIR
        rir_index = random.randint(0, len(params["myrir"]) - 1)

        my_rir = os.path.normpath(os.path.join(params["myrir"][rir_index]))
        (fs_rir, samples_rir) = wavfile.read(my_rir)

        my_channel = int(params["mychannel"][rir_index])

        if samples_rir.ndim == 1:
            samples_rir_ch = np.array(samples_rir)

        elif my_channel > 1:
            samples_rir_ch = samples_rir[:, my_channel - 1]
        else:
            samples_rir_ch = samples_rir[:, my_channel - 1]
            # print(samples_rir.shape)
            # print(my_channel)

        # REVERB MIXED TO THE CLEAN SPEECH
        clean = add_pyreverb(clean, samples_rir_ch)

        # generate noise
        noise, noise_sf, noise_cf, noise_laf, noise_index = gen_audio(
            False, params, noise_index, len(clean)
        )

        clean_clipped_files += clean_cf
        clean_low_activity_files += clean_laf
        noise_clipped_files += noise_cf
        noise_low_activity_files += noise_laf

        # mix clean speech and noise
        # if specified, use specified SNR value
        if not params["randomize_snr"]:
            snr = params["snr"]
        # use a randomly sampled SNR value between the specified bounds
        else:
            snr = np.random.randint(params["snr_lower"], params["snr_upper"])

        # NOISE ADDED TO THE REVERBED SPEECH
        clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(
            params=params, clean=clean, noise=noise, snr=snr
        )
        # Uncomment the below lines if you need segmental SNR and comment the above lines using snr_mixer
        # clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(params=params,
        #                                                         clean=clean,
        #                                                          noise=noise,
        #                                                         snr=snr)
        # unexpected clipping
        if (
            is_clipped(clean_snr)
            or is_clipped(noise_snr)
            or is_clipped(noisy_snr)
        ):
            print(
                "\nWarning: File #"
                + str(file_num)
                + " has unexpected clipping, "
                + "returning without writing audio to disk"
            )
            continue

        clean_source_files += clean_sf
        noise_source_files += noise_sf

        # write resultant audio streams to files
        hyphen = "-"
        clean_source_filenamesonly = [
            i[:-4].split(os.path.sep)[-1] for i in clean_sf
        ]
        clean_files_joined = hyphen.join(clean_source_filenamesonly)[
            :MAXFILELEN
        ]
        noise_source_filenamesonly = [
            i[:-4].split(os.path.sep)[-1] for i in noise_sf
        ]
        noise_files_joined = hyphen.join(noise_source_filenamesonly)[
            :MAXFILELEN
        ]

        noisyfilename = (
            clean_files_joined
            + "_"
            + noise_files_joined
            + "_snr"
            + str(snr)
            + "_tl"
            + str(target_level)
            + "_fileid_"
            + str(file_num)
        )

        # Period is not allowed in a WebDataset key name
        cleanfilename = "clean_fileid_" + str(file_num)
        cleanfilename = cleanfilename.replace(".", "_")
        noisefilename = "noise_fileid_" + str(file_num)
        noisefilename = noisefilename.replace(".", "_")

        file_num += 1

        # store statistics
        key = noisyfilename
        key = key.replace(".", "_")
        lang = params["split_name"].split("_")[0]
        t = (key, lang)

        # verify key is unique
        assert cleanfilename not in all_keys
        all_keys.add(cleanfilename)

        # Split the data between training and validation based on the file number
        if file_num % total_samples <= num_validation_samples:
            # Write to validation set
            valid_sample_keys[lang].append(key)
            valid_data_tuples.append(t)
            sample = {
                "__key__": key,
                "noisy_file": key,
                "clean_file": cleanfilename,
                "noise_file": noisefilename,
                "clean_audio.pth": torch.tensor(clean_snr).to(torch.float32),
                "noise_audio.pth": torch.tensor(noise_snr).to(torch.float32),
                "noisy_audio.pth": torch.tensor(noisy_snr).to(torch.float32),
            }
            valid_writer.write(sample)
        else:
            # Write to training set
            train_sample_keys[lang].append(key)
            train_data_tuples.append(t)
            sample = {
                "__key__": key,
                "noisy_file": key,
                "clean_file": cleanfilename,
                "noise_file": noisefilename,
                "clean_audio.pth": torch.tensor(clean_snr).to(torch.float32),
                "noise_audio.pth": torch.tensor(noise_snr).to(torch.float32),
                "noisy_audio.pth": torch.tensor(noisy_snr).to(torch.float32),
            }
            train_writer.write(sample)

    train_writer.close()
    valid_writer.close()

    # Write meta.json files for both training and validation
    train_meta_dict = {
        "language_id": lang,
        "sample_keys_per_language": train_sample_keys,
        "num_data_samples": len(train_data_tuples),
    }
    valid_meta_dict = {
        "language_id": lang,
        "sample_keys_per_language": valid_sample_keys,
        "num_data_samples": len(valid_data_tuples),
    }

    with (train_shards_path / "meta.json").open("w") as f:
        json.dump(train_meta_dict, f, indent=4)

    with (valid_shards_path / "meta.json").open("w") as f:
        json.dump(valid_meta_dict, f, indent=4)

    return (
        clean_source_files,
        clean_clipped_files,
        clean_low_activity_files,
        noise_source_files,
        noise_clipped_files,
        noise_low_activity_files,
    )


def main_body():  # noqa
    """Main body of this file"""

    params = dict()

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Data Directories and Settings
    params["split_name"] = hparams["split_name"]

    # Audio Settings
    params["fs"] = int(hparams["sampling_rate"])
    params["fs_input"] = int(
        hparams["input_sampling_rate"]
    )  # Sampling rate of input data
    params["audioformat"] = hparams["audioformat"]
    params["audio_length"] = float(hparams["audio_length"])
    params["silence_length"] = float(hparams["silence_length"])
    params["total_hours"] = float(hparams["total_hours"])

    # Clean Data Categories
    params["use_singing_data"] = int(hparams["use_singing_data"])
    if hasattr(hparams, "clean_singing"):
        params["clean_singing"] = str(hparams["clean_singing"])
    params["singing_choice"] = int(hparams["singing_choice"])

    params["use_emotion_data"] = int(hparams["use_emotion_data"])
    if hasattr(hparams, "clean_emotion"):
        params["clean_emotion"] = str(hparams["clean_emotion"])

    params["use_mandarin_data"] = int(hparams["use_mandarin_data"])
    if hasattr(hparams, "clean_mandarin"):
        params["clean_mandarin"] = str(hparams["clean_mandarin"])

    # Room Impulse Response (RIR) Settings
    params["rir_choice"] = int(hparams["rir_choice"])
    params["lower_t60"] = float(hparams["lower_t60"])
    params["upper_t60"] = float(hparams["upper_t60"])
    params["rir_table_csv"] = str(hparams["rir_table_csv"])

    # File Indexing
    if (
        hparams["fileindex_start"] != "None"
        and hparams["fileindex_end"] != "None"
    ):
        params["num_files"] = int(hparams["fileindex_end"]) - int(
            params["fileindex_start"]
        )
        params["fileindex_start"] = int(hparams["fileindex_start"])
        params["fileindex_end"] = int(hparams["fileindex_end"])
    else:
        params["num_files"] = int(
            (params["total_hours"] * 60 * 60) / params["audio_length"]
        )
        params["fileindex_start"] = 0
        params["fileindex_end"] = params["num_files"]

    print("Number of files to be synthesized:", params["num_files"])

    # Data Generation and Synthesis Settings
    params["is_test_set"] = utils.str2bool(str(hparams["is_test_set"]))
    params["clean_activity_threshold"] = float(
        hparams["clean_activity_threshold"]
    )
    params["noise_activity_threshold"] = float(
        hparams["noise_activity_threshold"]
    )
    params["snr_lower"] = int(hparams["snr_lower"])
    params["snr_upper"] = int(hparams["snr_upper"])
    params["randomize_snr"] = utils.str2bool(str(hparams["randomize_snr"]))
    params["target_level_lower"] = int(hparams["target_level_lower"])
    params["target_level_upper"] = int(hparams["target_level_upper"])

    if hasattr(hparams, "snr"):
        params["snr"] = int(hparams["snr"])
    else:
        params["snr"] = int((params["snr_lower"] + params["snr_upper"]) / 2)

    # Synthesized Data Destination
    params["samples_per_shard"] = hparams["samples_per_shard"]
    params["train_shard_destination"] = hparams["train_shard_destination"]
    params["valid_shard_destination"] = hparams["valid_shard_destination"]

    #### Shard data extraction ~~~
    # load the meta info json file

    with wds.gopen(hparams["clean_meta"], "rb") as f:
        clean_meta = json.load(f)
    with wds.gopen(hparams["noise_meta"], "rb") as f:
        noise_meta = json.load(f)

    def audio_pipeline(sample_dict: Dict, random_chunk=True):
        key = sample_dict["__key__"]
        audio_tensor = sample_dict["audio.pth"]

        sig = audio_tensor.squeeze()

        return {
            "sig": sig,
            "id": key,
        }

    clean_data = (
        wds.WebDataset(
            hparams["clean_fullband_shards"],
            cache_dir=hparams["shard_cache_dir"],
        )
        .repeat()
        .shuffle(1000)
        .decode("pil")
        .map(partial(audio_pipeline, random_chunk=True))
    )
    print(f"Clean data consist of {clean_meta['num_data_samples']} samples")

    noise_data = (
        wds.WebDataset(
            hparams["noise_fullband_shards"],
            cache_dir=hparams["shard_cache_dir"],
        )
        .repeat()
        .shuffle(1000)
        .decode("pil")
        .map(partial(audio_pipeline, random_chunk=True))
    )
    print(f"Noise data consist of {noise_meta['num_data_samples']} samples")

    params["clean_data"] = clean_data
    params["noise_data"] = noise_data

    # add singing voice to clean speech
    if params["use_singing_data"] == 1:
        raise NotImplementedError("Add sining voice to clean speech")
    else:
        print("NOT using singing data for training!")

    # add emotion data to clean speech
    if params["use_emotion_data"] == 1:
        raise NotImplementedError("Add emotional data to clean speech")
    else:
        print("NOT using emotion data for training!")

    # add mandarin data to clean speech
    if params["use_mandarin_data"] == 1:
        raise NotImplementedError("Add Mandarin data to clean speech")
    else:
        print("NOT using non-english (Mandarin) data for training!")

    # rir
    temp = pd.read_csv(
        params["rir_table_csv"],
        skiprows=[1],
        sep=",",
        header=None,
        names=["wavfile", "channel", "T60_WB", "C50_WB", "isRealRIR"],
    )
    temp.keys()
    # temp.wavfile

    rir_wav = temp["wavfile"][1:]  # 115413
    rir_channel = temp["channel"][1:]
    rir_t60 = temp["T60_WB"][1:]
    rir_isreal = temp["isRealRIR"][1:]

    rir_wav2 = [w.replace("\\", "/") for w in rir_wav]
    rir_channel2 = [w for w in rir_channel]
    rir_t60_2 = [w for w in rir_t60]
    rir_isreal2 = [w for w in rir_isreal]

    myrir = []
    mychannel = []
    myt60 = []

    lower_t60 = params["lower_t60"]
    upper_t60 = params["upper_t60"]

    if params["rir_choice"] == 1:  # real 3076 IRs
        real_indices = [i for i, x in enumerate(rir_isreal2) if x == "1"]

        chosen_i = []
        for i in real_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (
                float(rir_t60_2[i]) <= upper_t60
            ):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    elif params["rir_choice"] == 2:  # synthetic 112337 IRs
        synthetic_indices = [i for i, x in enumerate(rir_isreal2) if x == "0"]

        chosen_i = []
        for i in synthetic_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (
                float(rir_t60_2[i]) <= upper_t60
            ):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    elif params["rir_choice"] == 3:  # both real and synthetic
        all_indices = [i for i, x in enumerate(rir_isreal2)]

        chosen_i = []
        for i in all_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (
                float(rir_t60_2[i]) <= upper_t60
            ):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    else:  # default both real and synthetic
        all_indices = [i for i, x in enumerate(rir_isreal2)]

        chosen_i = []
        for i in all_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (
                float(rir_t60_2[i]) <= upper_t60
            ):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    params["myrir"] = myrir
    params["mychannel"] = mychannel
    params["myt60"] = myt60

    # Call main_gen() to generate audio
    (
        clean_source_files,
        clean_clipped_files,
        clean_low_activity_files,
        noise_source_files,
        noise_clipped_files,
        noise_low_activity_files,
    ) = main_gen(params)

    # Create log directory if needed, and write log files of clipped and low activity files
    log_dir = utils.get_dir(hparams, "log_dir", "Logs")

    utils.write_log_file(
        log_dir, "source_files.csv", clean_source_files + noise_source_files
    )
    utils.write_log_file(
        log_dir, "clipped_files.csv", clean_clipped_files + noise_clipped_files
    )
    utils.write_log_file(
        log_dir,
        "low_activity_files.csv",
        clean_low_activity_files + noise_low_activity_files,
    )

    # Compute and print stats about percentange of clipped and low activity files
    total_clean = (
        len(clean_source_files)
        + len(clean_clipped_files)
        + len(clean_low_activity_files)
    )
    total_noise = (
        len(noise_source_files)
        + len(noise_clipped_files)
        + len(noise_low_activity_files)
    )
    pct_clean_clipped = round(len(clean_clipped_files) / total_clean * 100, 1)
    pct_noise_clipped = round(len(noise_clipped_files) / total_noise * 100, 1)
    pct_clean_low_activity = round(
        len(clean_low_activity_files) / total_clean * 100, 1
    )
    pct_noise_low_activity = round(
        len(noise_low_activity_files) / total_noise * 100, 1
    )

    print(
        "\nOf the "
        + str(total_clean)
        + " clean speech files analyzed, "
        + str(pct_clean_clipped)
        + "% had clipping, and "
        + str(pct_clean_low_activity)
        + "% had low activity "
        + "(below "
        + str(params["clean_activity_threshold"] * 100)
        + "% active percentage)"
    )
    print(
        "Of the "
        + str(total_noise)
        + " noise files analyzed, "
        + str(pct_noise_clipped)
        + "% had clipping, and "
        + str(pct_noise_low_activity)
        + "% had low activity "
        + "(below "
        + str(params["noise_activity_threshold"] * 100)
        + "% active percentage)"
    )


if __name__ == "__main__":
    main_body()
