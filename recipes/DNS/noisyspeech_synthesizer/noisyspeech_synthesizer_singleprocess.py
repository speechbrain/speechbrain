"""
Source: https://github.com/microsoft/DNS-Challenge
Ownership: Microsoft

* Author
    chkarada
"""

# Note: This single process audio synthesizer will attempt to use each clean
# speech sourcefile once, as it does not randomly sample from these files

import sys
import os
from pathlib import Path
import glob
import random
from random import shuffle
import time

import numpy as np
from scipy import signal
from scipy.io import wavfile

import librosa

import utils
from audiolib import (
    audiowrite,
    segmental_snr_mixer,
    activitydetector,
    is_clipped,
)

import pandas as pd
import json
from functools import partial
from typing import Dict

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
        if "noisefilenames" in params.keys():
            data_iterator = iter(params["noise_data"])
            idx = index
        # if noise files are organized into individual subdirectories, pick a directory randomly
        else:
            noisedirs = params["noisedirs"]
            # pick a noise category randomly
            idx_n_dir = np.random.randint(0, np.size(noisedirs))
            source_files = glob.glob(
                os.path.join(noisedirs[idx_n_dir], params["audioformat"])
            )
            shuffle(source_files)
            # pick a noise source file index randomly
            idx = np.random.randint(0, np.size(source_files))

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

    if tries_left == 0 and not is_clean and "noisedirs" in params.keys():
        print(
            "There are not enough non-clipped files in the "
            + noisedirs[idx_n_dir]
            + " directory to complete the audio build"
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
    file_num = params["fileindex_start"]

    # write shards
    shards_path = Path(params["shard_destination"])
    shards_path.mkdir(exist_ok=True, parents=True)

    all_keys = set()
    pattern = str(shards_path / "shard") + "-%06d.tar"
    samples_per_shard = params["samples_per_shard"]

    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
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

            # get rir files and config

            # mix clean speech and noise
            # if specified, use specified SNR value
            if not params["randomize_snr"]:
                snr = params["snr"]
            # use a randomly sampled SNR value between the specified bounds
            else:
                snr = np.random.randint(
                    params["snr_lower"], params["snr_upper"]
                )

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
                + ".wav"
            )
            cleanfilename = "clean_fileid_" + str(file_num) + ".wav"
            noisefilename = "noise_fileid_" + str(file_num) + ".wav"
            audio_signals = [noisy_snr, clean_snr, noise_snr]

            file_num += 1

            if params["sharding"]:
                # verify key is unique
                assert cleanfilename not in all_keys
                all_keys.add(cleanfilename)
                ## write shards of data
                sample = {
                    "__key__": cleanfilename,
                    "clean_file": cleanfilename,
                    "noise_file": noisefilename,
                    "noisy_file": noisyfilename,
                    "language_id": params["split_name"],
                    "clean_audio.pth": torch.tensor(clean_snr),
                    "noise_audio.pth": torch.tensor(noise_snr),
                    "noisy_audio.pth": torch.tensor(noisy_snr),
                }
                sink.write(sample)
            else:
                noisypath = os.path.join(
                    params["noisyspeech_dir"], noisyfilename
                )
                cleanpath = os.path.join(
                    params["clean_proc_dir"], cleanfilename
                )
                noisepath = os.path.join(
                    params["noise_proc_dir"], noisefilename
                )

                file_paths = [noisypath, cleanpath, noisepath]

                ## write audio files to disk
                for i in range(len(audio_signals)):
                    try:
                        audiowrite(
                            file_paths[i], audio_signals[i], params["fs"]
                        )
                    except Exception as e:
                        print(str(e))

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

    if hparams["speech_dir"] != "None":
        clean_dir = hparams["speech_dir"]
    if not os.path.exists(clean_dir):
        assert False, "Clean speech data is required"

    if hparams["noise_dir"] != "None":
        noise_dir = hparams["noise_dir"]
    if not os.path.exists:
        assert False, "Noise data is required"

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
    params["shard_destination"] = hparams["shard_destination"]
    params["sharding"] = hparams["sharding"]

    # Output Directories
    params["noisyspeech_dir"] = utils.get_dir(
        hparams, "noisy_destination", "noisy"
    )
    params["clean_proc_dir"] = utils.get_dir(
        hparams, "clean_destination", "clean"
    )
    params["noise_proc_dir"] = utils.get_dir(
        hparams, "noise_destination", "noise"
    )

    #### Shard data extraction ~~~
    # load the meta info json file

    with wds.gopen(hparams["clean_meta"], "rb") as f:
        clean_meta = json.load(f)
    with wds.gopen(hparams["noise_meta"], "rb") as f:
        noise_meta = json.load(f)

    snt_len_sample = int(hparams["sampling_rate"] * hparams["audio_length"])

    def audio_pipeline(sample_dict: Dict, random_chunk=True):
        key = sample_dict["__key__"]
        audio_tensor = sample_dict["audio.pth"]

        # determine what part of audio sample to use
        audio_tensor = audio_tensor.squeeze()

        if random_chunk:
            if len(audio_tensor) - snt_len_sample - 1 <= 0:
                start = 0
            else:
                start = random.randint(
                    0, len(audio_tensor) - snt_len_sample - 1
                )

            stop = start + snt_len_sample
        else:
            start = 0
            stop = len(audio_tensor)

        sig = audio_tensor[start:stop]

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

    ## Extraction of clean speech samples takes place here.
    if hasattr(hparams, "speech_csv") and hparams["speech_csv"] != "None":
        cleanfilenames = pd.read_csv(hparams["speech_csv"])
        cleanfilenames = cleanfilenames["filename"]
    else:
        # cleanfilenames = glob.glob(os.path.join(clean_dir, params['audioformat']))
        cleanfilenames = []
        for path in Path(clean_dir).rglob("*.wav"):
            cleanfilenames.append(str(path.resolve()))

    shuffle(cleanfilenames)
    #   add singing voice to clean speech
    if params["use_singing_data"] == 1:
        all_singing = []
        for path in Path(params["clean_singing"]).rglob("*.wav"):
            all_singing.append(str(path.resolve()))

        if params["singing_choice"] == 1:  # male speakers
            mysinging = [
                s for s in all_singing if ("male" in s and "female" not in s)
            ]

        elif params["singing_choice"] == 2:  # female speakers
            mysinging = [s for s in all_singing if "female" in s]

        elif params["singing_choice"] == 3:  # both male and female
            mysinging = all_singing
        else:  # default both male and female
            mysinging = all_singing

        shuffle(mysinging)
        if mysinging is not None:
            all_cleanfiles = cleanfilenames + mysinging
    else:
        print("NOT using singing data for training!")
        all_cleanfiles = cleanfilenames

    #   add emotion data to clean speech
    if params["use_emotion_data"] == 1:
        all_emotion = []
        for path in Path(params["clean_emotion"]).rglob("*.wav"):
            all_emotion.append(str(path.resolve()))

        shuffle(all_emotion)
        if all_emotion is not None:
            all_cleanfiles = all_cleanfiles + all_emotion
    else:
        print("NOT using emotion data for training!")

    #   add mandarin data to clean speech
    if params["use_mandarin_data"] == 1:
        all_mandarin = []
        for path in Path(params["clean_mandarin"]).rglob("*.wav"):
            all_mandarin.append(str(path.resolve()))

        shuffle(all_mandarin)
        if all_mandarin is not None:
            all_cleanfiles = all_cleanfiles + all_mandarin
    else:
        print("NOT using non-english (Mandarin) data for training!")

    params["cleanfilenames"] = all_cleanfiles
    params["num_cleanfiles"] = len(params["cleanfilenames"])

    ## Extraction of noise samples takes place here.
    # If there are .wav files in noise_dir directory, use those
    # If not, that implies that the noise files are organized into subdirectories by type,
    # so get the names of the non-excluded subdirectories
    if hasattr(hparams, "noise_csv") and hparams["noise_csv"] != "None":
        noisefilenames = pd.read_csv(hparams["noise_csv"])
        noisefilenames = noisefilenames["filename"]
    else:
        noisefilenames = glob.glob(
            os.path.join(noise_dir, params["audioformat"])
        )

    if len(noisefilenames) != 0:
        shuffle(noisefilenames)
        params["noisefilenames"] = noisefilenames
    else:
        noisedirs = glob.glob(os.path.join(noise_dir, "*"))
        if hparams["noise_types_excluded"] != "None":
            dirstoexclude = hparams["noise_types_excluded"].split(",")
            for dirs in dirstoexclude:
                noisedirs.remove(dirs)
        shuffle(noisedirs)
        params["noisedirs"] = noisedirs

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
