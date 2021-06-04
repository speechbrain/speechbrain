#!/usr/bin/env python
# coding: utf-8
# author: Matteo Esposito 2021

## Creation of Multi-Speaker Datasets

import glob
import os
import random
import shutil
import sys
from distutils.dir_util import copy_tree

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataio import read_audio, write_audio

from datetime import datetime


# From: https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
def copy_folder_contents(source, destination):
    """Copy folder contents to another folder and create the new folder if it doesn't exist.

    Args:
        source (String): Source filepath
        destination (String): Destination filepath (new folder)
    """
    if os.path.exists(destination):
        shutil.rmtree(destination)
    os.makedirs(destination)
    copy_tree(source, destination)


# From: https://stackoverflow.com/questions/8428954/move-child-folder-contents-to-parent-folder-in-python
def move_to_root_folder(root_path, cur_path):
    """Utility to move all nested files from subdirectories to parent dir.

    Args:
        root_path (String): Root path
        cur_path (String): Current path
    """

    for filename in os.listdir(cur_path):
        if os.path.isfile(os.path.join(cur_path, filename)):
            shutil.move(
                os.path.join(cur_path, filename),
                os.path.join(root_path, filename),
            )
        elif os.path.isdir(os.path.join(cur_path, filename)):
            move_to_root_folder(root_path, os.path.join(cur_path, filename))
        else:
            sys.exit("Should never reach here.")
    # remove empty folders
    if cur_path != root_path:
        os.rmdir(cur_path)


def flatten_directory(original_data_path, destination_data_path, audio_folders):
    """Utility function to remove all subdirectories in a directory and place all files under the parent dir. 

    Args:
        original_data_path (String): Filepath for directory wished to be flattened.
    """

    # Folder management
    print(
        f"+++ Copying data from {original_data_path} to {destination_data_path}"
    )
    copy_folder_contents(original_data_path, destination_data_path)
    print(
        f"+++ Data copied from {original_data_path} to {destination_data_path}"
    )

    # Flatten each of the subdirs.
    for folder in audio_folders:
        current_path = os.path.join(destination_data_path, folder)
        move_to_root_folder(current_path, current_path)
        print(
            f"+++ Directory {current_path} flattened and ready for processing."
        )


def make_data(
    original_data_path, destination_data_path, max_samples, audio_folders
):
    flatten_directory(original_data_path, destination_data_path, audio_folders)

    print(f"+++ Starting data creation...")

    # Get list of flac audio files in LibriSpeech
    audio_files = []
    for folder in audio_folders:
        temp_path = os.path.join(hparams["new_data_folder"], folder)

        # Remove problematic flac file
        if folder == "dev-clean":
            os.remove(os.path.join(temp_path, "TEMP.flac"))

        audio_files.append(glob.glob(f"{temp_path}/*.flac"))

    nb_folders = len(audio_folders)

    # Get list of unique speaker ids
    unique_speakers = [[], [], []]
    for folder_idx in range(nb_folders):
        for path in audio_files[folder_idx]:
            unique_speakers[folder_idx].append(
                path.split("/")[-1].split("-")[0]
            )
        unique_speakers[folder_idx] = list(set(unique_speakers[folder_idx]))
        print(
            f"*** {audio_folders[folder_idx]} unique speakers = {len(unique_speakers[folder_idx])}"
        )

    # Crop all samples to max(original_duration, 5)
    for i in range(nb_folders):
        print(
            f"+++ Shortening and adding padding to all {audio_folders[i].split('/')[-1]} samples to create 5s total duration"
        )
        for recording in audio_files[i]:
            xs_speech = read_audio(recording)
            temp = xs_speech.unsqueeze(0)[0][
                :max_samples
            ]  # [batch, time, channels]
            write_audio(recording[:-5] + "-5s.flac", temp, 16000)
            os.remove(recording)  # Delete old flac files.

    # Add padding to make all clips exactly 5s
    audio_files = []
    for folder in audio_folders:
        audio_files.append(
            glob.glob(f"{hparams['new_data_folder']}{folder}/*.flac")
        )

    for i in range(nb_folders):
        print(
            f"+++ Adding padding to all {audio_folders[i].split('/')[-1]} samples to create 5s total duration"
        )
        for recording in audio_files[i]:
            # Read file
            xs_speech = read_audio(recording)
            xs_speech = xs_speech.unsqueeze(0)

            # Add padding
            padding_tensor = torch.zeros(size=(1, max_samples))
            padding_tensor[0, : xs_speech.size()[1]] = xs_speech

            # Write file
            write_audio(
                recording[:-5] + "-w-padding.flac",
                padding_tensor.reshape(-1),
                16000,
            )
            os.remove(recording)  # Delete old flac files.

    ## Combine 5s audio clips
    # Create speaker-1 to speaker-5 overlap folders to store new intersecting audio
    for folder in audio_folders:
        for i in range(1, 6):
            new_folder_path = os.path.join(
                hparams["new_data_folder"], folder + "/", f"{i}-speaker"
            )
            if os.path.exists(new_folder_path):
                shutil.rmtree(new_folder_path)
            os.makedirs(new_folder_path)

    # Get list of flac audio files in LibriSpeech post-padding
    audio_files = []
    for folder in audio_folders:
        temp_path = os.path.join(hparams["new_data_folder"], folder)
        audio_files.append(glob.glob(f"{temp_path}/*.flac"))

    # Get random number of unique speaker audio files and combine them (3 times per point, therefore augmenting data 3x)
    for folder_id in range(nb_folders):
        nb_speakers_log = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        for iteration in range(0, 3):
            print(
                f"Creating {audio_folders[folder_id]} random mixtures ({iteration + 1}/3)"
            )
            for path in audio_files[folder_id]:
                mix_files = []
                nb_speakers = random.randint(1, 5)
                nb_speakers_log[
                    str(nb_speakers)
                ] += 1  # Keeping a count of the number of 1-5 speaker signals created.
                speaker_indices = random.sample(
                    unique_speakers[folder_id], nb_speakers
                )

                # Get n unique files.
                for speaker_idx in speaker_indices:
                    samples_w_given_speaker = [
                        audio
                        for audio in audio_files[folder_id]
                        if audio.split("/")[-1].split("-")[0] == speaker_idx
                    ]
                    mix_files.append(random.choice(samples_w_given_speaker))

                # Read and combine n unique files.
                out = torch.zeros(size=(1, max_samples))
                for flac in mix_files:
                    # random_amplitude = random.uniform(0.5, 1)
                    signal = read_audio(flac)
                    out += signal / torch.norm(signal)

                # Normalize mixture
                # amp_final = random.uniform(0.5, 1)
                # mix_final = amp_final*out/abs(out)
                mix_final = out / torch.norm(out)

                # Output in appropriate folder
                mixture_name = (
                    "-".join(speaker_indices)
                    + "-id-"
                    + str(random.randint(0, 1000000))
                    + ".flac"
                )
                write_audio(
                    f"{hparams['new_data_folder']}{audio_folders[folder_id]}/{nb_speakers}-speaker/{mixture_name}",
                    out.reshape(-1),
                    16000,
                )

        print("Speaker mixture counts: ", nb_speakers_log)


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("\nCurrent Time =", current_time, "\n")

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Set some constants for data creation
    MAX_SAMPLES = hparams["max_duration"] * hparams["sampling_rate"]
    AUDIO_FOLDERS = hparams["subfolders"]

    # Create data
    make_data(
        original_data_path=hparams["original_data_folder"],
        destination_data_path=hparams["new_data_folder"],
        max_samples=MAX_SAMPLES,
        audio_folders=AUDIO_FOLDERS,
    )
