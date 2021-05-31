#!/usr/bin/env python
# coding: utf-8

## Creation of Multi-Speaker Datasets
import os
import glob
import torch
import random
import shutil
from speechbrain.dataio.dataio import read_audio, write_audio

os.chdir("/home/matteo/projects/speechbrain/")

DATADIR = "/home/matteo/projects/data/5s-LibriSpeech/"
AUDIO_FOLDERS = ["dev-clean", "test-clean", "train-clean-100"]
SAMPLING_RATE = 16000  # hz
MAX_DURATION = 5  # seconds
MAX_SAMPLES = MAX_DURATION * SAMPLING_RATE

# Get list of flac audio files in LibriSpeech
audio_files = []
for folder in AUDIO_FOLDERS:
    audio_files.append(
        glob.glob(f"/home/matteo/projects/data/5s-LibriSpeech/{folder}/*.flac")
    )

# Get list of unique speaker ids
unique_speakers = [[], [], []]
for folder_idx in range(3):
    for path in audio_files[folder_idx]:
        unique_speakers[folder_idx].append(path.split("/")[-1].split("-")[0])
    unique_speakers[folder_idx] = list(set(unique_speakers[folder_idx]))
    print(
        f"{AUDIO_FOLDERS[folder_idx]} unique speakers = {len(unique_speakers[folder_idx])}"
    )

# Crop all samples to max(original_duration, 5)
for i in range(0, 3):
    print(
        f"Shortening and adding padding to all {AUDIO_FOLDERS[i].split('/')[-1]} samples to create 5s total duration"
    )
    for recording in audio_files[i]:
        xs_speech = read_audio(recording)
        temp = xs_speech.unsqueeze(0)[0][
            :MAX_SAMPLES
        ]  # [batch, time, channels]
        write_audio(recording[:-5] + "-5s.flac", temp, 16000)


## Add padding to make all clips exactly 5s
audio_files = []
for folder in AUDIO_FOLDERS:
    audio_files.append(
        glob.glob(f"/home/matteo/projects/data/5s-LibriSpeech/{folder}/*.flac")
    )

for i in range(0, 3):
    print(
        f"Adding padding to all {AUDIO_FOLDERS[i].split('/')[-1]} samples to create 5s total duration"
    )
    for recording in audio_files[i]:

        # Read file
        xs_speech = read_audio(recording)
        xs_speech = xs_speech.unsqueeze(0)

        # Add padding
        padding_tensor = torch.zeros(size=(1, MAX_SAMPLES))
        padding_tensor[0, : xs_speech.size()[1]] = xs_speech

        # Write file
        write_audio(
            recording[:-5] + "-w-padding.flac",
            padding_tensor.reshape(-1),
            16000,
        )


## Combine 5s audio clips
# Make 1 to 5 overlap folders to store intersecting audio
for folder in AUDIO_FOLDERS:
    for i in range(1, 6):
        new_folder_path = os.path.join(DATADIR, folder + "/", f"{i}-speaker")
        if os.path.exists(new_folder_path):
            shutil.rmtree(new_folder_path)
        os.makedirs(new_folder_path)

# Get random number of unique speaker audio files and combine them (3 times per point, therefore augmenting data 3x)
for folder_id in range(0, 3):
    nb_speakers_log = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    for iteration in range(0, 3):
        print(
            f"Creating {AUDIO_FOLDERS[folder_id]} random mixtures ({iteration+1}/3)"
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
            out = torch.zeros(size=(1, MAX_SAMPLES))
            for flac in mix_files:
                #                 random_amplitude = random.uniform(0.5, 1)
                signal = read_audio(flac)
                out += signal / torch.norm(signal)

            # Normalize mixture
            #             amp_final = random.uniform(0.5, 1)
            #             mix_final = amp_final*out/abs(out)
            mix_final = out / torch.norm(out)

            # Output in appropriate folder
            mixture_name = (
                "-".join(speaker_indices)
                + "-id-"
                + str(random.randint(0, 1000000))
                + ".flac"
            )
            write_audio(
                f"/home/matteo/projects/data/5s-LibriSpeech/{AUDIO_FOLDERS[folder_id]}/{nb_speakers}-speaker/{mixture_name}",
                out.reshape(-1),
                16000,
            )

    print("Speaker mixture counts: ", nb_speakers_log)
