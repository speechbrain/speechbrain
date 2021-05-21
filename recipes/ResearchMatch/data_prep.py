"""
Data preparation.

Authors
Jianyuan Zhong 2021
"""

import os
import shutil
import subprocess
import logging
import textgrids
import json
import torchaudio
import random
from glob import glob

from speechbrain.dataio.dataio import read_audio, write_audio

logger = logging.getLogger(__name__)

CSV_FILE = "Syllabus.csv"
CMU_PHONES = {
    "N": 41,
    "OW": 42,
    "S": 43,
    "IH": 3,
    "G": 4,
    "AH": 5,
    "L": 6,
    "EH": 7,
    "M": 8,
    "D": 9,
    "B": 10,
    "Y": 11,
    "UW": 12,
    "AE": 13,
    "T": 14,
    "AY": 15,
    "K": 16,
    "DH": 17,
    "EY": 18,
    "IY": 19,
    "W": 20,
    "R": 21,
    "AA": 22,
    "F": 23,
    "Z": 24,
    "AO": 25,
    "NG": 26,
    "V": 27,
    "P": 28,
    "HH": 29,
    "UH": 30,
    "TH": 31,
    "AW": 32,
    "CH": 33,
    "JH": 34,
    "ER": 35,
    "OY": 36,
    "SH": 37,
    "ZH": 38,
    "TS": 39,
    "IX": 40,
    "<blank>": 0,
    "<bos>": 1,
    "<eos>": 2,
}


def prepare_syllabus(
    data_folder,
    save_folder,
    use_fields=["transcribed word"],
    train_json_file="SRT_train.json",
    dev_json_file="SRT_dev.json",
    test_json_file="SRT_test.json",
    skip_prep=False,
):
    text_grids = glob(
        os.path.join(data_folder, "**/*.TextGrid"), recursive=True
    )

    wav_files = glob(os.path.join(data_folder, "**/*.wav"), recursive=True)
    mp3_files = glob(os.path.join(data_folder, "**/*.mp3"), recursive=True)
    audio_files = wav_files + mp3_files

    if skip_prep:
        return

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    text_grids_ids = []
    for i, f in enumerate(text_grids):
        text_grids_id = f.split("/")[-1]
        # dst = os.path.join(save_folder, text_grids_id)

        # if not os.path.exists(dst):
        #     shutil.copy(f, dst)

        text_grids[i] = f
        text_grids_ids.append(text_grids_id.split(".TextGrid")[0])

    audio_ids = []
    for i, f in enumerate(audio_files):
        audio_id = f.split("/")[-1].replace(";", "_")
        dst = os.path.join(save_folder, audio_id)
        print(dst)
        if not os.path.exists(dst):
            shutil.copy(f, dst)

        audio_files[i] = dst
        audio_ids.append(audio_id.split(".")[0])

    # check whether a text_grid has its corresponding audio file
    # if not remove it from clips dictionary
    text_grids_filtered = []
    for i, id in enumerate(text_grids_ids):
        if id not in audio_ids:
            msg = "{} does not have a corresponding audio file, remove it from the data preparation".format(
                id
            )
            logger.info(msg)
        else:
            text_grids_filtered.append(text_grids[i])

    random.shuffle(text_grids_filtered)
    text_grids_filtered_train = text_grids_filtered[:-3]
    text_grids_filtered_dev = [text_grids_filtered[-3]]
    text_grids_filtered_test = text_grids_filtered[-2:]

    clips_train = process_text_grid(text_grids_filtered_train, use_fields)
    clips_train = process_audio_files(clips_train, audio_files, save_folder)

    create_json(clips_train, train_json_file)

    clips_dev = process_text_grid(text_grids_filtered_dev, use_fields)
    clips_dev = process_audio_files(clips_dev, audio_files, save_folder)

    create_json(clips_dev, dev_json_file)

    clips_test = process_text_grid(text_grids_filtered_test, use_fields)
    clips_test = process_audio_files(clips_test, audio_files, save_folder)

    create_json(clips_test, test_json_file)


def process_text_grid(text_grids, use_fields):
    clips = {}
    for grid in text_grids:
        print(grid)
        tg = textgrids.TextGrid(grid)

        contains_all_fields = True
        for field in use_fields:
            if field not in tg.keys():
                contains_all_fields = False

        if contains_all_fields:
            for field in use_fields:
                for syll in tg[field]:
                    # Convert Praat to Unicode in the label
                    label = syll.text.transcode()

                    # Conver IPA syllabus to CMU phones
                    if len(label) > 0:
                        result = subprocess.run(
                            [
                                "python",
                                "lexconvert.py",
                                "--phones2phones",
                                "unicode-ipa",
                                "cmu",
                                str(label),
                            ],
                            capture_output=True,
                        )
                        phones = (
                            result.stdout.decode("UTF-8")
                            .strip()
                            .replace("\n", "")
                        )

                        phone_list = phones.split()
                        phone_list = [
                            phn for phn in phone_list if phn in CMU_PHONES
                        ]
                        phones = " ".join(phone_list)
                        print(phones)

                        grid_name = grid.split(".TextGrid")
                        clip_name = grid_name[0].split("/")[
                            -1
                        ] + ".{}-{}".format(syll.xmin, syll.xmax)

                        if clip_name not in clips:
                            clips[clip_name] = {
                                "start": syll.xmin,
                                "end": syll.xmax,
                                "duration": syll.dur,
                                "TextGrid": grid,
                            }
                        clips[clip_name][field] = phones

    return clips


def process_audio_files(clips, audio_files, save_folder):

    audio_files = set(audio_files)

    for key, val in clips.items():
        text_grid = val["TextGrid"]

        audio_file_name = os.path.join(
            save_folder, text_grid.split(".TextGrid")[0].split("/")[-1] + ".wav"
        )
        if not os.path.exists(audio_file_name):
            audio_file_name = os.path.join(
                save_folder,
                text_grid.split(".TextGrid")[0].split("/")[-1] + ".mp3",
            )

        # read the corresponding portion
        _, sr = torchaudio.load(audio_file_name)

        start = int(val["start"] * sr)
        stop = int(val["end"] * sr)
        wav_obj = {"file": audio_file_name, "start": start, "stop": stop}
        data = read_audio(wav_obj)

        # downsample to 16000 khz
        resampled = torchaudio.transforms.Resample(sr, 16000)(data)

        # save the file to disk
        file_name = os.path.join(save_folder, key + ".wav")
        write_audio(file_name, resampled, 16000)

        # save the file name
        clips[key]["file"] = file_name

    return clips


def create_json(
    clips, json_file,
):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    clips: dict
        The list of wav files of a given data split.
    json_file : str
            The path of the output json file.
    """
    # Adding some Prints
    msg = "Creating %s..." % (json_file)
    logger.info(msg)
    json_dict = {}

    for key, val in clips.items():
        snt_id = key
        spk_id = key.split(".")[0]

        json_dict[snt_id] = {
            "wav": val["file"],
            "spk_id": spk_id,
            "snt_id": snt_id,
            "duration": val["duration"],
            "phn": val["transcribed word"],
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


# prepare_syllabus("/home/mila/j/jianyuan.zhong/ResearchMatch/Syllables_rep/", "/home/mila/j/jianyuan.zhong/ResearchMatch/Syllables_rep_prepared")
