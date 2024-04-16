"""
Code snippet to extract the audio from the raw spoken Multiwoz dataset and organize them per dialogue per ata split.

Author
    * Lucas Druart 2024
"""

import argparse
import os
from glob import iglob

import h5py
import scipy.io.wavfile
import tqdm


def pcm2wav(filepath):
    """
    Given a filepath to a h5p file (.hd5) we read the pcm audio of each turn and convert it to wav format.
    """
    data = h5py.File(filepath, "r")
    for group in list(data.keys()):
        # Selecting the pcm audio of each group (i.e. turn) and converting it to wav
        audio_pcm = data[group]["audio"][:]
        turn = group.split(" ")[-1]
        dialogue = group.split(" ")[-3].split(".")[0]
        newPath = os.path.join(
            os.path.dirname(filepath), dialogue, "Turn-" + turn + ".wav"
        )
        if not os.path.isdir(os.path.dirname(newPath)):
            os.mkdir(os.path.dirname(newPath))
        scipy.io.wavfile.write(newPath, 16000, audio_pcm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        required=False,
        type=str,
        help="The path where to find the hd5 files which contain the audio.",
        default=os.path.join(os.pardir, "data"),
    )
    args = parser.parse_args()

    splits = [
        "DSTC11_train_tts",
        "DSTC11_dev_tts",
        "DSTC11_dev_human",
        "DSTC11_test_tts",
        "DSTC11_test_human",
    ]
    for split in splits:
        print(f"Extracting audio from {split} split...")
        split_path = os.path.join(args.data_folder, split)
        for filePath in tqdm.tqdm(iglob(split_path + "/*.hd5")):
            pcm2wav(filePath)


main()
