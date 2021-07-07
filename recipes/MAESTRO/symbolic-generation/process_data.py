import pickle
import numpy as np
import pandas as pd
from hyperpyyaml import load_hyperpyyaml
import muspy as mp
import os
import speechbrain as sb
import sys
import logging
from more_itertools import locate
from tqdm import tqdm


def piano_roll_to_csv(piano_roll, dataset):
    """This function takes a piano roll and creates an input csv file usable by Speechbrain
    Arguments
    ---------
    piano_roll : list
        Multidimensional list of notes
    dataset : string
        Name of dataset (train, test, valid)
    """
    # Flatten roll and cast to string
    if hparams["data"] == "data/Piano-midi.de.pickle":
        flat_data = list(np.concatenate(piano_roll).flat)
    else:
        flat_data = list(sum(piano_roll, []))
    all_flat = [list(map(str, lst)) for lst in flat_data]
    all_flat = [",".join(lst) for lst in all_flat]

    # Create time dimension of defined size
    sequence_list = [
        all_flat[i : i + hparams["sequence_length"]]
        for i in range(0, len(all_flat), hparams["sequence_length"])
    ]

    # Make dataframe and write to CSV
    flat_pd = pd.DataFrame({"notes": sequence_list})
    flat_pd.to_csv(hparams[dataset + "_csv"], index_label="ID")
    return None


def midi_to_pianoroll(split, num_of_songs):
    """
    This function creates CSV from random selected files in MAESTRO dataset
    :param  set: "validation", "train", "test"
            num_of_songs: # of songs to select in the set (int)
    :return: Nothing
            Produces <data>/train.csv, valid.csv, test.csv

    """

    def parse_song(file):
        """
        This function converts one MIDI song to piano roll
        :param file: song to convert (str)
        :return: list of sequences
        """
        sequence_list = []
        piano_roll = mp.to_pianoroll_representation(mp.read(file))

        for seq in piano_roll:
            index_list = list(locate(list(map(bool, seq) and seq)))
            if sum(index_list) > 0:
                sequence_list.append(tuple(index_list))

        return sequence_list

    # Parameters definition
    if split == "valid":
        split = "validation"

    # Song selection
    df = pd.read_csv(os.path.join(hparams["path_name"], hparams["file_name"]))
    song_set = []
    print("Processing data")
    for i in tqdm(range(num_of_songs)):
        for song in df[df.split == split].sample(num_of_songs)["midi_filename"]:
            song_set.append(
                parse_song(os.path.join(hparams["path_name"], song))
            )
    return song_set


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    logging.info("generating datasets...")

    if not os.path.isdir(hparams["data_path"]):
        os.makedirs(hparams["data_path"])

    if hparams["pickle"] == "MAESTRO":
        split_songs = [
            ("train", hparams["MAESTRO"]["num_train_files"]),
            ("valid", hparams["MAESTRO"]["num_valid_files"]),
            ("test", hparams["MAESTRO"]["num_test_files"]),
        ]
        datasets = {}
        for split, songs in split_songs:
            datasets[split] = midi_to_pianoroll(split, songs)
    else:
        datasets = pickle.load(open(hparams["data"], "rb"))

    for dataset in datasets:
        piano_roll_to_csv(datasets[dataset], dataset)
