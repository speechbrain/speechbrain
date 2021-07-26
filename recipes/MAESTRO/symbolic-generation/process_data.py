import numpy as np
import pandas as pd
import muspy as mp
import os
from more_itertools import locate
from tqdm import tqdm


def piano_roll_to_csv(piano_roll, split, hparams):
    """This function takes a piano roll and creates an input csv file usable by Speechbrain
    Arguments
    ---------
    piano_roll : list
        Multidimensional list of notes
    split : string
        Name of split (train, test, valid)
    hparams : dictionary
        The experiment hparams file
    """
    # Flatten roll and cast to string
    if hparams["dataset_name"] == "Piano-Midi":
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
    flat_pd.to_csv(hparams[split + "_csv"], index_label="ID")
    return None


def midi_to_pianoroll(split, num_of_songs, hparams):
    """
    This function creates CSV from random selected files in MAESTRO dataset
    :param  set: "validation", "train", "test"
            num_of_songs: number of songs to select in the set (int)
            hparams: the hparams dictionary for the experiment defined by the .yaml file

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
    df = pd.read_csv(os.path.join(hparams["data_path"], hparams["maestro_csv"]))
    song_set = []
    print("Processing data")
    selected_songs = df[df.split == split].sample(num_of_songs)["midi_filename"]
    for song in tqdm(selected_songs):
        song_set.append(parse_song(os.path.join(hparams["data_path"], song)))
    return song_set
