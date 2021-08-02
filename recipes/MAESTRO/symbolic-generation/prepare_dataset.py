import os
import torch
import pickle
import shutil
import numpy as np
import pandas as pd
import muspy as mp
from more_itertools import locate
from tqdm import tqdm
from zipfile import ZipFile
from speechbrain.utils.data_utils import download_file


def prepare_dataset(
    data_folder, dataset_name, train_csv, valid_csv, test_csv, hparams
):
    """
    This class prepares the datasets for symbolic music generation.
    The supported datasets are MAESTRO_v2, MAESTRO_v3, JSB_chorales, MuseData, Nottingham, Piano-Midi

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original dataset is stored. If the  dataset is not
        detected there, we will automatically download there.
    dataset_name : str
        Name of the supported datasets. It must be one of
        [MAESTRO_v2, MAESTRO_v3, JSB_chorales, MuseData, Nottingham, Piano-Midi]
    train_csv : path
        path where the training csv file containing the annotations will be stored.
    valid_csv : path
        path where the validation csv file containing the annotations will be stored.
    test_csv : path
        path where the test csv file containing the annotations will be stored.
    hparams: dict
        hparam file where additional information of the splits are reported.
    """

    # check if the csv files exist, and if not create new ones
    train_csv_exists = True if os.path.isfile(train_csv) else False
    valid_csv_exists = True if os.path.isfile(valid_csv) else False
    test_csv_exists = True if os.path.isfile(test_csv) else False

    # set the names to name the downloaded file
    if dataset_name in ["MAESTRO_v2", "MAESTRO_v3"]:
        data_savepath = data_folder + ".zip"
    else:
        data_savename = data_folder.split("/")[-1] + ".pickle"
        data_savepath = os.path.join(data_folder, data_savename)

    # download dataset if needed
    if not os.path.exists(data_folder):
        download_data(data_folder, dataset_name, data_savepath)

    # generating csv files with the piano_roll annotation
    if not (train_csv_exists and valid_csv_exists and test_csv_exists):
        # if we work with MAESTRO
        if dataset_name in ["MAESTRO_v2", "MAESTRO_v3"]:
            split_songs = [
                ("train", hparams["MAESTRO_params"]["num_train_files"]),
                ("valid", hparams["MAESTRO_params"]["num_valid_files"]),
                ("test", hparams["MAESTRO_params"]["num_test_files"]),
            ]
            for split, songs in split_songs:
                save_piano_rolls(split, songs, hparams)
        else:
            # download the dataset in original format if it doesn't exist on data_path
            datasets = pickle.load(open(data_savepath, "rb"))

            for dataset in datasets:
                save_piano_rolls(datasets[dataset], dataset, hparams)


def download_data(data_folder, dataset_name, data_savepath):
    """Downloads the dataset" for symbolic music generation."""
    DL_link = return_DL_link(dataset_name)
    download_file(DL_link, data_savepath)

    if dataset_name in ["MAESTRO_v2", "MAESTRO_v3"]:

        with ZipFile(data_savepath, "r") as zipOb:
            zipOb.extractall(data_folder)

        if dataset_name == "MAESTRO_v2":
            files = os.listdir(os.path.join(data_folder, "maestro-v2.0.0"))
            for file in files:
                shutil.move(
                    os.path.join(data_folder, "maestro-v2.0.0", file),
                    data_folder,
                )
        elif dataset_name == "MAESTRO_v3":
            files = os.listdir(os.path.join(data_folder, "maestro-v3.0.0"))
            for file in files:
                shutil.move(
                    os.path.join(data_folder, "maestro-v3.0.0", file),
                    data_folder,
                )
        else:
            raise ValueError("Unsupported MAESTRO dataset name")


def return_DL_link(dataset_name):
    """Gets the download links for the specified datasets."""

    if dataset_name == "MAESTRO_v2":
        DL_link = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"
    elif dataset_name == "MAESTRO_v3":
        DL_link = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    elif dataset_name == "JSB_chorales":
        DL_link = (
            "http://www-ens.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle"
        )
    elif dataset_name == "Piano-Midi":
        DL_link = (
            "http://www-ens.iro.umontreal.ca/~boulanni/Piano-midi.de.pickle"
        )
    elif dataset_name == "Nottingham":
        DL_link = "http://www-ens.iro.umontreal.ca/~boulanni/Nottingham.pickle"
    elif dataset_name == "MuseData":
        DL_link = "http://www-ens.iro.umontreal.ca/~boulanni/MuseData.pickle"
    else:
        raise ValueError(
            "The dataset name you entered is not supported. Supported datasetnames are: MAESTRO_v2, MAESTRO_v3, JSB_chorales, Piano-Midi, Nottingham, MuseData"
        )

    return DL_link


def save_piano_rolls(counter, piano_roll, split, hparams):
    """This function takes a piano roll and saves an input csv file usable by Speechbrain

    Arguments
    ---------
    counter: int
        integer the song index
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

    # Create time dimension of defined size
    all_paths = []
    for i in range(0, len(flat_data), hparams["sequence_length"]):
        chunk = flat_data[i : i + hparams["sequence_length"]]

        pitches = []
        times = []
        for j, tupl in enumerate(chunk):
            pitches.extend([el - 21 for el in tupl])
            times.extend([j for el in tupl])

        save_path = os.path.join(
            hparams["data_folder"], split, "{}_{}.pkl".format(counter, i)
        )
        torch.save({"notes": pitches, "times": times}, save_path)
        all_paths.append(save_path)

    # open the data frame
    df_row = pd.DataFrame({"file_path": all_paths})
    try:
        df_all = pd.read_csv(hparams[split + "_csv"])
        df_all = df_all.append(df_row, ignore_index=True)

        df_all.reset_index()
        df_all.ID = df_all.index
    except FileNotFoundError:
        print("Data preparation...")
        df_all = df_row

    df_all.to_csv(hparams[split + "_csv"], index_label="ID")


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

    # parameters definition
    if split == "valid":
        split = "validation"

    # Song selection
    df = pd.read_csv(
        os.path.join(hparams["data_folder"], hparams["maestro_csv"])
    )
    print("Processing data")
    selected_songs = df[df.split == split].sample(num_of_songs)["midi_filename"]

    # to assure compatibility with the other datasets
    if split == "validation":
        split = "valid"

    os.makedirs(os.path.join(hparams["data_folder"], split))

    # save after each 100 songs
    counter = 0
    for song in tqdm(selected_songs):
        song = parse_song(os.path.join(hparams["data_folder"], song))

        save_piano_rolls(counter, [song], split, hparams)
        counter = counter + 1
