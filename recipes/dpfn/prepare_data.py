"""
The .csv preperation functions for WSJ0-Mix.

Author
 * Cem Subakan 2020

 """

import os
import csv
import json

spk_id_dict = {
    "001": "M",
    "002": "F",
    "00a": "F",
    "00b": "M",
    "00c": "M",
    "00d": "M",
    "00f": "F",
    "010": "M",
    "011": 0,
    "012": 1,
    "013": 2,
    "014": 3,
    "015": 4,
    "016": 5,
    "017": 6,
    "018": 7,
    "019": 8,
    "01L": "M",
    "01a": 9,
    "01b": 10,
    "01c": 11,
    "01d": 12,
    "01e": 13,
    "01f": 14,
    "01g": 15,
    "01h": "F",
    "01i": 16,
    "01j": 17,
    "01k": 18,
    "01l": 19,
    "01m": 20,
    "01n": 21,
    "01o": 22,
    "01p": 23,
    "01q": 24,
    "01r": 25,
    "01s": 26,
    "01t": 27,
    "01u": 28,
    "01v": 29,
    "01w": 30,
    "01x": 31,
    "01y": 32,
    "01z": 33,
    "020": 34,
    "021": 35,
    "022": 36,
    "023": 37,
    "024": 38,
    "025": 39,
    "026": 40,
    "027": 41,
    "028": 42,
    "029": 43,
    "02a": 44,
    "02b": 45,
    "02c": 46,
    "02d": 47,
    "02e": 48,
    "02f": "F",
    "050": 101,
    "051": 102,
    "052": 103,
    "053": 104,
    "200": "M",
    "201": "M",
    "202": "F",
    "203": "F",
    "204": 49,
    "205": 50,
    "206": 51,
    "207": 52,
    "208": 53,
    "209": 54,
    "20a": 55,
    "20b": 56,
    "20c": 57,
    "20d": 58,
    "20e": 59,
    "20f": 60,
    "20g": 61,
    "20h": 62,
    "20i": 63,
    "20j": 64,
    "20k": 65,
    "20l": 66,
    "20m": 67,
    "20n": 68,
    "20o": 69,
    "20p": 70,
    "20q": 71,
    "20r": 72,
    "20s": 73,
    "20t": 74,
    "20u": 75,
    "20v": 76,
    "22g": 105,
    "22h": 106,
    "400": "M",
    "401": 77,
    "403": 78,
    "404": 79,
    "405": 80,
    "406": 81,
    "407": 82,
    "408": 83,
    "409": 84,
    "40a": 85,
    "40b": 86,
    "40c": 87,
    "40d": 88,
    "40e": 89,
    "40f": 90,
    "40g": 91,
    "40h": 92,
    "40i": 93,
    "40j": 94,
    "40k": 95,
    "40l": 96,
    "40m": 97,
    "40n": 98,
    "40o": 99,
    "40p": 100,
    "420": 107,
    "421": 108,
    "422": 109,
    "423": 110,
    "430": "F",
    "431": "M",
    "432": "F",
    "440": 111,
    "441": 112,
    "442": 113,
    "443": 114,
    "444": 115,
    "445": 116,
    "446": 117,
    "447": 118,
}


def prepare_wsjmix(
    datapath,
    savepath,
    n_spks=2,
    skip_prep=False,
    librimix_addnoise=False,
    fs=8000,
):
    """
    Prepared wsj2mix if n_spks=2 and wsj3mix if n_spks=3.

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        librimix_addnoise: If True, add whamnoise to librimix datasets
    """

    if skip_prep:
        return

    if "wsj" in datapath:

        if n_spks == 2:
            assert (
                "2speakers" in datapath
            ), "Inconsistent number of speakers and datapath"
            create_wsj_csv(datapath, savepath)
        elif n_spks == 3:
            assert (
                "3speakers" in datapath
            ), "Inconsistent number of speakers and datapath"
            create_wsj_csv_3spks(datapath, savepath)
        else:
            raise ValueError("Unsupported Number of Speakers")

    if "DPFN" in datapath:

        create_dpfn_csv(datapath, savepath)

    else:
        print("Creating a csv file for a custom dataset")
        create_custom_dataset(datapath, savepath)


def create_custom_dataset(
    datapath,
    savepath,
    dataset_name="custom",
    set_types=["train", "valid", "test"],
    folder_names={
        "source1": "source1",
        "source2": "source2",
        "mixture": "mixture",
    },
):
    """
    This function creates the csv file for a custom source separation dataset
    """

    for set_type in set_types:
        mix_path = os.path.join(datapath, set_type, folder_names["mixture"])
        s1_path = os.path.join(datapath, set_type, folder_names["source1"])
        s2_path = os.path.join(datapath, set_type, folder_names["source2"])

        files = os.listdir(mix_path)

        mix_fl_paths = [os.path.join(mix_path, fl) for fl in files]
        s1_fl_paths = [os.path.join(s1_path, fl) for fl in files]
        s2_fl_paths = [os.path.join(s2_path, fl) for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        with open(
            os.path.join(savepath, dataset_name + "_" + set_type + ".csv"), "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                }
                writer.writerow(row)


def create_wsj_csv(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-2mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
        ]

        with open(savepath + "/wsj_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                }
                writer.writerow(row)


def create_wsj_csv_3spks(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-3mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")
        s3_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s3/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        s3_fl_paths = [s3_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "s3_wav",
            "s3_wav_format",
            "s3_wav_opts",
        ]

        with open(savepath + "/wsj_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path, s3_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths, s3_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "s3_wav": s3_path,
                    "s3_wav_format": "wav",
                    "s3_wav_opts": None,
                }
                writer.writerow(row)


def create_dpfn_csv(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-2mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        ex_path = os.path.join(datapath, "examples_" + set_type)
        ex_dirs = sorted(os.listdir(ex_path))

        mix_fl_paths = [
            os.path.join(ex_path, ex_dir, "mixture.wav") for ex_dir in ex_dirs
        ]
        s1_fl_paths = [
            os.path.join(ex_path, ex_dir, "s1.wav") for ex_dir in ex_dirs
        ]
        s2_fl_paths = [
            os.path.join(ex_path, ex_dir, "s2.wav") for ex_dir in ex_dirs
        ]
        s1_e_fl_paths = [
            os.path.join(ex_path, ex_dir, "s1_estimate.wav")
            for ex_dir in ex_dirs
        ]
        s2_e_fl_paths = [
            os.path.join(ex_path, ex_dir, "s2_estimate.wav")
            for ex_dir in ex_dirs
        ]
        metrics_paths = [
            os.path.join(ex_path, ex_dir, "metrics.json") for ex_dir in ex_dirs
        ]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s1_id",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "s2_id",
            "s1_e_wav",
            "s1_e_wav_format",
            "s1_e_wav_opts",
            "s2_e_wav",
            "s2_e_wav_format",
            "s2_e_wav_opts",
        ]

        with open(savepath + "/wsj_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for (
                i,
                (
                    mix_path,
                    s1_path,
                    s2_path,
                    s1_e_path,
                    s2_e_path,
                    metrics_path,
                ),
            ) in enumerate(
                zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    s2_fl_paths,
                    s1_e_fl_paths,
                    s2_e_fl_paths,
                    metrics_paths,
                )
            ):
                metrics = json.load(open(metrics_path))
                mix_wav = metrics["mix_path"]
                mix_wav_name = mix_wav.split("/")[-1]
                mix_details = mix_wav_name.split("_")
                spk1 = mix_details[0][:3]
                spk2 = mix_details[2][:3]
                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s1_id": spk_id_dict[spk1],
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "s2_id": spk_id_dict[spk2],
                    "s1_e_wav": s1_e_path,
                    "s1_e_wav_format": "wav",
                    "s1_e_wav_opts": None,
                    "s2_e_wav": s2_e_path,
                    "s2_e_wav_format": "wav",
                    "s2_e_wav_opts": None,
                }
                writer.writerow(row)
