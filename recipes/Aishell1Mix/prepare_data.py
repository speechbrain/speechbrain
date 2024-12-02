"""
The functions to create the .csv files for Aishell1Mix

Author
 * Cem Subakan 2020
"""

import csv
import functools
import glob
import os
import tarfile
import zipfile
from urllib.request import urlretrieve

import soundfile as sf
import tqdm.contrib.concurrent
from pysndfx import AudioEffectsChain


def prepare_aishell1mix(
    datapath,
    savepath,
    n_spks=2,
    skip_prep=False,
    aishell1mix_addnoise=False,
    fs=8000,
    datafreqs=["8k", "16k"],
    datamodes=["max", "min"],
    datatypes=["mix_clean", "mix_both", "mix_single"],
):
    """

    Prepare .csv files for aishell1mix

    Arguments:
    ----------
        datapath (str) : path for the aishell1mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        aishell1mix_addnoise: If True, add whamnoise to aishell1mix datasets
    """
    if skip_prep:
        return

    # create the datapath folder if it does not exist
    if not os.path.exists(datapath):
        print("the datapath does not exist, we are creating it")
        os.makedirs(datapath, exist_ok=False)

    aishell1_dir = os.path.join(datapath, "data_aishell")
    wham_dir = os.path.join(datapath, "wham_noise")
    aishell1mix_outdir = os.path.join(datapath, "aishell1mix")

    if not os.path.exists(aishell1_dir):
        print("Download Aishell1 into %s" % datapath)
        urlretrieve(
            "https://us.openslr.org/resources/33/data_aishell.tgz",
            os.path.join(datapath, "data_aishell.tgz"),
            reporthook=reporthook,
        )
        urlretrieve(
            "https://us.openslr.org/resources/33/resource_aishell.tgz",
            os.path.join(datapath, "resource_aishell.tgz"),
            reporthook=reporthook,
        )
        extracttar(os.path.join(datapath, "data_aishell.tgz"))
        files = glob.glob(os.path.join(aishell1_dir, "wav/*.gz"))
        for f in files:
            extracttar(f)
        extracttar(os.path.join(datapath, "resource_aishell.tgz"))

    if not os.path.exists(wham_dir):
        print("Download Wham noise dataset into %s" % datapath)
        urlretrieve(
            "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip",
            os.path.join(datapath, "wham_noise.zip"),
            reporthook=reporthook,
        )
        file = zipfile.ZipFile(os.path.join(datapath, "wham_noise.zip"))
        file.extractall(path=datapath)
        os.remove(os.path.join(datapath, "wham_noise.zip"))

    # augment train noise in wham
    # Get train dir
    subdir = os.path.join(wham_dir, "tr")
    # List files in that dir
    sound_paths = glob.glob(os.path.join(subdir, "**/*.wav"), recursive=True)
    # Avoid running this script if it already have been run
    if len(sound_paths) == 60000:
        print(
            "It appears that augmented files have already been generated.\n"
            "Skipping data augmentation."
        )
    elif len(sound_paths) != 20000:
        print(
            "It appears that augmented files have not been generated properly\n"
            "Resuming augmentation."
        )
        originals = [x for x in sound_paths if "sp" not in x]
        to_be_removed_08 = [
            x.replace("sp08", "") for x in sound_paths if "sp08" in x
        ]
        to_be_removed_12 = [
            x.replace("sp12", "") for x in sound_paths if "sp12" in x
        ]
        sound_paths_08 = list(set(originals) - set(to_be_removed_08))
        sound_paths_12 = list(set(originals) - set(to_be_removed_12))
        augment_noise(sound_paths_08, 0.8)
        augment_noise(sound_paths_12, 1.2)
    else:
        print(f"Augmenting {subdir} files")
        # Transform audio speed
        augment_noise(sound_paths, 0.8)
        augment_noise(sound_paths, 1.2)

    from scripts.create_aishell1_metadata import create_aishell1_metadata

    aishell1_md_dir = os.path.join(aishell1_dir, "metadata")
    os.makedirs(aishell1_md_dir, exist_ok=True)
    create_aishell1_metadata(aishell1_dir, aishell1_md_dir)

    from scripts.create_wham_metadata import create_wham_noise_metadata

    wham_md_dir = os.path.join(wham_dir, "meta")
    os.makedirs(wham_md_dir, exist_ok=True)
    create_wham_noise_metadata(wham_dir, wham_md_dir)

    from scripts.create_aishell1mix_metadata import create_aishell1mix_metadata

    aishell1mix_md_outdir = os.path.join(
        aishell1mix_outdir, "metadata", "Aishell1Mix%i" % n_spks
    )
    os.makedirs(aishell1mix_md_outdir, exist_ok=True)
    create_aishell1mix_metadata(
        os.path.join(aishell1_dir, "wav"),
        aishell1_md_dir,
        wham_dir,
        wham_md_dir,
        aishell1mix_md_outdir,
        n_spks,
    )

    from scripts.create_aishell1mix_from_metadata import create_aishell1mix

    aishell1mix_outdir = os.path.join(
        aishell1mix_outdir, "Aishell1Mix%i" % n_spks
    )
    os.makedirs(aishell1mix_outdir, exist_ok=True)
    create_aishell1mix(
        os.path.join(aishell1_dir, "wav"),
        wham_dir,
        aishell1mix_outdir,
        aishell1mix_md_outdir,
        datafreqs,
        n_spks,
        datamodes,
        datatypes,
    )

    if "Aishell1" in aishell1mix_outdir:
        # Aishell1 Mix2/3 datasets
        if n_spks == 2:
            assert (
                "Aishell1Mix2" in aishell1mix_outdir
            ), "Inconsistent number of speakers and datapath"
            create_aishell1mix2_csv(
                aishell1mix_outdir, savepath, addnoise=aishell1mix_addnoise
            )
        elif n_spks == 3:
            assert (
                "Aishell1Mix3" in aishell1mix_outdir
            ), "Inconsistent number of speakers and datapath"
            create_aishell1mix3_csv(
                aishell1mix_outdir, savepath, addnoise=aishell1mix_addnoise
            )
        else:
            raise ValueError("Unsupported Number of Speakers")
    else:
        raise ValueError("Unsupported Dataset")


def create_aishell1mix2_csv(
    datapath,
    savepath,
    addnoise=False,
    version="wav8k/min/",
    set_types=["train", "dev", "test"],
):
    """
    This functions creates the .csv file for the aishell1mix2 dataset
    """

    for set_type in set_types:
        if addnoise:
            mix_path = os.path.join(datapath, version, set_type, "mix_both/")
        else:
            mix_path = os.path.join(datapath, version, set_type, "mix_clean/")

        s1_path = os.path.join(datapath, version, set_type, "s1/")
        s2_path = os.path.join(datapath, version, set_type, "s2/")
        noise_path = os.path.join(datapath, version, set_type, "noise/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        noise_fl_paths = [noise_path + fl for fl in files]

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
            savepath + "/aishell1mix2_" + set_type + ".csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path, noise_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths, noise_fl_paths)
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
                    "noise_wav": noise_path,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                }
                writer.writerow(row)


def create_aishell1mix3_csv(
    datapath,
    savepath,
    addnoise=False,
    version="wav8k/min/",
    set_types=["train", "dev", "test"],
):
    """
    This functions creates the .csv file for the aishell1mix3 dataset
    """

    for set_type in set_types:
        if addnoise:
            mix_path = os.path.join(datapath, version, set_type, "mix_both/")
        else:
            mix_path = os.path.join(datapath, version, set_type, "mix_clean/")

        s1_path = os.path.join(datapath, version, set_type, "s1/")
        s2_path = os.path.join(datapath, version, set_type, "s2/")
        s3_path = os.path.join(datapath, version, set_type, "s3/")
        noise_path = os.path.join(datapath, version, set_type, "noise/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        s3_fl_paths = [s3_path + fl for fl in files]
        noise_fl_paths = [noise_path + fl for fl in files]

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
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        with open(
            savepath + "/aishell1mix3_" + set_type + ".csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for (
                i,
                (mix_path, s1_path, s2_path, s3_path, noise_path),
            ) in enumerate(
                zip(
                    mix_fl_paths,
                    s1_fl_paths,
                    s2_fl_paths,
                    s3_fl_paths,
                    noise_fl_paths,
                )
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
                    "noise_wav": noise_path,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                }
                writer.writerow(row)


def extracttar(filename):
    tar = tarfile.open(filename)
    tar.extractall(path=os.path.dirname(filename))
    tar.close()
    os.remove(filename)


def augment_noise(sound_paths, speed):
    print(f"Change speed with factor {speed}")
    tqdm.contrib.concurrent.process_map(
        functools.partial(apply_fx, speed=speed), sound_paths, chunksize=10
    )


def apply_fx(sound_path, speed):
    # Get the effect
    fx = AudioEffectsChain().speed(speed)
    s, rate = sf.read(sound_path)
    # Get 1st channel
    s = s[:, 0]
    # Apply effect
    s = fx(s)
    # Write the file
    sf.write(
        f"""{sound_path.replace('.wav', f"sp{str(speed).replace('.', '')}" + '.wav')}""",
        s,
        rate,
    )


def reporthook(blocknum, blocksize, totalsize):
    print(
        "\rdownloading: %5.1f%%" % (100.0 * blocknum * blocksize / totalsize),
        end="",
    )
