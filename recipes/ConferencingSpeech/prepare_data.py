"""
The .csv preperation functions for WSJ0-Mix.

Author
 * Cem Subakan 2020

 """

import os
import csv

def prepare_cs21(
    train_path,
    dev_simu_path,
    dev_real_path,
    eval_path,
    savepath,
    sources=["mix","noise","reverb","clean"],
    skip_prep=False,
    librimix_addnoise=False,
    fs=16000,
):
    """
    Prepared ConferencingSpeech 2021

    Arguments:
    ----------
        train_path (str) : path for the ConferencingSpeech simulated training set.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        librimix_addnoise: If True, add whamnoise to librimix datasets
    """
    if not skip_prep:
        print("Creating a csv file for a custom dataset")
        create_cs21_dataset(train_path, dev_simu_path, dev_real_path,eval_path, savepath)


def create_cs21_dataset(
    train_path,
    dev_simu_path,
    dev_real_path,
    eval_path,
    savepath,
    task=1, # change to 2 for multi array setup (not implemented fully yet)
    dataset_name="cs21",
    sim_folder_names={
        "clean": "noreverb_ref",
        "reverb_mix": "reverb_ref",
        "noise_mix": "noreverb_mix",
        "mixture": "mix",
    },
    real_folder_names={
        "real": "real-recording",
        "real_speaker": "semi-real-realspk",
        "real_noise": "semi-real-playback",
    },
    noise_only_mix=False,
):
    """
    This function creates the csv file for a custom source separation dataset
    """
    datapath_map = {"train_simu":train_path,"dev_simu":dev_simu_path,"dev_real":dev_real_path, "eval_real":eval_path}
    array_list = ["simu_non_uniform"] if task==1 else ["simu_circle","simu_linear","simu_non_uniform"];
    for set_type, set_path in datapath_map.items():
        if "_simu" in set_type:
            for array in array_list:
                mix_path = os.path.join(set_path, array, sim_folder_names["mixture"])
                noise_mix_path = os.path.join(set_path, array, sim_folder_names["noise_mix"])
                reverb_mix_path = os.path.join(set_path, array,sim_folder_names["reverb_mix"])
                clean_path = os.path.join(set_path, array, sim_folder_names["clean"])

                files = os.listdir(mix_path)

                mix_fl_paths = [os.path.join(mix_path, fl) for fl in files]
                noise_mix_fl_paths = [os.path.join(noise_mix_path, fl) for fl in files] if noise_only_mix==True else ["" for fl in files]
                reverb_mix_fl_paths = [os.path.join(reverb_mix_path, fl) for fl in files]
                clean_fl_paths = [os.path.join(original_path, fl) for fl in files]

            csv_columns = [
                "ID",
                "duration",
                "mix_wav",
                "mix_wav_format",
                "mix_wav_opts",
                "noise_wav",
                "noise_wav_format",
                "noise_wav_opts",
                "reverb_wav",
                "reverb_wav_format",
                "reverb_wav_opts",
                "clean_wav",
                "clean_wav_format",
                "clean_wav_opts",
            ]

            with open(
                os.path.join(savepath, dataset_name + "_" + set_type + ".csv"), "w"
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for i, (mix_path, noise_path, reverb_path, clean_path) in enumerate(
                    zip(mix_fl_paths, noise_mix_fl_paths, reverb_mix_fl_paths, clean_fl_paths)
                ):

                    row = {
                        "ID": i,
                        "duration": 1.0,
                        "mix_wav": mix_path,
                        "mix_wav_format": "wav",
                        "mix_wav_opts": None,
                        "noise_wav": noise_path,
                        "noise_wav_format": "wav",
                        "noise_wav_opts": None,
                        "reverb_wav": reverb_path,
                        "reverb_wav_format": "wav",
                        "reverb_wav_opts": None,
                        "clean_wav": clean_path,
                        "clean_wav_format": "wav",
                        "clean_wav_opts": None,
                    }
                    writer.writerow(row)
        elif "_real" in set_type:
            if task == 1:
                real_path = os.path.join(set_path,real_folder_names["real-recording"],"1")
                real_spk_path = os.path.join(set_path,real_folder_names["semi-real-realspk"],"1")
                real_plybk_path = os.path.join(set_path,real_folder_names["semi-real-playback"],"1")

                real_fl_paths = [os.path.join(real_path, fl) for fl in os.listdir(real_path)]
                real_spk_fl_paths = [os.path.join(real_spk_path, fl) for fl in os.listdir(real_spk_path)]
                real_plybk_fl_paths = [os.path.join(real_plybk_path, fl) for fl in os.listdir(real_plybk_path)]

                csv_columns = [
                    "ID",
                    "duration",
                    "real_wav",
                    "real_wav_format",
                    "real_wav_opts",
                    "real_spk_wav",
                    "real_spk_wav_format",
                    "real_spk_wav_opts",
                    "real_plybk_wav",
                    "real_plybk_wav_format",
                    "real_plybk_wav_opts",
                ]

                with open(
                    os.path.join(savepath, dataset_name + "_" + set_type + ".csv"), "w"
                ) as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for i, (real_path, real_spk_path, real_plybk_path) in enumerate(
                        zip(mix_fl_paths, noise_mix_fl_paths, reverb_mix_fl_paths, clean_fl_paths)
                    ):

                        row = {
                            "ID": i,
                            "duration": 1.0,
                            "real_wav": real_path,
                            "real_wav_format": "wav",
                            "real_wav_opts": None,
                            "real_spk_wav": real_spk_path,
                            "real_spk_wav_format": "wav",
                            "real_spk_wav_opts": None,
                            "real_plybk_wav": real_plybk_path,
                            "real_plybk_wav_format": "wav",
                            "real_plybk_wav_opts": None,
                        }
                        writer.writerow(row)
            else:
                raise Exception("Task 2 not implemented yet")

        

        
        


       