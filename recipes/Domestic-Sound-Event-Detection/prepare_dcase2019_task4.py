import os
import csv
import logging
import json

# from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def create_file_path(folder_path, dataset_type, name):
    """Function to create file path (gets call by prepare_dcase2019_task4).

    Arguments
    ---------
    folder_path : string
        path tof directory.
    dataset_type : string
        type of the dataset (ie "train", "validation", "test").
    name : string
        name of the dataset (ie "weak", "strong", "unlabel_in_domain").
    Returns
    -------
    string
        concatenate path
    """
    # creating a output directory if necessary
    dir = os.path.join(folder_path, dataset_type)
    if not check_folders(dir):
        os.makedirs(dir)

    json_name = os.path.join(dir, name.split(".")[0] + ".json")

    return json_name


def prepare_dcase2019_task4(
    save_json_dir,
    wav_dir,
    meta_dir,
    wav_dir_unlabel=None,
    missing_file_dir=None,
):
    """Prepare the metadata as Json file to work better with SB pipeline (.tsv --> .json)
    this function insure aswell that missing files will be remove.

    Arguments
    ---------
    save_json_dir : string
        path to directory where json files will be saved.
    wav_dir : string
        path to directory where wav files are stored.
    meta_dir : string
        path to directory where tsv files are stored.
    wav_dir_unlabel : string
        path to directory where unlabelled wav files are store.
        * this is a large folder, make sure you change in yaml if unlabel are store else where
        please refer to README.md
    missing_file_dir : string
        path to directory where missing files are stored.

    Returns
    -------
    None
    """
    for dataset in ["weak.tsv", "synthetic.tsv", "unlabel_in_domain.tsv"]:
        tsv_file = os.path.join(meta_dir, "train", dataset)
        json_file = create_file_path(save_json_dir, "train", dataset)
        missing_file = os.path.join(
            missing_file_dir, "missing_files_" + dataset
        )
        if dataset == "unlabel_in_domain.tsv" and wav_dir_unlabel is not None:
            audio_folder = os.path.join(
                wav_dir_unlabel
            )  # 'train', dataset.split('.')[0])
        else:
            audio_folder = os.path.join(wav_dir, "train", dataset.split(".")[0])
        create_json(audio_folder, tsv_file, json_file, missing_file)

        if dataset == "weak.tsv":
            json_file = create_file_path(save_json_dir, "toy", dataset)
            create_json(
                audio_folder, tsv_file, json_file, missing_file, toy=True
            )
        if dataset == "synthetic.tsv":
            json_file = create_file_path(save_json_dir, "toy", dataset)
            create_json(
                audio_folder, tsv_file, json_file, missing_file, toy=True
            )
        if dataset == "unlabel_in_domain.tsv":
            json_file = create_file_path(save_json_dir, "toy", dataset)
            create_json(
                audio_folder, tsv_file, json_file, missing_file, toy=True
            )

    ## validation
    for dataset in [
        "validation.tsv",
        "eval_dcase2018.tsv",
        "test_dcase2018.tsv",
    ]:
        tsv_file = os.path.join(meta_dir, "validation", dataset)
        json_file = create_file_path(save_json_dir, "validation", dataset)
        missing_file = os.path.join(
            missing_file_dir, "missing_files_" + dataset
        )
        audio_folder = os.path.join(wav_dir, "validation")

        create_json(audio_folder, tsv_file, json_file, missing_file)

        if dataset == "validation.tsv":
            json_file = create_file_path(save_json_dir, "toy", dataset)
            create_json(
                audio_folder, tsv_file, json_file, missing_file, toy=True
            )

    ## test:
    for dataset in ["public.tsv"]:
        tsv_file = os.path.join(meta_dir, "eval", dataset)
        json_file = create_file_path(save_json_dir, "eval", dataset)
        missing_file = os.path.join(
            missing_file_dir, "missing_files_" + dataset
        )
        audio_folder = os.path.join(wav_dir, "eval", dataset.split(".")[0])

        create_json(audio_folder, tsv_file, json_file, missing_file)

        if dataset == "public.tsv":
            json_file = create_file_path(save_json_dir, "toy", dataset)
            create_json(
                audio_folder, tsv_file, json_file, missing_file, toy=True
            )


# flake8: noqa: C901
def create_json(wav_folder, tsv_file, json_file, missing=None, toy=False):
    """Create json files (gets call by prepare_dcase2019_task4.

    Arguments
    ---------
    wav_folder : string
        path to directory where wav files are stored.
    tsv_file : string
        path to tsv file (meta data).
    jason_file : string
        path to file where json will be saved.
    missing : string
        path to directory where missing files are.
    toy : bool
        if True, small subset will be created.

    Returns
    -------
    None
    """

    if not check_folders(tsv_file):
        raise NameError(f"{tsv_file} doesn't exist!")
    if not check_folders(missing):
        missing = None

    json_dict = {}
    filename_dict = {}

    if missing is not None:

        missing_files = {"filename": []}
        with open(missing, encoding="utf-8", mode="r") as miss:
            reader_miss = csv.reader(miss, dialect="excel-tab")
            next(reader_miss, None)
            for line in reader_miss:
                missing_files["filename"].append(line[0])

    else:
        missing_files = {"filename": []}
    with open(tsv_file, "r") as f_in:
        # Write header unchanged
        reader = csv.reader(f_in, dialect="excel-tab")
        head = next(reader, None)
        if "event_label" in head:
            uttid = 0
            for line in reader:
                if line[0] in missing_files["filename"]:
                    continue
                elif line[-1] == "":
                    continue
                elif not line[0] in filename_dict.keys():
                    filename_dict[line[0]] = [0, uttid]
                    json_dict[f"{uttid}"] = {
                        "filepath": os.path.join(wav_folder, line[0])
                    }
                    json_dict[f"{uttid}"].update(
                        {
                            "event_label": {
                                "event_label_"
                                f"{filename_dict[line[0]][0]}": line[1:]
                            }
                        }
                    )
                    json_dict[f"{uttid}"].update({"event_labels": [line[-1]]})
                    uttid += 1
                else:
                    filename_dict[line[0]][0] += 1
                    json_dict[f"{filename_dict[line[0]][1]}"][
                        "event_label"
                    ].update(
                        {
                            "event_label_"
                            f"{filename_dict[line[0]][0]}": line[1:]
                        }
                    )
                    if (
                        not line[-1]
                        in json_dict[f"{filename_dict[line[0]][1]}"][
                            "event_labels"
                        ]
                    ):
                        json_dict[f"{filename_dict[line[0]][1]}"][
                            "event_labels"
                        ].append(line[-1])
        elif "event_labels" in head:
            uttid = 0
            for line in reader:
                if line[0] in missing_files["filename"]:
                    continue
                elif line[-1] == "":
                    continue
                else:
                    for i in range(len(head)):
                        if i == 0:
                            json_dict[f"{uttid}"] = {
                                "filepath": os.path.join(wav_folder, line[i])
                            }
                        else:
                            if head[i] == "event_labels":
                                json_dict[f"{uttid}"].update(
                                    {head[i]: line[i].split(",")}
                                )
                            else:
                                json_dict[f"{uttid}"].update({head[i]: line[i]})
                    json_dict[f"{uttid}"].update(
                        {
                            "event_label": {
                                "event_label_0": [0, 0, line[-1].split(",")[0]]
                            }
                        }
                    )
                    uttid += 1
        else:
            uttid = 0
            for line in reader:
                if line[0] in missing_files["filename"]:
                    continue
                elif line[-1] == "":
                    continue
                else:
                    for i in range(len(head)):
                        if i == 0:
                            json_dict[f"{uttid}"] = {
                                "filepath": os.path.join(wav_folder, line[i])
                            }
                    json_dict[f"{uttid}"].update({"event_labels": ["unknown"]})
                    json_dict[f"{uttid}"].update(
                        {"event_label": {"event_label_0": [0, 0, "unknown"]}}
                    )
                    uttid += 1

    if toy:
        toy_json_dict = {
            ks: json_dict[ks] for ks in list(json_dict.keys())[:10]
        }
        with open(json_file, mode="w") as json_f:
            json.dump(toy_json_dict, json_f, indent=2)
    else:
        with open(json_file, mode="w") as json_f:
            json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file.split('/')[-1]} successfully created!")
