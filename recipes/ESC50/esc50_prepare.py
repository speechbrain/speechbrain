"""
Creates data manifest files for ESC50
If the data does not exist in the specified --data_folder, we download the data automatically.

https://github.com/karolpiczak/ESC-50/

Authors:
 * Cem Subakan 2022, 2023
 * Francesco Paissan 2022, 2023

 Adapted from the Urbansound8k recipe.
"""

import json
import os
import shutil

import torch
import torchaudio

import speechbrain as sb
from speechbrain.dataio.dataio import load_data_csv, read_audio
from speechbrain.utils.fetching import LocalStrategy, fetch
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

ESC50_DOWNLOAD_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
MODIFIED_METADATA_FILE_NAME = "esc50_speechbrain.csv"

ACCEPTABLE_FOLD_NUMS = [1, 2, 3, 4, 5]


def download_esc50(data_path):
    """
    This function automatically downloads the ESC50 dataset to the specified data path in the data_path variable

    Arguments
    ---------
    data_path: str or Path
        Directory used to save the dataset.
    """
    if not os.path.exists(os.path.join(data_path, "meta")):
        print(
            f"ESC50 is missing. We are now downloading it. Be patient it's a 600M file. You can check {data_path}/temp_download to see the download progression"
        )
        temp_path = os.path.join(data_path, "temp_download")

        # download the data
        archive_path = fetch(
            "master.zip",
            "https://github.com/karolpiczak/ESC-50/archive/",  # noqa ignore-url-check
            savedir=temp_path,
            # URL, so will be fetched directly in the savedir anyway
            local_strategy=LocalStrategy.COPY_SKIP_CACHE,
        )

        # unpack the .zip file
        shutil.unpack_archive(archive_path, data_path)

        # move the files up to the datapath
        files = os.listdir(os.path.join(data_path, "ESC-50-master"))
        for fl in files:
            shutil.move(os.path.join(data_path, "ESC-50-master", fl), data_path)

        # remove the unused datapath
        shutil.rmtree(os.path.join(data_path, "temp_download"))
        shutil.rmtree(os.path.join(data_path, "ESC-50-master"))

        print(f"ESC50 is downloaded in {data_path}")


def prepare_esc50(
    data_folder,
    audio_data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    train_fold_nums=[1, 2, 3],
    valid_fold_nums=[4],
    test_fold_nums=[5],
    skip_manifest_creation=False,
):
    """
    Prepares the json files for the ESC50 dataset.
    Prompts to download the dataset if it is not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the ESC50 dataset (including the metadata) is stored.
    audio_data_folder: str
        Path to the folder where the ESC50 dataset audio files are stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    train_fold_nums : list or int (integers [1,5])
        A list of integers defining which pre-defined "folds" to use for training. Must be
        exclusive of valid_folds and test_folds.
    valid_fold_nums : list or int (integers [1,5])
        A list of integers defining which pre-defined "folds" to use for validation. Must be
        exclusive of train_folds and test_folds.
    test_fold_nums : list or int (integers [1,5])
        A list of integers defining which pre-defined "folds" to use for test. Must be
        exclusive of train_folds and valid_folds.
    skip_manifest_creation : bool
        Whether to skip over the manifest creation step.

    Returns
    -------
    None

    Example
    -------
    >>> data_folder = '/path/to/ESC-50-master'
    >>> prepare_urban_sound_8k(data_folder, 'train.json', 'valid.json', 'test.json', [1,2,3], [4], [5])
    """
    download_esc50(data_folder)

    # Tease params to correct type if necessary
    if type(train_fold_nums) is int:
        train_fold_nums = [train_fold_nums]
    if type(valid_fold_nums) is int:
        valid_fold_nums = [valid_fold_nums]
    if type(test_fold_nums) is int:
        test_fold_nums = [test_fold_nums]

    # Validate passed fold params
    for fold_num in train_fold_nums:
        if fold_num not in ACCEPTABLE_FOLD_NUMS:
            print(
                f"Train fold numbers {train_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            logger.info(
                f"Train fold numbers {train_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            return
    for fold_num in valid_fold_nums:
        if fold_num not in ACCEPTABLE_FOLD_NUMS:
            print(
                f"Validation fold numbers {valid_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            logger.info(
                f"Validation fold numbers {valid_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            return
    for fold_num in test_fold_nums:
        if fold_num not in ACCEPTABLE_FOLD_NUMS:
            print(
                f"Test fold numbers {test_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            logger.info(
                f"Test fold numbers {test_fold_nums}, contains an invalid value. Must be in {ACCEPTABLE_FOLD_NUMS}"
            )
            return

    # Check if train, and valid and train and test folds are exclusive
    if folds_overlap(train_fold_nums, valid_fold_nums):
        print(
            f"Train {train_fold_nums}, and Valid {valid_fold_nums} folds must be mutually exclusive!"
        )
        logger.info(
            f"Train {train_fold_nums}, and Valid {valid_fold_nums} folds must be mutually exclusive!"
        )
        return
    if folds_overlap(train_fold_nums, test_fold_nums):
        print(
            f"Train {train_fold_nums} and Test {test_fold_nums} folds must be mutually exclusive!"
        )
        logger.info(
            f"Train {train_fold_nums} and Test {test_fold_nums} folds must be mutually exclusive!"
        )
        return

    # If the dataset doesn't exist yet, prompt the user to set or download it

    # Don't need to do this every single time
    if skip_manifest_creation is True:
        return

    # If our modified metadata file does not exist, create it
    esc50_speechbrain_metadata_csv_path = os.path.join(
        os.path.abspath(data_folder), "metadata/", MODIFIED_METADATA_FILE_NAME
    )
    if not os.path.exists(esc50_speechbrain_metadata_csv_path):
        esc50_speechbrain_metadata_csv_path = create_metadata_speechbrain_file(
            data_folder
        )

    # Read the metadata into a dictionary
    # Every key of this dictionary is now one of the sound filenames, without the ".wav" suffix
    metadata = load_data_csv(esc50_speechbrain_metadata_csv_path)

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )

    # Creating json files
    create_json(metadata, audio_data_folder, train_fold_nums, save_json_train)
    create_json(metadata, audio_data_folder, valid_fold_nums, save_json_valid)
    create_json(metadata, audio_data_folder, test_fold_nums, save_json_test)


def create_json(metadata, audio_data_folder, folds_list, json_file):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    metadata: dict
        A dictionary containing the ESC50 metadata file modified for the
        SpeechBrain, such that keys are IDs (which are the .wav file names without the file extension).
    audio_data_folder : str or Path
        Data folder that stores ESC50 samples.
    folds_list : list of int
        The list of folds [1,5] to include in this batch
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list
    json_dict = {}

    for ID, sample_metadata in metadata.items():
        fold_num = int(sample_metadata["fold"])
        if fold_num in folds_list:
            # Reading the signal (to retrieve duration in seconds)
            wav_file = os.path.join(
                os.path.abspath(audio_data_folder),
                # "fold" + str(fold_num) + "/",
                sample_metadata["filename"],
            )
            try:
                signal = read_audio(wav_file)
                file_info = torchaudio.info(wav_file)

                # If we're using sox/soundfile backend, file_info will have the old type
                if isinstance(file_info, torchaudio.AudioMetaData):
                    duration = signal.shape[0] / file_info.sample_rate
                else:
                    duration = signal.shape[0] / file_info[0].rate

                # Create entry for this sample ONLY if we have successfully read-in the file using SpeechBrain/torchaudio
                json_dict[ID] = {
                    "wav": sample_metadata["filename"],
                    "classID": int(sample_metadata["target"]),
                    "class_string": sample_metadata["class_string"],
                    # "salience": int(sample_metadata["salience"]),
                    "fold": sample_metadata["fold"],
                    "duration": duration,
                }
            except Exception:
                print(
                    f"There was a problem reading the file:{wav_file}. Skipping duration field for it."
                )
                logger.exception(
                    f"There was a problem reading the file:{wav_file}. Skipping it."
                )

    # Writing the dictionary to the json file
    # Need to make sure sub folder "manifest" exists, if not create it
    parent_dir = os.path.dirname(json_file)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    with open(json_file, mode="w", encoding="utf-8") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def folds_overlap(list1, list2):
    """Returns True if any passed lists has incorrect type OR has items in common.

    Arguments
    ---------
    list1 : list
        First list for comparison.
    list2 : list
        Second list for comparison.

    Returns
    -------
    overlap : bool
        Whether lists overlap.
    """
    if not isinstance(list1, list) or not isinstance(list2, list):
        return True
    if any(item in list1 for item in list2):
        return True
    return False


def check_folders(*folders):
    """Returns False if any passed folder does not exist.

    Arguments
    ---------
    *folders: list
        Folders to check.

    Returns
    -------
    pass: bool
    """
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def full_path_to_audio_file(data_folder, slice_file_name, fold_num):
    """Get path to file given slice file name and fold number

    Arguments
    ---------
    data_folder : str
        Folder that contains the dataset.
    slice_file_name : str
        Filename.
    fold_num : int
        Fold number.

    Returns
    -------
    string containing absolute path to corresponding file
    """
    return os.path.join(
        os.path.abspath(data_folder),
        "audio/",
        "fold" + str(fold_num) + "/",
        slice_file_name,
    )


def create_metadata_speechbrain_file(data_folder):
    """Get path to file given slice file name and fold number

    Arguments
    ---------
    data_folder : str
        ESC50 data folder.

    Returns
    -------
    string containing absolute path to metadata csv file modified for SpeechBrain or None if source file not found
    """
    import pandas as pd

    esc50_metadata_csv_path = os.path.join(
        os.path.abspath(data_folder), "meta/esc50.csv"
    )
    if not os.path.exists(esc50_metadata_csv_path):
        return None

    esc50_metadata_df = pd.read_csv(esc50_metadata_csv_path)
    # SpeechBrain wants an ID column
    esc50_metadata_df["ID"] = esc50_metadata_df.apply(
        lambda row: removesuffix(row["filename"], ".wav"), axis=1
    )
    esc50_metadata_df = esc50_metadata_df.rename(
        columns={"category": "class_string"}
    )

    esc50_speechbrain_metadata_csv_path = os.path.join(
        os.path.abspath(data_folder), "meta/", MODIFIED_METADATA_FILE_NAME
    )
    esc50_metadata_df.to_csv(esc50_speechbrain_metadata_csv_path, index=False)
    return esc50_speechbrain_metadata_csv_path


def removesuffix(some_string, suffix):
    """Removed a suffix from a string

    Arguments
    ---------
    some_string : str
        Any string.
    suffix : str
        Suffix to be removed from some_string.

    Returns
    -------
    string resulting from suffix removed from some_string, if found, unchanged otherwise
    """
    if some_string.endswith(suffix):
        return some_string[: -1 * len(suffix)]
    else:
        return some_string


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_audio_folder = hparams["audio_data_folder"]
    config_sample_rate = hparams["sample_rate"]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        wave_file = data_audio_folder + "/{:}".format(wav)

        sig, read_sr = torchaudio.load(wave_file)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        sig = sig.float()
        sig = sig / sig.max()
        return sig

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_string")
    @sb.utils.data_pipeline.provides("class_string", "class_string_encoded")
    def label_pipeline(class_string):
        yield class_string
        class_string_encoded = label_encoder.encode_label_torch(class_string)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "class_string_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="class_string",
    )

    return datasets, label_encoder
