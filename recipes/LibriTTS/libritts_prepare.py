"""
LibriTTS data preparation

Authors
 * Pradnya Kandarkar 2022
"""

import json
import logging
import os
import random
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm

import speechbrain as sb
from speechbrain.dataio.batch import PaddedData
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.preparation import FeatureExtractor
from speechbrain.utils.data_utils import download_file, get_all_files

logger = logging.getLogger(__name__)
LIBRITTS_URL_PREFIX = "https://www.openslr.org/resources/60/"


def prepare_libritts(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    sample_rate,
    split_ratio=[80, 10, 10],
    save_folder=None,
    libritts_subsets=None,
    train_split=None,
    valid_split=None,
    test_split=None,
    seed=1234,
    model_name=None,
    extract_features=None,
    extract_features_opts=None,
    device="cpu",
):
    """
    Prepares the json files for the LibriTTS dataset.
    Downloads the dataset if it is not found in the `data_folder` as expected.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the LibriTTS dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    sample_rate : int
        The sample rate to be used for the dataset
    split_ratio : list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    save_folder : str | path-like
        The folder where processed data will be saved
    libritts_subsets: list
        List of librispeech subsets to use (e.g., dev-clean, train-clean-100, ...) for the experiment.
        This parameter will be ignored if explicit data splits are provided.
        Explicit data splits parameters: "train_split", "valid_split", "test_split"
    train_split : list
        List of librispeech subsets to use (e.g.,train-clean-100, train-clean-360) for the experiment training stage.
    valid_split : list
        List of librispeech subsets to use (e.g., dev-clean) for the experiment validation stage.
    test_split : list
        List of librispeech subsets to use (e.g., test-clean) for the experiment testing stage.
    seed : int
        Seed value
    model_name : str
        Model name (used to prepare additional model specific data)
    extract_features : list
        The list of features to be extracted
    extract_features_opts : dict
        Options for feature extraction
    device : str | torch.device
        The device for to be used for computation

    Returns
    -------
    result : None
    """

    # Setting the seed value
    random.seed(seed)

    if save_folder is None:
        save_folder = data_folder

    # Checks if this phase is already done (if so, skips it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    extract_features_context = None
    extract_features_folder = None
    if extract_features:
        extract_features_context = get_context(
            extract_features=extract_features,
            extract_features_opts=extract_features_opts or {},
            device=device,
        )
        extract_features_folder = Path(save_folder) / "features"

    # If specific splits are provided, creates data manifest files accordingly
    if train_split:
        wav_list = prepare_split(data_folder, train_split)
        create_json(
            data_folder,
            wav_list,
            save_json_train,
            sample_rate,
            model_name,
            extract_features,
            extract_features_context,
            extract_features_folder,
            extract_features_opts,
            device,
        )
    if valid_split:
        wav_list = prepare_split(data_folder, valid_split)
        create_json(
            data_folder,
            wav_list,
            save_json_valid,
            sample_rate,
            model_name,
            extract_features,
            extract_features_context,
            extract_features_folder,
            extract_features_opts,
            device,
        )
    if test_split:
        wav_list = prepare_split(data_folder, test_split)
        create_json(
            data_folder,
            wav_list,
            save_json_test,
            sample_rate,
            model_name,
            extract_features,
            extract_features_context,
            extract_features_folder,
            extract_features_opts,
            device,
        )

    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed.")
        return

    # If specific splits are not provided, and a list of subsets if provided, creates train, valid, test splits
    # Creates data manifest files according to the data splits
    if libritts_subsets:
        wav_list = prepare_split(data_folder, libritts_subsets)
        # Random split the signal list into train, valid, and test sets.
        data_split = split_sets(wav_list, split_ratio)
        # Creating json files
        create_json(data_split["train"], save_json_train, sample_rate)
        create_json(data_split["valid"], save_json_valid, sample_rate)
        create_json(data_split["test"], save_json_test, sample_rate)


def prepare_split(data_folder, split_list):
    """
    Processes the provided list of LibriTTS subsets and creates a list of all the .wav files present in the subsets.
    Downloads the LibriTTS subsets as required.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the LibriTTS dataset is stored
    split_list : list
        List of librispeech subsets to process (e.g., dev-clean, train-clean-100, ...)

    Returns
    -------
    wav_list : list
        List of all .wav files to be processed
    """
    extension = [".wav"]  # The expected extension for audio files
    wav_list = list()  # Stores all audio file paths for the dataset

    # For every subset of the dataset, if it doesn't exist, downloads it
    for subset_name in split_list:

        subset_folder = os.path.join(data_folder, subset_name)
        subset_archive = os.path.join(subset_folder, subset_name + ".tar.gz")

        subset_data = os.path.join(subset_folder, "LibriTTS")
        if not check_folders(subset_data):
            logger.info(
                f"No data found for {subset_name}. Checking for an archive file."
            )
            if not os.path.isfile(subset_archive):
                logger.info(
                    f"No archive file found for {subset_name}. Downloading and unpacking."
                )
                subset_url = LIBRITTS_URL_PREFIX + subset_name + ".tar.gz"
                download_file(subset_url, subset_archive)
                logger.info(f"Downloaded data for subset {subset_name}.")
            else:
                logger.info(
                    f"Found an archive file for {subset_name}. Unpacking."
                )

            shutil.unpack_archive(subset_archive, subset_folder)

        # Collects all files matching the provided extension
        wav_list.extend(get_all_files(subset_folder, match_and=extension))
    wav_list = [
        file_name
        for file_name in wav_list
        if not Path(file_name).name.startswith("._")
    ]
    return wav_list


def create_json(
    data_folder,
    wav_list,
    json_file,
    sample_rate,
    model_name=None,
    extract_features=None,
    extract_features_context=None,
    extract_features_folder=None,
    extract_features_opts=None,
    device="cpu",
):
    """
    Creates the json file given a list of wav files.
    Arguments
    ---------
    data_folder : str | path-like
        The folder where the dataset is stored
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    sample_rate : int
        The sample rate to be used for the dataset
    model_name : str
        Model name (used to prepare additional model specific data)
    extract_features: list, optional
        If specified, feature extraction will be performed
    extract_features_context: types.SimpleNamespace, optional
        Context for feature extraction (pretrained models, etc)
    extract_features_folder : path-like, optional
        The folder where extracted features will be saved
    extract_features_opts : dict, optional
        Options for feature extraction
    device : str
        Device for to be used for computation (used as required)

    """

    json_dict = {}

    # Processes all the wav files in the list
    for wav_file in tqdm(wav_list):

        # Reads the signal
        signal, sig_sr = torchaudio.load(wav_file)
        duration = signal.shape[1] / sig_sr
        # Manipulates path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-6:])

        # Gets the path for the text files and extracts the input text
        normalized_text_path = os.path.join(
            "/", *path_parts[:-1], uttid + ".normalized.txt"
        )
        with open(normalized_text_path) as f:
            normalized_text = f.read()
            if normalized_text.__contains__("{"):
                normalized_text = normalized_text.replace("{", "")
            if normalized_text.__contains__("}"):
                normalized_text = normalized_text.replace("}", "")

        # Resamples the audio file if required
        if sig_sr != sample_rate:
            resampled_signal = torchaudio.functional.resample(
                signal, sig_sr, sample_rate
            )
            os.unlink(wav_file)
            torchaudio.save(wav_file, resampled_signal, sample_rate=sample_rate)

        # Gets the speaker-id from the utterance-id
        spk_id = uttid.split("_")[0]

        # Creates an entry for the utterance
        json_dict[uttid] = {
            "uttid": uttid,
            "wav": relative_path,
            "duration": duration,
            "spk_id": spk_id,
            "label": normalized_text,
            "segment": True if "train" in json_file else False,
        }

    # Feature Extraction
    if extract_features:
        extract_features_folder.mkdir(exist_ok=True, parents=True)
        prepare_features(
            data=json_dict,
            data_folder=data_folder,
            save_path=extract_features_folder,
            features=extract_features,
            context=extract_features_context,
            options=extract_features_opts,
            device=device,
        )

    # Writes the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    *filenames : list
        A list of filenames expected to be generated

    Returns
    -------
    result : bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if filename is not None and not os.path.isfile(filename):
            return False
    return True


def split_sets(wav_list, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.

    Arguments
    ---------
    wav_list : list
        A list of all the signals in the dataset
    split_ratio: list
        A list composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Returns
    -------
    result: dict
        A dictionary containing train, valid, and test splits.
    """
    # Random shuffles the list
    random.shuffle(wav_list)
    tot_split = sum(split_ratio)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = wav_list[0:n_snts]
        del wav_list[0:n_snts]
    data_split["test"] = wav_list

    return data_split


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def get_alignment_path(data_folder, alignments_folder, file_name):
    """Returns the path in the LibriSpeech-Alignments dataset
    corresponding to the specified file path in LibriSpeech

    Arguments
    ---------
    data_folder: str
        the path to LibriSpeech
    alignments_folder: str
        the path to LibriSpeech-Alignments
    file_name: str
        the file name within LibriSpeech

    Returns
    -------
    file_name: str
        the alignment file path
    """
    file_name = Path(file_name)
    data_folder = Path(data_folder)
    if file_name.parts[0] == "{data_root}":
        file_name_rel = file_name.relative_to("{data_root}")
    else:
        file_name_rel = file_name.relative_to(data_folder)
    data_slice = file_name_rel.parts[0]

    textgrid_folder = file_name_rel.relative_to(
        Path(data_slice) / "LibriTTS" / data_slice
    ).parent.parent
    textgrid_file_name = f"{file_name_rel.stem}.TextGrid"
    textgrid_path = (
        Path(alignments_folder)
        / data_slice
        / textgrid_folder
        / textgrid_file_name
    )

    return textgrid_path


def parse_alignments(file_name):
    """Parses a given LibriSpeech-Alignments TextGrid file and
    converts the results to the desired format (to be used in JSON
    metadata)

    Arguments
    ---------
    file_name : path-like
        the file name of the TextGrid file

    Returns
    -------
    details: dict
        the metadata details
    """
    try:
        import textgrids
    except ImportError:
        logger.error(
            "Parsing LibriSpeech-alignments requires the"
            "praat-textgrids package"
        )
        raise
    if not file_name.exists():
        return {
            "has_alignments": False,
            "phn": [],
            "phn_stress": [],
            "phn_start": [],
            "phn_end": [],
            "phn_count": 0,
            "wrd": [],
            "wrd_start": [],
            "wrd_end": [],
            "wrd_count": 0,
            "unk_count": None,
        }

    text_grid = textgrids.TextGrid()
    text_grid.read(file_name)
    word_intervals = [
        {**word, "label": word["label"].upper()}
        for word in text_grid.interval_tier_to_array("words")
    ]
    phn_intervals = text_grid.interval_tier_to_array("phones")
    details = {}
    details.update(intervals_to_dict(word_intervals, "wrd"))
    phn = intervals_to_dict(phn_intervals, "phn")
    phn_stress = phn["phn"]
    phn_nostress = remove_stress_marks(phn_stress)
    phn["phn"] = phn_nostress
    phn["phn_stress"] = phn_stress
    details.update(phn)
    details["unk_count"] = sum(wrd == "<UNK>" for wrd in details["wrd"])
    details["has_alignments"] = True

    return details


INTERVAL_MAP = [("label", ""), ("begin", "_start"), ("end", "_end")]
INTERVAL_EMPTY_LABELS = {"", "sil", "sp", "spn"}


def intervals_to_dict(intervals, prefix):
    """
    Converts a parsed list of intervals from PRAAT TextGrid
    to a learning-friendly array

    Arguments
    ---------
    intervals: list
        A list of raw TextGrid intervals, as returned by
        TextGrid.interval_tier_to_array
    prefix: str
        the prefix to add

    Returns
    -------
    result: dict
        A dictionary of the form
            {
                "{prefix}": <list of labels>,
                "{prefix}_start": <list of begin values>,
                "{prefix}_end": <list of end values>,
                "{prefix}_count: <number of intervals>
            }

    """
    # Remove meaningless labels
    intervals_clean = [
        interval
        for interval in intervals
        if interval["label"] not in INTERVAL_EMPTY_LABELS
    ]
    result = {
        f"{prefix}{suffix}": [interval[key] for interval in intervals_clean]
        for key, suffix in INTERVAL_MAP
    }
    # This will map space labels to a single one
    result[f"{prefix}_count"] = len(intervals_clean)
    return result


RE_STRESS_MARK = re.compile(r"\d$")


def remove_stress_marks(phn):
    """Removes stress marks from a phoneme annotation

    Arguments
    ---------
    phn: list
        a list of phoneme annotations with or without stress marks

    Returns
    -------
    result: list
        a list of phoneme annotations without stress marks
    """
    return [RE_STRESS_MARK.sub("", item) for item in phn]


INLINE_FEATURES = ["audio_ssl_len", "wrd", "phn", "phn_stress"]


def prepare_features(
    data, data_folder, save_path, features, context, options=None, device="cpu"
):
    """Performs feature extraction

    Arguments
    ---------
    data: dict
        A preprocessed dataset
    data_folder : str | path-like
        The path where the dataset is stored
    save_path : str | path-like
        The path where the features will be saved
    features: list
        The list of feature extractions to be performed
    context : dict
        The context, providing auxiliary elements for
        feature extractions, such as pretrained models
    options : dict
        Feature extraction options
    device : str | torch.device
        The device to use for feature extraction
    """
    dataset = DynamicItemDataset(data)
    feature_extractor = FeatureExtractor(
        save_path=save_path,
        src_keys=["sig", "wav"],
        id_key="uttid",
        dataloader_opts=options.get("dataloader_opts", {}),
        device=device,
    )

    token_model_kwargs = options.get("token_model_kwargs", {})
    ssl_layers = options.get("ssl_model_layers")
    alignments_folder = options.get("data_folder_alignments", data_folder)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal."""
        wav = wav.format(data_root=data_folder)
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    @sb.utils.data_pipeline.takes("sig")
    @sb.utils.data_pipeline.provides("sig_resampled")
    def resample_pipeline(sig):
        sig_data = resample(
            waveform=sig.data,
            orig_freq=options["sample_rate"],
            new_freq=options["model_sample_rate"],
        )
        return PaddedData(sig_data, sig.lengths)

    @sb.utils.data_pipeline.takes("sig_resampled")
    @sb.utils.data_pipeline.provides("audio_tokens", "audio_emb")
    def token_pipeline(sig):
        result = context.token_model(
            sig.data, sig.lengths, **token_model_kwargs
        )
        tokens, emb = result[:2]
        yield PaddedData(tokens, sig.lengths)
        yield PaddedData(emb, sig.lengths)

    @sb.utils.data_pipeline.takes("sig_resampled")
    @sb.utils.data_pipeline.provides("audio_ssl", "audio_ssl_len")
    def ssl_pipeline(sig):
        ssl_raw = context.ssl_model(sig.data, sig.lengths)
        ssl = ssl_raw[ssl_layers].permute(1, 2, 0, 3)
        yield PaddedData(ssl, sig.lengths)
        yield (sig.lengths * ssl.size(1)).tolist()

    @sb.utils.data_pipeline.takes("sig_resampled")
    @sb.utils.data_pipeline.provides("spk_emb")
    def spk_emb_pipeline(sig):
        mel_spec = context.spk_emb_model.mel_spectogram(audio=sig.data)
        return context.spk_emb_model.encode_mel_spectrogram_batch(
            mel_spec, sig.lengths
        )

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides(
        "has_alignments", "wrd", "phn", "phn_stress", "unk_count"
    )
    def alignments_pipeline(wav):
        alignment_items = [
            parse_alignments(
                get_alignment_path(data_folder, alignments_folder, wav_item)
            )
            for wav_item in wav
        ]
        alignments = batchify(alignment_items)
        yield alignments["has_alignments"]
        yield alignments["wrd"]
        yield alignments["phn"]
        yield alignments["phn_stress"]
        yield alignments["unk_count"]

    dynamic_items = [
        resample_pipeline,
        token_pipeline,
        ssl_pipeline,
        spk_emb_pipeline,
        alignments_pipeline,
    ]

    dataset.add_dynamic_item(audio_pipeline)
    for dynamic_item in dynamic_items:
        feature_extractor.add_dynamic_item(dynamic_item)
    feature_keys = [key for key in features if key not in INLINE_FEATURES]
    inline_keys = [key for key in features if key in INLINE_FEATURES]
    feature_extractor.set_output_features(feature_keys, inline_keys=inline_keys)
    with torch.no_grad():
        feature_extractor.extract(dataset, data)


def batchify(values):
    keys = next(iter(values)).keys()
    return {key: [item[key] for item in values] for key in keys}


def get_context(extract_features, extract_features_opts, device):
    """
    Gets the context (pretrained models, etc) for feature extraction

    Arguments
    ---------
    extract_features : list
        A list of features to extract
        Available features:
        audio_tokens - raw tokens
        audio_emb - embeddings from the model
    extract_features_opts : dict
        Options for feature extraction
    device : str|torch.Device
        The device on which extraction will be run

    Returns
    -------
    context: SimpleNamespace
        The context object
    """
    context = {}
    if any(key in extract_features for key in ["audio_tokens", "audio_emb"]):
        context["token_model"] = extract_features_opts["token_model"].to(device)
    if "audio_ssl" in extract_features:
        context["ssl_model"] = extract_features_opts["ssl_model"].to(device)
    if "spk_emb" in extract_features:
        context["spk_emb_model"] = extract_features_opts["spk_emb_model"](
            run_opts={"device": device}
        )
    return SimpleNamespace(**context)
