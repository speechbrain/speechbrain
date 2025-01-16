"""
This is a preparation script for the YODAS dataset for The LargeScaleASR Set.

Download instructions:
    1. https://huggingface.co/datasets/espnet/yodas
Reference: https://arxiv.org/abs/2406.00899


Author
-------
 * Titouan Parcollet, 2024
"""

import csv
import functools
import logging
import os
import shutil
from dataclasses import dataclass

import torchaudio
from datasets import Audio
from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.pretrained import EncoderClassifier
from speechbrain.utils.parallel import parallel_map
from speechbrain.utils.text_normalisation import TextNormaliser

normaliser = Normalizer(input_case="cased", lang="en")


logger = logging.getLogger(__name__)

HF_HUB = "espnet/yodas"
LOWER_DURATION_THRESHOLD_IN_S = 3.0
UPPER_DURATION_THRESHOLD_IN_S = 40
LOWER_WORDS_THRESHOLD = 3
SAMPLING_RATE = 16000


@dataclass
class TheLoquaciousRow:
    ID: str
    duration: float
    start: float
    wav: str
    spk_id: str
    sex: str
    text: str


def prepare_yodas(
    hf_download_folder,
    huggingface_folder,
    train_subsets,
    dev_test_subset,
):
    """Download the dataset and csv for YODAS.

    en000 and en001 only are used for The LargeScaleASR Set. This is arbitrary.
    However en000 and en001 are supposed to total 10k hours. After cleaning from
    this function, only 5k will be left. The filtering steps are as follows:

    1. Do a first filtering of the samples based on text and creates a csv out
    of it.
    2. The .csv is loaded as a SB dataset and language ID using a SB model is
    performed -- returning a large dict of boolean (english or not).
    3. The initial CSV file is further trimmed with the outcome of the language
    ID.

    Parameters
    ----------
    hf_download_folder : str
        The path where HF stored the dataset. Important, you must set the global
        env variable HF_HUB_CACHE to the same path as HuggingFace is primilarily
        using this to know where to store datasets.
    huggingface_folder : str
        The path to the folder where the CSV files will be saved.
    train_subsets : list
        Target subset. "en000" and "en001" for The LargeScaleASR Set.
        e.g. ["en000", "en001"]
    dev_test_subset : str
        Target subset for creating dev and test. Only 1.5 hours each will be
        taken (so 3 hours in total).

    """

    if not os.path.isdir(hf_download_folder):
        msg = "You must download the dataset with HuggingFace before starting "
        msg += "this recipe. Please check the HuggingFace hub of YODAS."
        raise ValueError(msg)

    if len(train_subsets) == 0:
        raise ImportError("At least one YODAS subset must be specified.")

    # Setting the save folder
    manifest_folder = os.path.join(huggingface_folder, "manifests")
    wav_folder = os.path.join(huggingface_folder, os.path.join("data", "yodas"))
    os.makedirs(wav_folder, exist_ok=True)

    save_csv_train = manifest_folder + "/yodas_train.csv"
    save_csv_dev = manifest_folder + "/yodas_dev.csv"
    save_csv_test = manifest_folder + "/yodas_test.csv"

    # check if the data is already prepared
    if os.path.isfile(save_csv_train):
        logger.info("Skipping train preparation, completed in previous run.")
    else:

        hf_dataset = load_and_concatenate_datasets(
            dev_test_subset,
            hf_download_folder,
        )

        logger.info(f"Preparing CSV of the YODAS in {save_csv_train}...")

        HF_create_csv(save_csv_train, hf_dataset[0], wav_folder)

    if os.path.isfile(save_csv_dev) and os.path.isfile(save_csv_test):
        logger.info("Skipping dev/test preparation, completed in previous run.")
    else:
        hf_dataset = load_and_concatenate_datasets(
            dev_test_subset, hf_download_folder
        )

        logger.info("Preparing dev test CSV of the YODAS...")

        HF_create_csv_dev_test(
            save_csv_dev, save_csv_test, hf_dataset[0], wav_folder
        )

    logger.info("Data preparation completed!")


def load_and_concatenate_datasets(subsets, hf_download_folder):
    """Load/download and concatenate all the specified subsets from YODAS. Multiple subsets cannot be loaded all at once with
    HuggingFace so this function makes it possible.

    Parameters
    ----------
    subsets : list
        Target subset. e.g. ["en000", "en001"]
    hf_download_folder : str
        The path where HF stored the dataset. Important, you must set the global
        env variable HF_HUB_CACHE to the same path as HuggingFace is primilarily
        using this to know where to store datasets.

    Returns
    -------
    List of HuggingFace dataset.
    """

    try:
        import datasets
        from datasets import concatenate_datasets, load_dataset
    except ImportError as error:
        raise ImportError(
            f"{str(error)}\n" f"HuggingFace datasets must be installed."
        )

    # Managing the download dir as HF can be capricious with this.
    if "HF_HUB_CACHE" in os.environ:
        hf_caching_dir = os.environ["HF_HUB_CACHE"]
    elif "HF_HOME" in os.environ:
        hf_caching_dir = os.environ["HF_HOME"]
    else:
        hf_caching_dir = os.environ["XDG_CACHE_HOME"]

    if hf_caching_dir != hf_download_folder:
        msg = "HuggingFace HF_HUB_CACHE or HF_HOME is not equal to the given"
        msg += " hf_download_folder. Make sure to set these variables properly."
        raise Exception(msg)

    logger.info("Loading dataset from: " + str(hf_caching_dir))

    import multiprocessing

    nproc = (
        multiprocessing.cpu_count() // 2 + 1
    )  # we don't want to use all cores

    # Setting no download mode for HuggingFace. Only cache.
    # We remove progress bars as they repeat for each DDP process.
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    datasets.disable_progress_bars()
    datasets_list = []
    for subset in subsets:
        hf_data = load_dataset(
            HF_HUB,
            name=subset,
            split=["train"],
            num_proc=nproc,
            cache_dir=hf_caching_dir,
        )
        datasets_list.append(
            hf_data[0].cast_column("audio", Audio(decode=False))
        )

    os.environ["HF_DATASETS_OFFLINE"] = "0"

    # Datasets need to be concatenated back.
    final_dataset = []
    if len(datasets_list) > 1:
        final_dataset.append(concatenate_datasets(datasets_list, split="train"))
    else:
        final_dataset.append(datasets_list[0])

    datasets.enable_progress_bars()

    return final_dataset


def HF_create_csv(
    csv_file,
    hf_dataset,
    save_folder,
):
    """
    Create a CSV file based on a HuggingFace dataset. This function is particularly long to execute on YODAS as it follows 3 steps:

    1. Do a first filtering of the samples based on text and creates a csv out
    of it.
    2. The .csv is loaded as a SB dataset and language ID using a SB model is
    performed -- returning a large dict of boolean (english or not).
    3. The initial CSV file is further trimmed with the outcome of the language
    ID.

    Parameters
    ----------
    csv_file : str
        The path to the CSV file to be created.
    hf_dataset : huggingface dataset,
        The huggingface dataset.
    save_folder : str
        Where the wav files will be stored.

    """

    logger.info("Step 1: Text filtering...")

    hf_dataset = hf_dataset.select_columns(["utt_id", "audio", "text"])

    total_duration = 0.0
    nb_samples = 0

    #
    # Step 1 first filtering based on text
    #
    text_normaliser = TextNormaliser()
    line_processor = functools.partial(
        HF_process_line_first_txt_filter, text_normaliser=text_normaliser
    )
    csv_file_tmp = csv_file + ".tmp"
    csv_file_tmp_2 = csv_file + "_lid.tmp"
    if not os.path.isfile(csv_file_tmp) or not os.path.isfile(csv_file_tmp_2):
        with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            header = ["ID", "duration", "start", "wav", "spk_id", "sex", "text"]
            csv_writer.writerow(header)

            for row in parallel_map(line_processor, hf_dataset):
                if row is None:
                    continue

                csv_writer.writerow(
                    [
                        row.ID,
                        str(row.duration),
                        str(row.start),
                        row.wav,
                        row.spk_id,
                        row.sex,
                        row.text,
                    ]
                )

                total_duration += row.duration
                nb_samples += 1

        logger.info(f"First filtering. Number of samples in: {nb_samples}")
        logger.info(f"Total duration: {round(total_duration / 3600, 2)} Hours")

    #
    # Step 2 language ID.
    #
    if not os.path.isfile(csv_file_tmp_2):
        perform_language_id(
            csv_file_tmp,
            csv_file_tmp_2,
            "en",
        )

    valid_corpus_lines = open(
        csv_file_tmp_2, "r", encoding="utf-8"
    ).readlines()[1:]

    #
    # Step 3 Final text normalization and copy of audio files and csv saving
    #
    logger.info("Step 3: Final filtering, audio copy and csv creation...")
    line_processor = functools.partial(
        process_line_copy_wav_and_last_filter, save_folder=save_folder
    )
    total_duration = 0.0
    nb_samples = 0
    csv_file_tmp_3 = csv_file + "_last.tmp"
    with open(csv_file_tmp_3, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header = ["ID", "duration", "start", "wav", "spk_id", "sex", "text"]
        csv_writer.writerow(header)

        for row in parallel_map(line_processor, valid_corpus_lines):
            if row is None:
                continue

            csv_writer.writerow(
                [
                    row.ID,
                    str(row.duration),
                    str(row.start),
                    row.wav,
                    row.spk_id,
                    row.sex,
                    row.text,
                ]
            )

            total_duration += row.duration
            nb_samples += 1

    logger.info(f"First filtering. Number of samples in: {nb_samples}")
    logger.info(f"Total duration: {round(total_duration / 3600, 2)} Hours")
    os.replace(csv_file_tmp_3, csv_file)
    os.remove(csv_file_tmp)


def HF_create_csv_dev_test(
    csv_file_dev,
    csv_file_test,
    hf_dataset,
    save_folder,
    duration_dev=3600,
    duration_test=7200,
):
    """
    Create a CSV file based on a HuggingFace dataset. This function is particularly long to execute on YODAS as it follows 3 steps:

    1. Do a first filtering of the samples based on text and creates a csv out
    of it.
    2. The .csv is loaded as a SB dataset and language ID using a SB model is
    performed -- returning a large dict of boolean (english or not).
    3. The initial CSV file is further trimmed with the outcome of the language
    ID.

    Parameters
    ----------
    csv_file_dev : str
        The path to the CSV file to be created.
    csv_file_test : str
        The path to the CSV file to be created.
    hf_dataset : huggingface dataset,
        The huggingface dataset.
    save_folder : str
        Where the wav files will be stored.
    duration_dev : int
        Duration to select from for the dev set.
    duration_test : int
        Duration to select from for the test set.

    """

    logger.info("Step 1: Text filtering...")

    hf_dataset = hf_dataset.select_columns(["utt_id", "audio", "text"])

    total_duration = 0.0
    nb_samples = 0

    #
    # Step 1 first filtering based on text
    #
    text_normaliser = TextNormaliser()
    line_processor = functools.partial(
        HF_process_line_first_txt_filter, text_normaliser=text_normaliser
    )
    csv_file_tmp = csv_file_dev + ".tmp"
    csv_file_tmp_2 = csv_file_dev + "_lid.tmp"
    if not os.path.isfile(csv_file_tmp):
        with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            header = ["ID", "duration", "start", "wav", "spk_id", "sex", "text"]
            csv_writer.writerow(header)

            for row in parallel_map(line_processor, hf_dataset):
                if row is None:
                    continue

                csv_writer.writerow(
                    [
                        row.ID,
                        str(row.duration),
                        str(row.start),
                        row.wav,
                        row.spk_id,
                        row.sex,
                        row.text,
                    ]
                )

                total_duration += row.duration
                nb_samples += 1

        logger.info(f"First filtering. Number of samples in: {nb_samples}")
        logger.info(f"Total duration: {round(total_duration / 3600, 2)} Hours")

    #
    # Step 2 language ID.
    #
    if not os.path.isfile(csv_file_tmp_2):
        perform_language_id(
            csv_file_tmp,
            csv_file_tmp_2,
            "en",
            random=True,
            stop_at=duration_dev + duration_test,
        )

    valid_corpus_lines = open(
        csv_file_tmp_2, "r", encoding="utf-8"
    ).readlines()[1:]

    #
    # Step 3 Final text normalization and copy of audio files and csv saving
    #
    logger.info("Step 3: Final filtering, audio copy and csv creation...")
    line_processor = functools.partial(
        process_line_copy_wav_and_last_filter, save_folder=save_folder
    )
    total_duration = 0.0
    nb_samples = 0
    csv_file_tmp_3 = csv_file_dev + "_last.tmp"
    with open(csv_file_tmp_3, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header = ["ID", "duration", "start", "wav", "spk_id", "sex", "text"]
        csv_writer.writerow(header)

        for row in parallel_map(line_processor, valid_corpus_lines):
            if row is None:
                continue

            csv_writer.writerow(
                [
                    row.ID,
                    str(row.duration),
                    str(row.start),
                    row.wav,
                    row.spk_id,
                    row.sex,
                    row.text,
                ]
            )

            total_duration += row.duration
            nb_samples += 1

    os.replace(csv_file_tmp_3, csv_file_dev)
    os.remove(csv_file_tmp)

    # Split dev csv into two files (dev and test) with s
    dev_content = open(csv_file_dev, mode="r", encoding="utf-8").readlines()

    dev_file = open(csv_file_dev, mode="w", encoding="utf-8")
    test_file = open(csv_file_test, mode="w", encoding="utf-8")
    header = dev_content[0]
    dev_file.write(header)
    test_file.write(header)

    dev_duration = 0.0
    test_duration = 0.0
    for line in dev_content[1:]:
        ID, duration, start, wav, spk_id, sex, text = line.split(",")
        if dev_duration + float(duration) <= duration_dev:
            dev_file.write(line)
            dev_duration += float(duration)
        else:
            test_file.write(line)
            test_duration += float(duration)

    logger.info(f"Total dev duration: {round(dev_duration / 3600, 2)} Hours")
    logger.info(f"Total test duration: {round(test_duration / 3600, 2)} Hours")

    dev_file.close()
    test_file.close()


def perform_language_id(
    csv_file_in,
    csv_file_out,
    true_if_lang="en",
    random=False,
    stop_at=-1,
):
    """
    Creates a SpeechBrain dataset out of the csv_file and performs batched language ID. Results are written back to the csv form only containing
    the samples that matched the language.

    Parameters
    ----------
    csv_file_in: str
        Path to the original csv file.
    csv_file_out: str
        Path to store the modified csv file with matching samples.
    true_if_lang: str, optional
        Not used as of now, but language to match.
    random: bool, optional
        Weither to load batches randomly. This is much slower but important for
        the dev and test were we only pick a small number of hours (coupled with stop_at).
    stop_at: int
        Total duration in second to stop at (useful for dev and test).

    Returns
    -------
        None
    """
    logger.info("Step 2: language ID...")

    # Load dataset and sort to minimise padding
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_file_in
    )

    # If random we remove short sentences as it's for the test/dev.
    if random:
        LOWER_DURATION_THRESHOLD_IN_S = 10
        batch_size = 16
    else:
        batch_size = 128

    train_data = train_data.filtered_sorted(
        sort_key="duration",
        key_min_value={"duration": LOWER_DURATION_THRESHOLD_IN_S},
    )

    @sb.utils.data_pipeline.takes(
        "id", "wav", "duration", "start", "spk_id", "sex", "text"
    )
    @sb.utils.data_pipeline.provides("sig", "csv_row")
    def audio_pipeline(id, wav, duration, start, spk_id, sex, text):
        sig = sb.dataio.dataio.read_audio(wav)
        yield sig
        csv_row = (
            str(id)
            + ","
            + str(duration)
            + ","
            + str(start)
            + ","
            + str(wav)
            + ","
            + str(spk_id)
            + ","
            + str(sex)
            + ","
            + str(text)
        )
        yield csv_row

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline)
    sb.dataio.dataset.set_output_keys(
        [train_data], ["id", "sig", "duration", "csv_row"]
    )
    train_loader = sb.dataio.dataloader.DataLoader(
        train_data,
        collate_fn=PaddedBatch,
        batch_size=batch_size,
        num_workers=16,
        shuffle=random,
    )

    # Define language ID model
    lid_model = EncoderClassifier.from_hparams(
        "speechbrain/lang-id-voxlingua107-ecapa",
        run_opts={
            "device": "cuda",
            "precision": "fp32",
            "data_parallel_backend": "True",
        },
    )

    nb_samples = 0
    total_duration = 0
    with open(csv_file_out, mode="w", newline="", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(
            ["ID", "duration", "start", "wav", "spk_id", "sex", "text"]
        )
        with tqdm(train_loader) as t:
            for batch in t:
                batch = batch.to("cuda")
                wavs, wav_lens = batch.sig
                out = lid_model.classify_batch(wavs, wav_lens)
                for i, sample in enumerate(out[3]):
                    if "English" in sample:
                        csv_writer.writerow(batch.csv_row[i].split(","))
                        nb_samples += 1
                        total_duration += batch.duration[i].item()
                if stop_at != -1 and total_duration >= stop_at:
                    break

    logger.info(f"Second filtering. Number of samples in: {nb_samples}")
    logger.info(f"Total duration: {round(total_duration / 3600, 2)} Hours")


def process_line_copy_wav_and_last_filter(row, save_folder):
    """
    Takes a csv line, copy the wav into save_folder and write in a csv back with a last filter if necessary.

    Parameters
    ----------
    row: str
        The csv line to be processed.
    save_folder: str
        Path to where the wav files will be stored.

    Returns
    -------
    TheLoquaciousRow
        A dataclass containing the information about the line.
    """
    id, duration, start, wav, spk_id, sex, text = row.split("\n")[0].split(",")

    if text:

        save_audio_path = os.path.join(save_folder, id)
        # Copy the file if not already existing.
        if not os.path.isfile(save_audio_path):
            shutil.copyfile(wav, save_audio_path)

        row = TheLoquaciousRow(
            ID=id,
            duration=float(duration),
            start=-1,
            wav=save_audio_path,
            spk_id=None,
            sex=None,
            text=text,
        )

        return row
    else:
        return None


def HF_process_line_first_txt_filter(row, text_normaliser):
    """
    Process the audio line and return the utterances for the given split.
    This is used to generate a first CSV file with text filters applied.
    YODAS is very noisy so multiple steps are necessary (first text, this function, then audio language).

    Parameters
    ----------
    row: dict
        The audio line to be processed.
    text_normaliser: speechbrain.utils.text_normalisation.TextNormaliser

    Returns
    -------
    TheLoquaciousRow
        A dataclass containing the information about the line.
    """

    info = torchaudio.info(row["audio"]["path"], backend="soundfile")

    # Should not happen with YODAS, but just in case...
    if info.sample_rate != SAMPLING_RATE:
        return None

    text_nemo = normaliser.normalize(str(row["text"]))
    text = text_normaliser.english_specific_preprocess(text_nemo)

    if text:
        audio_id = row["utt_id"] + ".wav"
        duration = info.num_frames / SAMPLING_RATE
        audio_path = row["audio"]["path"]

        if (
            duration < LOWER_DURATION_THRESHOLD_IN_S
            or duration > UPPER_DURATION_THRESHOLD_IN_S
        ):
            return None

        #  Way too uncertain pronunciation (time/multiply/etc).
        if "ASTERISK" in text:
            return None

        if text is None or len(text.split(" ")) < LOWER_WORDS_THRESHOLD:
            return None

        row = TheLoquaciousRow(
            ID=audio_id,
            duration=duration,
            start=-1,
            wav=audio_path,
            spk_id=None,
            sex=None,
            text=text,
        )

        return row
    else:
        return None
