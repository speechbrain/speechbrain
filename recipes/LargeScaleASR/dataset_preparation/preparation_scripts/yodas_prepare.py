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
class LargeScaleASRRow:
    ID: str
    duration: float
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
    save_csv_dev = manifest_folder + "/yodas_dev_cleaned.csv"
    save_csv_test = manifest_folder + "/yodas_test_cleaned.csv"

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
            header = ["ID", "duration", "wav", "spk_id", "sex", "text"]
            csv_writer.writerow(header)

            for row in parallel_map(line_processor, hf_dataset):
                if row is None:
                    continue

                csv_writer.writerow(
                    [
                        row.ID,
                        str(row.duration),
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

    logger.info("Last step: copying files")
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
        header = ["ID", "duration", "wav", "spk_id", "sex", "text"]
        csv_writer.writerow(header)

        for row in parallel_map(line_processor, valid_corpus_lines):
            if row is None:
                continue

            csv_writer.writerow(
                [
                    row.ID,
                    str(row.duration),
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
):
    """
    Dev and test for YODAS are based on manually checked sentences. The only thing necessary is to find the path of each element in the dictionary variable DICT_DEV and DICT_TEST (see end of this file).

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
    """

    logger.info("Step 1: Text filtering...")

    hf_dataset = hf_dataset.select_columns(["utt_id", "audio", "text"])

    total_duration = 0.0
    nb_samples = 0

    #
    # DEV
    #
    line_processor = functools.partial(HF_process_line_dev_test, DICT_DEV)
    csv_file_tmp = csv_file_dev + ".tmp"
    if not os.path.isfile(csv_file_tmp):
        with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            header = ["ID", "duration", "wav", "spk_id", "sex", "text"]
            csv_writer.writerow(header)

            for row in parallel_map(line_processor, hf_dataset):
                if row is None:
                    continue

                csv_writer.writerow(
                    [
                        row.ID,
                        str(row.duration),
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

    logger.info("Copy dev files...")
    line_processor = functools.partial(
        process_line_copy_wav_and_last_filter, save_folder=save_folder
    )

    valid_corpus_lines = open(csv_file_tmp, "r", encoding="utf-8").readlines()[
        1:
    ]
    csv_file_tmp_2 = csv_file_dev + "_last.tmp"
    with open(csv_file_tmp_2, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header = ["ID", "duration", "wav", "spk_id", "sex", "text"]
        csv_writer.writerow(header)

        for row in parallel_map(line_processor, valid_corpus_lines):
            if row is None:
                continue

            csv_writer.writerow(
                [
                    row.ID,
                    str(row.duration),
                    row.wav,
                    row.spk_id,
                    row.sex,
                    row.text,
                ]
            )

    os.replace(csv_file_tmp_2, csv_file_dev)
    os.remove(csv_file_tmp)

    #
    # TEST
    #
    line_processor = functools.partial(HF_process_line_dev_test, DICT_TEST)
    csv_file_tmp = csv_file_test + ".tmp"
    if not os.path.isfile(csv_file_tmp):
        with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            header = ["ID", "duration", "wav", "spk_id", "sex", "text"]
            csv_writer.writerow(header)

            for row in parallel_map(line_processor, hf_dataset):
                if row is None:
                    continue

                csv_writer.writerow(
                    [
                        row.ID,
                        str(row.duration),
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

    logger.info("Copy test files...")
    line_processor = functools.partial(
        process_line_copy_wav_and_last_filter, save_folder=save_folder
    )
    valid_corpus_lines = open(csv_file_tmp, "r", encoding="utf-8").readlines()[
        1:
    ]
    csv_file_tmp_2 = csv_file_test + "_last.tmp"
    with open(csv_file_tmp_2, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header = ["ID", "duration", "wav", "spk_id", "sex", "text"]
        csv_writer.writerow(header)

        for row in parallel_map(line_processor, valid_corpus_lines):
            if row is None:
                continue

            csv_writer.writerow(
                [
                    row.ID,
                    str(row.duration),
                    row.wav,
                    row.spk_id,
                    row.sex,
                    row.text,
                ]
            )

    os.replace(csv_file_tmp_2, csv_file_test)
    os.remove(csv_file_tmp)


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
        "id", "wav", "duration", "spk_id", "sex", "text"
    )
    @sb.utils.data_pipeline.provides("sig", "csv_row")
    def audio_pipeline(id, wav, duration, spk_id, sex, text):
        sig = sb.dataio.dataio.read_audio(wav)
        yield sig
        csv_row = (
            str(id)
            + ","
            + str(duration)
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
        csv_writer.writerow(["ID", "duration", "wav", "spk_id", "sex", "text"])
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
    LargeScaleASRRow
        A dataclass containing the information about the line.
    """
    id, duration, wav, spk_id, sex, text = row.split("\n")[0].split(",")

    if text:

        save_audio_path = os.path.join(save_folder, id)
        # Copy the file if not already existing.
        if not os.path.isfile(save_audio_path):
            shutil.copyfile(wav, save_audio_path)

        row = LargeScaleASRRow(
            ID=id,
            duration=float(duration),
            wav=save_audio_path,
            spk_id=None,
            sex=None,
            text=text,
        )

        return row
    else:
        return None


def HF_process_line_dev_test(row, clean_txt_dict):
    """
    Process the audio line and return the utterances for the given split.
    This function relies on DICT_DEV and DICT_TEST which contain normalised
    transcription (manually checked by the author of the dataset). This is because YODAS transcriptions are very noisy. These sentences come from
    en003.

    Parameters
    ----------
    row: dict
        The audio line to be processed.
    clean_txt_dict: dict
        One of DICT_DEV or DICT_TEST as defined below. It can be changed with
        whatever dict that contains the wav filename and the transcription. We just provide our own set, but users can manually transcribe more sentences if they wish.

    Returns
    -------
    LargeScaleASRRow
        A dataclass containing the information about the line.

    """

    audio_id = row["utt_id"] + ".wav"

    if audio_id not in clean_txt_dict:
        return None

    info = torchaudio.info(row["audio"]["path"], backend="soundfile")

    # Should not happen with YODAS, but just in case...
    if info.sample_rate != SAMPLING_RATE:
        return None

    text = clean_txt_dict[audio_id]

    audio_id = row["utt_id"] + ".wav"
    duration = info.num_frames / SAMPLING_RATE
    audio_path = row["audio"]["path"]

    row = LargeScaleASRRow(
        ID=audio_id,
        duration=duration,
        wav=audio_path,
        spk_id=None,
        sex=None,
        text=text,
    )

    return row


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
    LargeScaleASRRow
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

        row = LargeScaleASRRow(
            ID=audio_id,
            duration=duration,
            wav=audio_path,
            spk_id=None,
            sex=None,
            text=text,
        )

        return row
    else:
        return None


DICT_TEST = test = {
    "O8_CzVPBq1g-00047-00042334-00043424.wav": "I'VE I'VE EXPERIENCED AND I GREW UP IN LOS ANGELES AND I MOVED TO NEW YORK SO IT'S QUITE EXTREME AND",
    "J-hp6I6hdZI-00353-00157101-00158196.wav": "A LITTLE BUMP A LITTLE PUSH TO SUSTAIN YOURSELF MAKING COMICS I MEAN JUST OUT",
    "_0azrnupduM-00093-00097700-00099700.wav": "AND ALONG THE WAY WE WOULD LIKE TO TALK A BIT ABOUT HOW THIS PROCESS OF INTEGRATING ARCHIVAL FORENSICS CREATED AN IMPETUS FOR US TO UPDATE POLICIES AND PROCEDURES AND LOOK AT OTHER AREAS THAT ARE PERTINENT TO DIGITAL ARCHIVING FROM A FORENSICS ",
    "oVpyoAJOX4Q-00063-00037339-00038373.wav": "TEST HE WILL RECEIVE THE CROWN OF LIFE WHICH GOD HAS PROMISED TO THOSE WHO LOVE HIM",
    "FwONAfQHQKY-00027-00025104-00026112.wav": "THAT'S THAT LOWER BOUND OBJECT BUT THEN WE HAVE TWO COLUMNS HERE AND WE ONLY WANT TO WORK",
    "7L4GmXPjLCy-00106-00059610-00060620.wav": "THIS RED LINE THAT HAS BEEN CRUDELY DRAWN ON BY MS PAINT IS FOLLOWING SOME NON EXISTENT TREND AND IS UP AND DOWN AND THEN UP",
    "khIf260dJ5y-00246-00078488-00080183.wav": "NOW THERE'S A WAIT HERE FOR THE CONTEXT TO COME BACK ONLINE",
    "sCZcMd5R0fo-00026-00026106-00027268.wav": "AGAIN WE TAKE THE EXPONENT NINETEEN AND DIVIDE IT BY THE INDEX WHICH IS FOUR",
    "txeQSfXGy68-00082-00042841-00043895.wav": "HOWEVER IT'S A DAD WHO FILMED IT WITH HIS NINE YEAR OLD DAUGHTER LIKE SHE'S RUNNING THE CAMERA FOR HIM",
    "3hOHMDxGDSg-00077-00069771-00071598.wav": "NECK OF THE WOODS ON OBSERVE ABILITY AND WHAT THEY'RE WORKING ON THERE ARE A LOT OF TOPICS THAT ARE YOU KNOW ADJACENT SUCH AS EDGE NETWORKS AND HOW DO YOU YOU KNOW OBSERVE THEM AND REAL TIME USER AMOUNT AND MONITORING A BPM AND WORK THAT'S BEING DONE ON PROFILING",
    "NRfmbW4Ja3g-00105-00082158-00083160.wav": "MORNING GREAT LITTLE SPOT HERE DOWN BY THE RIVER IT WAS SUPER NICE LAST NIGHT QUIET AND WE REALLY",
    "spqY8VfMxQk-00009-00008504-00009632.wav": "SPIES HERE AND THERE ALL THROUGHOUT THE BOOKS THOSE PROMISES OF GOOD BLESSINGS FROM GOD ARE",
    "0DlB_eflFyo-00023-00017752-00018824.wav": "THERE SO LET'S DO A DIFFERENT BATTLE PLAN HERE SINCE THEY'RE ALL KIND OF BEING LAZY",
    "E6nJ_KFL180-00031-00019482-00020562.wav": "ALLOCATED IN ONE TO SIXTEEN PRBS OVER FOUR TO FOURTEEN SYMBOLS AS A SIDENOTE PUCCH FORMAT ZERO AND TWO USE FEW",
    "fevs-b2fuN8-00110-00114100-00115300.wav": "MENTIONED LIKE WE LOOKED AT THIS FEE FOR SERVICE MODEL AND IN FACT IN OUR IN OUR PLANNING PHASE WHEN WE WERE REALLY JUST THINKING ABOUT THE NETWORK AND AND BUILDING IT OUT KIND OF ON PAPER WE THOUGHT",
    "r1LABD3dMnE-00021-00027100-00028300.wav": "THE ATTACKERS FIRST SHOUTED SLOGANS AND SHOWED BLACK FLAGS AND LATER THRASHED HIM EVEN AS HE FELL ON THE GROUND AND HIS AIDES DID THEIR BEST TO PROTECT HIM",
    "7S3nwMnW1mQ-00006-00010100-00011400.wav": "THE DOCUMENTS ARE IN RESPONSE TO A COURT ORDER FROM A MAY FIFTH TWENTY FIFTEEN LAWSUIT FILED AGAINST THE STATE DEPARTMENT JUDICIAL WATCH INC US",
    "uqdRXLcYC94-00066-00041109-00042402.wav": "GO FAMILY LOVE THIS AND AND BECAUSE OLD QUEEN WHO SITS INSIDE THE CAGE",
    "iTsN4gYDQiI-00249-00226451-00227503.wav": "I CAN SAY ANYTHING PROFOUND REALLY BUT LIKE FOR ME WHAT WAS WELL UNFORTUNATELY FOR ME WHAT STARTED THE PROCESS OF LEARNING WAS BURNING OUT SO",
    "XZ1f89-ecUA-00096-00105611-00106689.wav": "COMMUNITIES AND WITH INCREASED ATTENTION TO DIVERSITY EQUITY INCLUSION THAT WILL CERTAINLY SHAPE THE WAY THAT WE WE",
    "IWS56NrfDP4-00145-00112535-00113583.wav": "EVEN AS BY THE LORD WHO IS THE SPIRIT HE REGULATES WHO ENTERS AND STAYS AND WHO DOESN'T",
    "7JBmsYyzc40-00251-00181596-00182790.wav": "RUMBLESTITCH TWO TWO ROCK TWO ROCK SEATS OF EVIL AND JUST TWO ROCK SOME STREET FIGHTER SEVERAL",
    "-Qydq5EtDOc-00035-00035607-00037029.wav": "YOU KNOW DO YOU THINK OKAY THAT'S DOABLE IF YOU THINK IT'S DOABLE PROBABLY YOU'LL DO IT BUT QUITE OFTEN YOU GET OVERWHELMED YOU FEEL LIKE OH THERE'S TOO MUCH THIS COURSE IS TOO DIFFICULT THIS IS GOING TO BE REALLY BORING IT'S GOING TO TAKE FOREVER",
    "FkQfXGqx-no-00032-00055060-00056424.wav": "LABELLED THE GRANDFATHER OF GOGGLEBOX BY ITS CREATOR BERNICOFF SADLY DIED ON SATURDAY AGED EIGHTY THREE AFTER A SHORT ILLNESS",
    "lzEsx1EMC1g-00007-00007128-00008616.wav": "MAKE SURE THAT THAT IS CORRECT BEFORE I SCORE THIS",
    "ZjvShf8A-3c-00166-00095104-00096808.wav": "BY THE WAY THE ICON HERE A GROUP OF SEALS IS CALLED A POD SO THAT'S WHERE WE GOT THE",
    "fOxaq-9aUb4-00530-00298050-00299170.wav": "CREATED AN OUTDOOR INTERACTIVE EXHIBIT",
    "3hOHMDxGDSg-00242-00222773-00223992.wav": "AS WELL AS TIGRAN AND THEN DEPTH SO WE MEET EVERY OTHER WEEK SO NOT THIS WEEK BUT WE WILL MEET AGAIN NEXT THURSDAY TO TALK ABOUT THE PROPOSAL FANTASTIC",
    "5Bc5Wktp2kI-00048-00035128-00036584.wav": "INTO CUSTOM WITHERS THE YOUTUBER WITHERS ARE SO STRONG THAT ONE OF THEM DEFEATED THE ENDER DRAGON AND",
    "9SjZfen0nQy-00223-00167016-00168024.wav": "BY THEN WE EVALUATED THE DAY PLOT MOST OF THE SAMPLES OR I THINK ALL OF THE SAMPLES FALL",
    "_0azrnupduM-00131-00138300-00139300.wav": "AND SO WHAT WE'D LIKE TO INTRODUCE YOU TO NOW IS WHAT WE'RE CURRENTLY WORKING ON AN END TO END DIGITAL PILOT THAT'S RUNNING FROM DECEMBER TWENTY TWENTY TWO TO SEPTEMBER TWENTY TWENTY THREE",
    "i2MbB4Fmc_A-00018-00020100-00021500.wav": "DECISION OR ACTION THAT MIGHT NOT TAKE PLACE IN THE SHORT TERM BUT MAY DEVELOP LATER ON AND WHILE THERE IS SOMETIMES A DEGREE OF UNCERTAINTY AROUND THE CLOSE OF AN INDIVIDUAL ASSESSMENT PROJECT MOST ASSESSMENTS ARE PART OF A LARGER AND ONGOING ASSESSMENT",
    "xXADsgcd-2c-00175-00129409-00132233.wav": "UM THEY'RE ABOUT TWENTY SEVEN DIFFERENT WAYS TO ACTUALLY ACCOMPLISH THIS WITH THE UNDERLYING LIBRARY AND THIS IS WHAT I MEAN BY I WANT TO HIDE SOME OF THE COMPLEXITY YOU HAVE TO MAKE SOME DECISIONS BUT THE MOST COMMON THING IN THE WORLD IS YOU'RE GOING TO PUT LIKE THERE WILL GO AT THE TOP OF THE PAGE AND THEN YOU'RE GOING TO PUT THE TITLE OF THE DOCUMENT OR YOU KNOW THE QUOTE FOR WHO YOU KNOW AND YOU'RE GOING TO HAVE THESE THINGS BUT YOU DON'T YOU KNOW THEY'RE GOING TO SHOW YOU WHAT THEY ALREADY HAVE RIGHT THIS IS WHAT ALWAYS HAPPENS IT'S LIKE OH",
    "6h7YN83tGo4-00020-00016848-00018056.wav": "UH UI THE ON SCREEN KEYBOARD THE THE STATUS OF OF THE PROJECT COULD BE EXPLAINED BY",
    "FwONAfQHQKY-00009-00007464-00008584.wav": "NINETY FIVE PERCENT PREDICTION INTERVAL AND SO WE EXPECT NINETY FIVE PERCENT OF OBSERVATIONS TO FALL WITHIN THAT RANGE",
    "iTsN4gYDQiI-00165-00158325-00159682.wav": "YEAH THANKS PATRICIA SO ONE OF THE THINGS THAT FOR ME REALLY BECAME OBVIOUS ONLY WHEN IS SORT OF THIS THE STRUCTURES STARTED SHIFTING THAT THERE WERE CERTAIN LESSONS THAT I COULD ONLY START I'M LEARNING AS A RESULT OF THAT",
    "_0azrnupduM-00236-00253800-00255400.wav": "ALSO BY IMAGING USB'S AT TIMES WE'VE BEEN MANAGED TO KIND OF CARVE TO TO RETRIEVE DELETED FILES THAT IN SOME CASES COULD BE NECESSARY SO THAT'S THE THAT'S GENERALLY THE APPROACH THAT WE'RE TAKING",
    "ROQ5k7sCN7I-00029-00029343-00030362.wav": "NEGATIVE TWELVE X IS LESS THAN OR EQUAL TO TWELVE DIVIDE BY A NEGATIVE TWELVE ON BOTH SIDES",
    "-0BVrzfoOmA-00003-00002016-00003032.wav": "NECK PAIN FOR MANY YEARS IT STARTED WITH AN INJURY I HAD WHEN I WAS SEVENTEEN I ACTUALLY JUST GOT UP OUT",
    "m9SKdfjeyDy-00003-00004013-00005267.wav": "A SHEEPISH KEIRA DRESSED IN PLUNGING RED BLOUSE AND DENIM SHORTS BENT DOWN TO RECOVER HER BELONGINGS AFTER THEY APPEARED TO BE TOO HEAVY FOR THE FLIMSY WHITE PAPER BAG",
    "6h7YN83tGo4-00108-00099016-00100056.wav": "UH MANJARO IS SHIPPING SOME NATIVE KERNEL FOR PINE PHONE AND THEY ARE TRYING TO SUPPORT SUPPORT UH",
    "W1PY-i2pV1E-00040-00039668-00040727.wav": "PARENTHESIS ONE COMMA SIX BRACKET SO THE INTERVAL IS FROM ONE NOT INCLUDING ONE TO",
    "4cgR6PgJ9Ju-00015-00010504-00011628.wav": "AHEAD DRUPALCON MUNICH WE ORGANIZE THE DRUPAL AUSTRIA ROADSHOW IN ORDER TO MAKE SURE LOTS OF AUSTRIAN ATTEND AND OUR COMMUNITIES GROW",
    "xIJG6z2D-44-00067-00056087-00057727.wav": "YOU HAVE TWO MAJOR TECHNIQUES ONE IS SLICE RECORDING THE OTHER ONE IS DISASSOCIATED",
    "xXADsgcd-2c-00143-00106963-00108817.wav": "I'VE GOT A TECHSSIDED THEY CAN EITHER BE A STYLE OR STRAIGHT I'M ACTUALLY GOING TO CHANGE THIS I'VE BEEN THROUGH ABOUT TWENTY SEVEN VERSIONS OF THIS DESIGN BECAUSE I KEEP ITERATING OVER IT EVENTUALLY I'LL QUIT AND RELEASE SOMETHING THAT I'LL LET YOU GUYS KNOW WHEN I DO THAT",
    "m2YoEAYvC9u-00028-00018784-00019784.wav": "DAM RIGHT BEHIND ME AND THIS IS THE FIRST STOP AFTER LAUNCHING WE'RE GOING TO CHECK SAUNA CAPE",
    "iTsN4gYDQiI-00463-00394026-00395292.wav": "IS INCREDIBLY IMPORTANT TO TO DO THIS KIND OF WORK SO ALSO IF YOU'RE IN A COMMUNITY WHERE YOU YOU HAVE THE OPPORTUNITY TO CREATE THOSE SPACES AS A PARTICIPANT OR THE MANAGER I THINK THAT'S",
    "f7RwlHD7Dtg-00307-00202038-00203046.wav": "THANK YOU THIS REVIVING ENERGY PRICES IS TO RECREATE AND PARENT YOURSELF",
    "yYHpUFmoCpA-00081-00054664-00056227.wav": "I JUST TURNED ON MY MICROPHONE SO HOPEFULLY THAT WILL SYNC UP AND I'M DOING A LOT OF THIS AND I DON'T WANNA",
    "mT7YJBluMFk-00018-00014382-00016054.wav": "IT CAN RUIN A MARRIAGE SO EASILY BRENDAN OPENED UP ON HOW HIS WIFE OF EIGHT YEARS FEELS ABOUT HIS APPEARANCE ON THE SHOW AND INSISTED THAT SHE TRUSTS HIM AFTER HE WAS FORCED TO DENY REPORTS LAST MONTH THAT HE WAS OVERLY CLOSE WITH STRICTLY CO STAR NADIYA BYCHKOVA",
    "yfLoti8wIQ4-00202-00141432-00142552.wav": "MANAGE STOCK SKU SO THE SKU WOULD BE COURSE NUMBER EIGHTY SEVEN LET'S DO THAT LET'S JUST CALL IT NUMBER EIGHTY SEVEN",
    "ATkvfokAgfo-00059-00054560-00056184.wav": "THAT'S HERE LIKE IT'S SO GOOD IT'S VERY SWEET I WILL SAY",
    "fevs-b2fuN8-00021-00020800-00021800.wav": "A CLOSED CAPTION TRANSCRIPT AVAILABLE AND PLEASE AVAIL YOURSELF OF THAT IF IT'S HELPFUL TO YOU I THINK",
    "l-HeX6jYiCI-00000-00000021-00001044.wav": "HI EVERYBODY THIS IS AGNES AND IT'S SUCCESS STORY TIME NOW TODAY THE STORY IS ABOUT HOW TO GET",
    "-Qydq5EtDOc-00202-00213591-00214596.wav": "SEE ANY RESULTS FROM IT BUT EVENTUALLY YOU WILL AND THEN MAYBE WHEN THE NEXT TERM STARTS FOR THOSE OF YOU WHO ARE GRADUATE STUDENTS AND JUST HAVE CONSTANT WORK",
    "Ihwr_fevmwo-00009-00005850-00007629.wav": "AND ALSO PERSONALLY LAST WEEK WE TALKED ABOUT PUTTING TOGETHER A GRANT TRACKING TYPE OF A WORKSHEET TO TRACK GRANT OPPORTUNITIES AND I'D LOVE TO GET THAT STARTED TODAY IF WE HAVE TIME IS THERE ANYTHING ELSE THAT PEOPLE WOULD LIKE TO TALK ABOUT",
    "A06AYpw_Ory-00007-00011600-00013700.wav": "THE FILM IS SET TO HIT THE SCREENS ON JULY THIRTEENTH TWO THOUSAND EIGHTEEN IT PRESENTS THE STORY OF INTERNATIONAL HOCKEY PLAYER SANDEEP SINGH WHO WAS PARALYZED AND USED A WHEELCHAIR FOR TWO YEARS AFTER AN ACCIDENTAL GUNSHOT INJURED HIM IN TWO THOUSAND SIX",
    "qLZvCXRZOFy-00096-00047800-00048800.wav": "JUST TALENT WHATEVER THAN THERE IS FOR THE ONE WHO HOARDS ALL THESE THINGS",
    "Uggy7AgxZwu-00020-00015301-00016310.wav": "TO DIVERT MUCH OF THE ENERGY THAT IS GOING TOWARDS YOUR THINKING BRAIN TOWARDS THAT",
    "3PYrtkzrMqu-00036-00031058-00032193.wav": "THIS IS JUST EQUAL TO NEGATIVE COSECANT X COTANGENT X MINUS THE DERIVATIVE OF COTANGENT",
    "SzuxfcBJn6g-00034-00043112-00044984.wav": "LET'S TRY THIS WE'RE GOING TO WE'RE GONNA AUTOMATE IT IN",
    "SvAel6mB0oM-00000-00000003-00001071.wav": "HOW TO WASH YOUR HANDS PROPERLY FROM FECES AND BACTERIA STOOL MANNER",
    "pEyd5AdCdKk-00514-00230136-00231172.wav": "THAT YOU KNOW KIND OF ADDS TO THE OTHER POSITIVEPROPERTIES OF HORNS IN TERMS OF THE UNLICENSEDWISP NETWORKS",
    "Uggy7AgxZwu-00066-00048495-00049502.wav": "THAT ACTUALLY IS QUIET WHEN YOU GO TO BED WHEN YOU DON'T HAVE TO",
    "IWS56NrfDP4-00039-00030839-00031888.wav": "HEBREWS CHAPTER ELEVEN LET'S READ VERSE FIVE BY FAITH ENOCH WAS WHAT TRANSLATED THAT HE SHOULD NOT SEE",
    "I9A8nInjDCk-00009-00005200-00006268.wav": "IF YOU'RE AN INTERPRETER WHO BELIEVES THAT PEOPLE SAY WHAT THEY MEAN TO SAY AND THAT IT'S NOT ALWAYS UP TO US TO FIGURE OUT WHY THEY SAY IT THAT WAY",
    "LdHusFnwlQ0-00015-00031986-00033986.wav": "THERE IS CIVIL SOCIETY FUNDING  FIFTY SIX THOUSAND US DOLLARS FROM FOUNDATION OPEN SOCIETY INSTITUTE AND NON FINANCIAL SUPPORT OF TWENTY FOUR THOUSAND US DOLLARS AS VOLUNTARY ENGAGEMENT BY THE LOCAL COMMUNITIES THE PROJECT EXECUTION WAS",
    "xXADsgcd-2c-00089-00068318-00070214.wav": "DOES EVERYBODY KIND OF UNDERSTAND THE OR THE PROBLEM BY TRYING TO SOLVE IT'S NOT EVERYBODY'S PROBLEM PROBABLY NOT EVEN MOST OF SHARP PEOPLE FROM BUT IT WAS MY PROBLEM AND I WANTED A SOLUTION SO THIS IS WHAT I CAME UP WITH YEAH I'M GOING TO SHOW YOU A LITTLE BIT OF CODE",
    "IWS56NrfDP4-00041-00032776-00033912.wav": "THIS TESTIMONY THAT HE PLEASED GOD A TESTIMONY IS SOMETHING YOU SAY AND BEFORE SAY BEFORE BEFORE HIS",
    "hDpFpa0M6YU-00303-00176865-00177994.wav": "NOW I WANT TO TALK A LITTLE BIT ABOUT THE FUTURE AND THEN I'LL TAKE SOME OF YOUR COMMENTS",
    "6h7YN83tGo4-00121-00110976-00112183.wav": "LIKE TO HAVE WORKING PHONE WE WOULD LIKE TO SEND TEXT MESSAGES LIKE THE SMS MESSAGES UH",
    "kNq2bP9uWUQ-00005-00004752-00005862.wav": "THE OEN'S COMMUNITY NORMS WHICH YOU CAN SEE ON OUR OEN COMMUNITY HUB WHICH IS GREAT A GREAT RESOURCE",
    "Ek3fHIq1O28-00013-00013060-00014294.wav": "MURRAY HANAN THE WOULD BE KILLER'S FORMER LAWYER SAID POLICE DECIDED NOT TO CHARGE THE YOUNG MAN WITH TREASON WHICH IN NINETEEN EIGHTY ONE CARRIED THE DEATH PENALTY BECAUSE THEY HAD RECEIVED AN ORDER FROM UP TOP",
    "fOxaq-9aUb4-00364-00203700-00204706.wav": "REPAIR IS A BIG ONE BUT OK",
    "fevs-b2fuN8-00125-00133000-00134200.wav": "IN VALUE FOR OUR ACTUAL CURATORS SO THAT WAS YOU KNOW A HUGE REALIZATION FOR US THAT REALLY MAD THAT THIS WASN'T NECESSARILY GOING TO BE THE BEST APPROACH FINALLY",
    "ZLNLj7eM27u-00000-00000456-00002016.wav": "HELLO THERE WELCOME TO MY CHANNEL MY NAME IS DOUG AND I'M BACK WITH ANOTHER FOUNTAIN PEN VIDEO UH",
    "xQ9DRNW1-5Q-00054-00027848-00029260.wav": "NOW THAT THE ARRAY IS BUILT THE LAST STEP IS TO FORMAT IT WITH THE DISK MANAGEMENT UTILITY",
    "-Qydq5EtDOc-00047-00050628-00052485.wav": "SO THE CYCLE NOT ONLY TENDS TO REPEAT BUT IT TENDS TO GET WORSE AND WORSE AND WORSE AS IT CYCLES AND SO THEN WHEN YOU FINALLY DO DO THE WORK IT'S IT'S LIKE WHOA THIS IS EVEN MORE OVERWHELMING AND YOU CAN BE MORE PANICKY AND DESPERATE FEELING",
    "-FReKYSxT74-00003-00002618-00003768.wav": "GONNA SAY NO LET'S GET THIS DUB LET'S DO IT I'M GOING TOWARDS THE BAG MAN",
    "yxFPnxqxqQu-00026-00036200-00037600.wav": "AND HOW MANY SUCH FOUNDATIONS WILL THERE BE UNFORTUNATELY WE HAVE NOT PUT THE FOUNDATION DESIGN PROJECT HERE THERE ARE PROBABLY AT LEAST FIFTY OF THEM HERE",
    "y62q2Roz78u-00740-00468128-00469560.wav": "BACK TO THEM IF YOU EVER NEED HELP CONGRATS YOU FINISHED THE FIRST MODULE OF THE COURSE",
    "lzEsx1EMC1g-00234-00220623-00221623.wav": "THERE WHICH WAS AMAZING OKAY OKAY OKAY THIS IS WHERE WE ARE BUT THAT'S DEFINITELY WHAT IT NEEDED",
    "-2V5jdMxRMo-00073-00062405-00063600.wav": "TIME SO HERE AGAIN WITH THE GREEN SOLUTION WE HAVE A STABLE SOLUTION BUT NOT SO ACCURATE",
    "idIrEq4X9HY-00013-00012445-00013497.wav": "I TOOK ONE TWO FIFTY FIVE GALLON DRUMS AND CUT IT IN HALF WE WELDED THE LID BACK ON AND",
    "4yY9E0fVdXM-00576-00334200-00335420.wav": "FOR HAVING DONE THE WORK SO DO YOU HAVE ANY RECOMMENDATIONS FOR THAT",
    "xIJG6z2D-44-00237-00192418-00194102.wav": "LIKE OR IT MAKE A CIRCUIT LIKE THIS SO THERE IS SEVERAL WAYS YOU CAN MAKE CIRCUITS AND",
    "_7YTZI0xnZ8-00175-00121944-00123088.wav": "AN EXAMPLE OF OF OF SOME SOME INVITATION TO GO TAKE A JOURNEY THEN WE'VE GOT THE EXPLORE TAB",
    "_ZnZUYLEQMA-00003-00003807-00005310.wav": "AH SORRY GUYS AH GONNA STOP IT I HATE NINTENDO LAND I H E I MEAN H A T E IT",
    "BEuS6KCFJtc-00122-00052685-00053744.wav": "IN HERE WORDPRESS WEB DESIGN SEE IF THAT GETS US ANYTHING ELSE NOW THERE'S TWO",
    "xXADsgcd-2c-00139-00102507-00104481.wav": "YOU KNOW THERE'S A LOT BUT LIKE THESE ARE THESE ARE THE KINDS OF THINGS RIGHT THAT YOU WORRY ABOUT WHEN YOU'RE FORMATTING A DOCUMENT RIGHT YOU WANT TO HAVE YOU WANNA BE ABLE TO TELL YOU KNOW THE SYSTEM WHETHER YOU WANT THE TEXT LEFT RIGHT CENTER JUSTIFIED EVERYBODY THAT THAT'S PRETTY STRAIGHTFORWARD RIGHT",
    "LdHusFnwlQ0-00009-00019940-00021943.wav": "THE NEWLY AMENDED ACT HAS COVERED ACCESSIBLE STANDARD IN LINE WITH CRPD ARTICLE NINE SIX THOUSAND SEVEN HUNDRED AND SIXTY FIVE PERSONS WITH DISABILITIES AND THEIR FAMILIES RECEIVED FIRST PHASE AMOUNT FOR THE BUILDING RECONSTRUCTION EIGHTY SEVEN PUBLIC",
    "-Qydq5EtDOc-00085-00089475-00090498.wav": "SO THE POINT OF THIS QUADRANT SYSTEM IS NOT TO BE JUDGY OF EVERY TIME YOU KNOW YOU DO SOMETHING PLEASURABLE IT IS YOU KNOW JUST SORT OF A FRAMEWORK FOR THINKING ABOUT THINGS",
    "GNLx0X8lMty-00101-00050298-00051402.wav": "AND FINALLY THE OCTAGON WHERE WE GET TEN EIGHTY ANOTHER C SHARP SUDDENLY GEOMETRY IS",
    "dV4EdMj52DI-00357-00197816-00198840.wav": "THAT SITUATION A FRIEND OF MINE LOVED STUFFING PACKAGES WITH BIG HEAVY MANUALS TO ADD A MORE PERCEIVED",
    "FkQfXGqx-no-00009-00015258-00016704.wav": "ONE KICKED THINGS OFF BY PRAISING ALWAYS JUNE SUCH A LOVELY MAN AND THE SORT OF RELATIONSHIP MOST CAN ONLY DREAM OF HAVING RIP LOVELY LEON",
    "0m6rrJzRG-E-00034-00044117-00046819.wav": "SO THIS IS HOW YOUR WORK WILL LOOK LIKE",
    "auDdyjzQ2bA-00130-00076074-00077401.wav": "NOW WE'RE GONNA HEAD TO MY CLOSET AND GET OUT MY OUTFIT FOR OUR TRIP WELCOME TO MY CLOSET",
    "shjJsUifGjA-00072-00056741-00059145.wav": "BETTER CLEAN UP THIS MESS AND GET STARTED AGAIN WHAT ARE YOU GONNA DO I THINK I CAN PICK UP THE TRAIL OF THE MEN WHO DID THIS",
    "fevs-b2fuN8-00024-00023700-00025400.wav": "HEIDI IMKER IS FROM THE UNIVERSITY OF ILLINOIS AT URBANA CHAMPAIGN I ALWAYS SEEM TO WANT TO SAY CHAMPAIGN URBANA AND LISA JOHNSTON IS FROM THE UNIVERSITY OF MINNESOTA THEY",
    "-PT2TCNtr3y-00051-00038952-00040016.wav": "EVIDENCE REMEMBER BRIAN STEVENSON'S BOOK JUST MERCY WHICH YOU READ FOR ORIENTATION",
    "B9oeT7SkMiU-00298-00204072-00205080.wav": "THAT YOU HAVE NETFLIX IN YOUR TV AND YOU ARE TEMPTED BY NETFLIX AND YOU NEED TO OVERCOME NETFLIX AND YOU",
    "xXADsgcd-2c-00227-00181135-00183031.wav": "LIKE IT'S THE ANIMAL THE DOG AND THE CAT AND ALL THAT STUFF IT'S LIKE HAVE YOU EVER WRITTEN ANYTHING LIKE THAT I MEAN SOME PEOPLE DO I MEAN IF YOU COULD WRITE A GAME YOU'RE GONNA HAVE LOTS OF YOU YOU MIGHT HAVE LOTS OF OBJECTS TO DO STUFF LIKE I TOTALLY GET THAT RIGHT I MEAN THAT MAKES PERFECT SENSE BUT IN THE BUSINESS WORLD",
    "SK5HEeyi14M-00075-00051199-00052408.wav": "FSK HAS LOST THEIR TEN PERCENT BUFF NOW GIVING YANG GUY A LITTLE BREATHING ROOM HEY GUY",
    "VhPdG7wMqNk-00073-00070600-00071900.wav": "EVEN BASIC QUALITATIVE RESEARCH TAKES TIME AND IF A TEAM IS NEW TO RESEARCH METHODOLOGIES AND PROCESSES SUCH AS GAINING IRB APPROVAL AND WORKING WITH THE OFFICE OF SPONSORED PROGRAMS ADDITIONAL TIME SHOULD BE BUILT INTO THE PROJECT",
    "-2V5jdMxRMo-00026-00019398-00020469.wav": "OVER Y IS EQUAL TO NEGATIVE A DT INTEGRATING BOTH SIDES FROM Y NOT TO Y AND FROM ZERO TO",
    "XZ1f89-ecUA-00006-00014190-00015465.wav": "OKAY WHY DON'T WE GO AHEAD AND GET STARTED THANK YOU FOR JOINING US FOR THIS LAST DAY OF THE FIRST WEEK OF THE CNI VIRTUAL FALL TWENTY TWENTY MEMBER MEETING",
    "-Qydq5EtDOc-00320-00336642-00337737.wav": "SO THAT'S THE PREVIOUS SLIDES DURING THAT PERIOD IMMEDIATELY BEFORE EXAMS THIS SLIDE IS ILLUSTRATING THAT EVEN THROUGHOUT THE TERM",
    "3nosgaDEatg-00061-00052120-00053316.wav": "ANOTHER VERY CONFUSING THING ABOUT THE IS THE SECOND TIME YOU TALK ABOUT THE SAME NOUN",
    "L8FaMU19dFu-00705-00445858-00446955.wav": "INSTEAD OF OH INSTEAD OF THE THE LOGIT AND THEN I SPECIFY MY DATA FRAME IN ORDER TO ADD THAT VALUE TO THE DATA FRAME",
    "EsP0yM0TK9c-00137-00132289-00133348.wav": "IS A VERY HIDDEN BELL",
    "7JBmsYyzc40-00387-00288966-00290190.wav": "SHOW YOU WHAT PLAYSTATION ONE GAMES RUN LIKE HERE AND WELL IT WAS LET ME FIND THAT THERE WE GO",
    "UhgkcpLrTbE-00096-00038856-00039900.wav": "THE BROKER ALSO OFFERS CFDS ON STOCKS COMMODITIES BONDS AND OTHERS INDICES ETFS SUCH AS EURO STOXX FIFTY",
    "0kDM9FM54m4-00024-00023205-00024498.wav": "WE CAN GET FROM OUR PHASE DIAGRAMS IS WHICH STATE IS MORE DENSE THAN THE OTHER SO TO FIGURE THAT OUT WE TAKE A POINT NEAR THE LINE SEPARATING THE PHASES THAT WE'RE LOOKING AT AND WE GO UP IN PRESSURE",
    "y62q2Roz78u-00345-00221616-00222784.wav": "MEET ME OVER IN THE NEXT VIDEO WHERE WE'LL TALK ABOUT INTERPRETED AND COMPILED LANGUAGES",
    "Aqu2m6oB1ec-00071-00049774-00051022.wav": "THE HARDWARE WALLET IS ACTUALLY MORE SECURE THAN THE SEED AND THE REASON FOR THAT IS BECAUSE THE HARDWARE WALLET MOST HARDWARE WALLETS STORE THE SEEDS IN",
    "SJV-LD4EC14-00279-00240244-00241332.wav": "CC STANDS FOR CREATIVE COMMONS AND THEY MAKE LOTS OF DIFFERENT TYPES OF LICENSES AND THE CC ZERO LICENSE IS THE MOST OPEN",
    "vKHDcopIhd0-00269-00104235-00105338.wav": "EVERY LOVING THOUGHT IS TRUE EVERYTHING ELSE IS AN APPEAL FOR HEALING AND HELP REGARDLESS OF THE FORM IT TAKES",
    "ejB39NMNzLY-00039-00030036-00031056.wav": "THE DETOX CURE FOR THE MOMENT WHICH IS ADDING ADDING FOR MY BODY IT'S CHALLENGING OF COURSE I'M",
    "HklTO86yfvI-00160-00080180-00081392.wav": "THIRTY TWO X",
    "-el4rVvWNEI-00003-00002232-00003232.wav": "LOOK AT THE FADING STARS THE MOON THE TREETOPS AT THE DAWNING LIGHT AS IT ARRIVES",
    "6h7YN83tGo4-00214-00191832-00193384.wav": "WHAT QUESTION WAS ALREADY ANSWERED UM",
    "uv_A3Gn6eDI-00008-00004995-00007242.wav": "SO IF WE MAKE IT ACROSS HERE",
    "Fvbz9kR7Dcg-00170-00122110-00123146.wav": "THE SIDE SEAMS THIS INCLUDED UNPICKING ALL MY TOPSTITCHING AND THE POCKETS",
    "Tbks2pJBbdo-00181-00135312-00136512.wav": "BY A COLON AND THEN SAYING IWC WORKFLOWS SO IN QUOTES SO THAT THE PASSING OF THIS",
    "H8HdPQHnx5y-00547-00242286-00243286.wav": "LIKEWISE WE HAVE SOME DEMOS THIS IS KIND OF WHAT I SHOWED",
    "_0azrnupduM-00177-00185700-00187200.wav": "OKAY THIS IS JUST THIS IS THE LAST SLIDE SO JUST VERY QUICKLY YOU KNOW WHAT WE ARE DOING AS WELL IS I'M CHALLENGING ESTABLISH ARCHIVAL PROCESSES SO AS LEO HAS MENTIONED BEFORE MENTIONED BEFORE YOU KNOW WE DO OPERATE IN A VERY KIND OF HISTORICALLY PAPER ENVIRONMENT",
    "L8FaMU19dFu-00738-00469274-00470299.wav": "USES THE PROPENSITY SCORE IT'S A FUNCTION OF THE PROPENSITY SCORE AND THE PROPENSITY SCORE IS JUST KIND OF ITS OWN THING CONSTANT THING",
    "OC2PoyDoTdy-00009-00012366-00013416.wav": "WHAT WOULD HAPPEN SHE WANTS US BACK SHE CAN'T SHE CAN'T JESUS CHRIST WOULD FILL THAT VOID IN THE NAME OF",
    "AMmQbDReEQk-00014-00010956-00012594.wav": "TO FIND FIXED POINTS WHAT WE NEED TO DO IS AS BEFORE PUTTING ALL DERIVATIVES EQUAL TO ZERO SO FOR TWO DIMENSIONAL SYSTEM WE NEED TO SOLVE DX OVER DT EQUALS TO ZERO AND DY OVER DT EQUALS TO ZERO",
    "idIrEq4X9HY-00009-00008496-00010022.wav": "REALLY I USED A BRAND NEW SHOVEL HANDLE FOR THE THE HANDLE AND SOME DECORATIVE",
    "SJV-LD4EC14-00331-00289244-00290544.wav": "I'M ADDING OUR LICENSE WE ADDED A NEW FILE OKAY SO YOU'VE SEEN THAT YOU CAN CLICK ON THE THE TOP SORT OF RIGHT WHERE IT SAYS ADD FILE AND YOU CAN",
    "Y5z0kSFlkMY-00032-00024272-00025336.wav": "SO I OPEN THE UPLOAD MANAGER HERE I GO TO CHOOSE REMOTE FILES AND NOW IN THIS DIALOG I'M SEEING",
    "SK5HEeyi14M-00123-00109673-00110747.wav": "ANY GUY WITH TWENTY FIVE SECONDS REMAINING IN THERE THIRTY PERCENT BUFF HAND GUY RAT ONE THOUSAND FOUR KILLS",
    "EsP0yM0TK9c-00145-00141405-00142691.wav": "HONESTLY IF I'M NOT GETTING ALL THOSE I PROBABLY SHOULDN'T WASTE THE AMMO",
    "otKu0tzQpwg-00060-00033795-00035169.wav": "YOU WE BPD HAVE EXPERIENCED THE STIGMA AND ALL OF THAT AND YOU KNOW I KNOW THERE'S A LOT OF PEOPLE WHO DON'T WANT ME TO TALK ABOUT THIS AND HERE'S THE THING LIKE WHEN IT COMES TO BORDERLINE PERSONALITY DISORDER AND I'M SOMEBODY WHO LOOKS AT THE",
    "FkQfXGqx-no-00044-00075984-00077662.wav": "LEON SAYS OH I THINK SO I BELIEVE IT AS TEARS WELL UP IN THE LOVING COUPLE'S EYES HE ADDS I'D JOIN YOU YOU SEE ALWAYS JUNE",
    "xIJG6z2D-44-00075-00064745-00066167.wav": "LIKE YOU MAKE SLICES LIKE THIS SO YOU ESSENTIALLY GET YOU CAN MAKE SLICES IN A DIFFERENT IN DIFFERENT",
    "kCBY4uqNAnk-00000-00000003-00001006.wav": "COMPUTER SCIENCE UNPLUGGED IS A FREE COLLECTION OF ACTIVITIES THAT EXPOSE",
    "EhdI72IN7LM-00235-00122622-00123680.wav": "I PRESS ESC AND NOTHING HAPPENS OR IN THE TEXT ONLY GETS PASTED IF I PRESS PASTE ANYWAY",
    "BxOp3887UXY-00070-00045800-00047181.wav": "GET IT ALL LOOMED BACK UP BUILT BACK UP AND LET'S SEE IF WE CAN FIX IT",
    "fevs-b2fuN8-00099-00104500-00105800.wav": "TERMS OF YOU KNOW WE SO WE ENGAGED A PANEL AND SO THAT INCLUDED A VARIETY OF ACADEMIC INSTITUTIONS OF DIFFERENT SCOPES AND SIZES AND THEN ALSO SOME DATA CURATION ORGANIZATIONS TO COMPLETELY OUTSIDE OF YOU KNOW ACADEMIC LIBRARIES TO TRY AND GET A BALANCE",
    "r9yzr5MZh9g-00093-00086586-00087600.wav": "TRY AND FIGURE OUT A WAY TO FILTER IT OUT OF A REAL EEG SIGNAL LET ME JUST STOP THIS FOR A SEC",
    "-Qydq5EtDOc-00167-00174414-00175896.wav": "IT SOUNDS LIKE MAYBE YOUR YOUR PROCRASTINATION I WANT LIKE LAST WEEK I THINK IT WAS I WENT TO THIS HUGE WORKSHOP ABOUT PROCRASTINATION WAS TWO AND A HALF HOURS LONG AND THE GUY WENT INTO A WHOLE BUNCH OF",
    "Kr5LECV3cXu-00745-00325855-00326920.wav": "THE INTERESTING CONTRAST BETWEEN COMMODITIES AND MONEY IS THAT COMMODITIES ENTER INTO CIRCULATION",
    "wjQIZLeYqpk-00000-00000072-00001592.wav": "BUT THEN THE PART THAT I FIND PERSONALLY MOST MOTIVATING IS THAT IT CREATES A SENSE OF ADVENTURE",
    "0QoohmfEEkk-00000-00000000-00001362.wav": "IN TODAY'S DISCUSSION WE WILL TRY TO FIND OUT WHY CHINA AND RUSSIA ARE CLOSER THAN EVER BEFORE",
    "8S9J3K0mEAc-00028-00018492-00019632.wav": "THERE HERE IT IS THICKEN YOU NEED TO SELECT THIS TO SEE THE TOOL CLICK HERE",
    "yJzm8nl_Cik-00016-00013200-00014700.wav": "WORKERS ARE BLASTING THE SITE TO REMOVE A STAGGERING TOTAL OF TWO HUNDRED AND TWENTY THOUSAND CUBIC METRES OF ROCK TO MAKE ROOM FOR THE HUNDRED AND FIFTY BY THREE HUNDRED METRE PLATFORM",
    "fevs-b2fuN8-00189-00200300-00201600.wav": "THAT IS THE ENTIRE ETHOS OF WHY WE'RE HERE SO THAT'S ONE THING TOO WHERE WE'VE REALLY KIND OF STRUGGLED AROUND YOU KNOW WE CAN DO THIS IN CERTAIN WAYS BUT THEN WHO DO WE LEAVE BEHIND OR HOW DO WE NOT HELP THE REST OF THE COMMUNITY SO THAT'S ONE",
    "ygAaAvhKuHE-00037-00022028-00023040.wav": "DESIGN IT SYNTHESIZE IT TEST IT START ALL OVER AGAIN SAME CYCLE UNTIL YOU FIND A COMPOUND THAT BIND OR FILL",
    "u4isvz18EJM-00094-00064444-00065710.wav": "WILL CLARIFY HMM THAT WAS GOOD THAT WAS GOOD YEAH SO IT'S ABOUT PRIORITIZING IT'S ABOUT OKAY SO I",
    "5pZcbmzezOU-00028-00014357-00015686.wav": "ELSEWHERE IN THE STATE MY OER ROAD CREW COMES WITH ME REGULARLY SO FAR AND THEN",
    "QOmo8c57uO8-00223-00078458-00080601.wav": "LIFE BEFORE DEATH STRENGTH BEFORE WEAKNESS READING BEFORE FINDING OUT HI JJ HI JJ HI JJ YOU KNOW ITS VERY COOL JJ THANK YOU FOR SHOWING ME THERE'S MY CHILD",
    "_0azrnupduM-00032-00029200-00030900.wav": "I WOULD BE HARD PRESSED TO ARTICULATE EXACTLY WHY THE SOURCE CODE FOR DAVID'S GOOEY BASED GUI CLICKED FOR ME WHERE OTHER METHODS OF ACCOMPLISHING THE SAME TASK HAD NOT BUT SOMETHING ABOUT THIS PARTICULAR TOOL AND CONFIGURATION HIT THAT PARTICULAR SWEET SPOT FOR BOTH SOFTWARE AND PROFESSIONAL DEVELOPMENT",
    "fevs-b2fuN8-00233-00245900-00247100.wav": " I THINK THAT'S EXACTLY RIGHT JUST TO BE EXPLICIT WE HAVE THOUGHT OF OTHER THINGS AND WE'VE YOU KNOW SORT OF PARED IT DOWN IN TERMS OF LIKE WELL WHAT IF THIS DIDN'T HAPPEN OR THIS AND PEOPLE PULL OUT AND SO WHAT ENDS UP HAPPENING IS THAT YOU HAVE A REDUCT",
    "Ci3-Qmv38Yy-00035-00020339-00021857.wav": "SO THIS TRUCK IS A BEAST THE BEST THING ABOUT THIS TRUCK IS THAT TWO WHEEL DRIVE WE DEFINITELY",
    "uv_A3Gn6eDI-00310-00294494-00297081.wav": "WHAT THE WHAT IS THAT I FOUGHT IT BEFORE ONE BEFORE SO THAT DID FORTY NINE DAMAGE",
    "m3rJs0Q8bPQ-00080-00073095-00074219.wav": "YOU'RE SCRIPTED UP AND GO THROUGH ALL OF YOUR SCENES SO THAT YOU HAVE YOUR TAT SO YOUR TEXT",
    "qC2XKbLbRk8-00131-00102028-00103189.wav": "PUTTING UP A BETTER FIGHT THIS TIME AROUND IT SEEMS TRYING TO KEEP IT CLOSE",
    "_0azrnupduM-00189-00198600-00200300.wav": "GITHUB YOU KNOW FINANCIAL COSTS MIGHT NOT BE THE ONLY FACTOR GIVEN SAY THEIR GITHUB'S CONTROVERSIAL COPILOT AI PROGRAM WHERE THEY SCRAPED OPEN CODE REPOSITORIES INTO A YOU KNOW AUTO GENERATING CODE PRODUCT SO",
    "i2MbB4Fmc_A-00071-00073900-00075200.wav": "TIME AND RESET TARGETS AS NEEDED CLEARLY ASSESSMENT RESULTS CAN BE INVALUABLE IN GAUGING WHAT TARGETS ARE ATTAINABLE AND CONTINUOUSLY UPDATING AND PROVIDING RATIONALES FOR STRETCHBOWLS",
    "L8FaMU19dFu-01121-00722561-00723663.wav": "THAT BODY OF LITERATURE IS REALLY REALLY INFORMATIVE THIS TYPE OF THING YEAH YEAH AND IT'S NOT AND IT'S ALSO YOU KNOW NOT AS OBVIOUSLY",
    "yR-hdnNRXdc-00044-00022744-00023984.wav": "KNOWLEDGE THE CORE CYCLE IS AN INTERPRETATION OF THE PDCA THE PLAN DO CHECK AND ADJUST CYCLE",
    "_F_NrNcJNFg-00004-00003150-00004264.wav": "BUT BEFORE THAT HAPPENED FOR THE LAST THREE MONTHS WE HAVE BEEN CONDUCTING A GRASSROOTS POLL OF OUR MEMBERS AND TENS OF THOUSANDS OF OUR MEMBERS VOTED IN THIS POLL",
    "XQHiBnIUzJY-00163-00119254-00120822.wav": "ALL THE NUMBERS FROM YOU RUN THREE SIX TWELVE TWENTY ONE AND THIRTY SIX X TWO YOU CAN READ OUT X THREE YOU HAVE TO CALCULATE BUT",
    "VhPdG7wMqNk-00026-00021900-00023000.wav": "NEXT SLIDE PLEASE SOURCE SPACE BEFORE THE SOURCE IS OUR UNDERGRADUATE RESEARCH CENTER AND IT WAS STACK SPACE BEFORE AFTER THE STACKS WERE REMOVED IT WAS A VERY",
    "iTsN4gYDQiI-00023-00030866-00032292.wav": "PLEASE EMAIL THE TURING WAY AT GMAIL DOT COM OR YOU CAN DIRECTLY REACH OUT TO ANNE OR MALVIKA SHARAN WHO IS THE CO LEAD OF THE TURING WAY YOU CAN EMAIL THEM TO THEIR PRIVATE EMAIL AND YOU CAN SEE THAT INFORMATION AGAIN ON THE ETHER PAD",
    "zPWTzX2pYjc-00034-00019200-00021009.wav": "FOR MORE RESOURCES ON CITING IN MLA STYLE SEE THE KIRKWOOD LIBRARY WEBSITE",
    "_NIgqn87tiy-00029-00026416-00028183.wav": "WITH AN ESTIMATION OF THE NUMBER OF SIMULTANEOUSLY AVAILABLE CHANNELS FOR THE EXPERIMENT",
    "uv_A3Gn6eDI-00228-00219752-00221181.wav": "GOTTA KEEP MOVING THIS IS NO ORDINARY BATTLE THIS IS A WORKOUT",
    "hDpFpa0M6YU-00015-00008907-00009990.wav": "THESE ARE DISTRIBUTED A LITTLE BIT MORE THAN ONE STUDY SECTION AND WE HAVE A HUNDRED AND SEVENTY FOUR STANDING",
    "EsP0yM0TK9c-00095-00089538-00090713.wav": "MAYBE SHOTGUN IS NOT THE RIGHT IDEA BUT",
    "TrCKlDGlwrk-00044-00028416-00029564.wav": "AND THIS IS THE PARKING GARAGE THIS AH THIS UNIT HAS AH TWO PARKING SPOTS AND THEY ARE",
    "7L4GmXPjLCy-00064-00035430-00036620.wav": "GO AND READ THROUGH WHAT THE UN'S IPCC IS TELLING EVERYONE WITH REGARDS TO THE SETTLED SCIENCE OF THE ONE POINT FIVE DEGREES CELSIUS OF",
    "O8_CzVPBq1g-00006-00004031-00005250.wav": "SHOT ON A ROOFTOP IN WILLIAMSBURG IN TWO THOUSAND AND FOUR OF A SUBJECT SHE GOES BY THE NAME OF KULLY PHOTOGRAPHY",
    "Kd3ZBjr29hy-00062-00067096-00068168.wav": "TO KNOW WHAT I HAVE TO DO WITH MY HAIR LITTLE DID I KNOW THAT I HAVE TO READ THE INSTRUCTIONS THAT",
    "ZLNLj7eM27u-00006-00006120-00007452.wav": "THAT I USE IN THAT IS J HERBIN KYANITE DO NEPAL IT'S A BEAUTIFUL TEAL BLUE TURQUOISE KIND OF BLUE THAT HAS",
    "plmlE78FQzu-00016-00016800-00018700.wav": "RF ID TAGS ARE GOING TO BE PLACED ON MANY OBJECTS TO KEEP RECORDS OF WHERE THEY ARE STOCK CHECKING MOVEMENT OF GOODS DRUGS BLOOD SAMPLES ETC CONVENTIONAL RF ID TAGGING WOULD BE THE SUCCESS RATE OF MONITORING A LARGE NUMBER OF RF ID TAGS CAN BE AS LOW AS SEVENTY TO SEVENTY",
    "SzuxfcBJn6g-00040-00050656-00052360.wav": "THAT I REALLY LIKE AND I WANT TO ADD IN THERE SO WE'RE GOING TO GET THAT GETTER DONE",
    "f7RwlHD7Dtg-00667-00434004-00435054.wav": "IS GOING ON FIRST SATCH OH YOU ARE ABOUT TO BE ROMANCE THIS YEAR DIVINE MASCULINE COMING INSIDE",
    "1QPFu-914pc-00114-00086536-00087600.wav": "AS TO LINE VARIATION WELL THIS IS A VERY STIFF CHINESE STEEL NIB AND SO YOU'RE NOT GOING TO GET",
    "-Qydq5EtDOc-00006-00007176-00008301.wav": "I KNOW THAT ELSEWHERE IN CANADA A LOT OF LANDS ANCESTRAL LANDS OF FIRST NATIONS AND OTHER INDIGENOUS PEOPLE WERE SUBJECT TO TREATY IN BRITISH COLUMBIA",
    "xXADsgcd-2c-00410-00337142-00338938.wav": "UM I REALLY LIKE THE IDEA OF HAVING REUSABLE STYLES WHICH THEY ALLOW YOU TO DO IN A VERY WORD PROCESSING SORT OF WAY YOU DON'T SEE THAT IN THE CODE I HAVE HERE BUT BASICALLY YOU CAN DEFINE A STYLE GIVE IT A NAME IF THEY REUSE IT ANYWHERE IN THE DOCUMENT UM",
    "xXADsgcd-2c-00350-00289514-00290766.wav": "LIKE JCJ HAS WRITTEN AND I THINK ACTUALLY UNDERLYING THAT X EIGHTY SIX INTERPRETER IS A LIST INTERPRETER I DON'T THINK IT I THINK THE COMPUTATION EXPRESSION BUILDER IS MOSTLY SYNTACTIC SUGAR",
    "hnQ9nrNz6y8-00130-00077824-00078856.wav": "IMPROVES LIKE THIS LIPIDEMIA INFLAMMATION IMPROVES SUPERFICIAL BLOOD FLOW AT THE VENOUS LEVEL UM",
    "nFgtCRh-ZaM-00005-00009340-00010348.wav": "SO I STUCK THE SEED I JUST LEFT IT IN THE SHOW IT'LL BREAK IT OUT STUCK IT IN",
    "qVSgC7x--2Y-00012-00015832-00016984.wav": "LET ME PUT A BOX AROUND THOSE TWO PROPERTIES AND NOW WE'LL TURN THIS INTO A DEFINITION",
    "j__SO3-9oE0-00350-00368456-00369784.wav": "NEVER BEAT ME NICE TRY BUT THEY'LL SEE I ACHIEVE EVERYTHING WHILE THEY STAY SALTY",
    "UHBmdAHtEW4-00075-00060216-00061288.wav": "MULTIPLE CHOICES IN OKAY AND SO THERE IS OUR DROP ZONE PERFECT OKAY SO THAT'S WHERE LEARN",
    "VhPdG7wMqNk-00063-00060000-00061000.wav": "WE ALSO NEED TO EXPLAIN HOW AND WHY OF THESE COMMUNITIES ESPECIALLY IF THERE ARE ONGOING CONCERNS AROUND OFFICE MEETING AND INSTRUCTIONAL SPACES",
    "Ihwr_fevmwo-00013-00011508-00012579.wav": "I'D BE UP FOR IT OKAY AND WHAT WOULD THAT WHAT WOULD THAT REPRESENT WOULD THAT BE WRITING REQUIREMENTS OR WOULD IT BE TALKING ABOUT IMPLEMENTATION APPROACH OR",
    "EsP0yM0TK9c-00016-00010775-00012056.wav": "MADE IT SOUND LIKE SHE KNEW WHERE I WAS AT THERE'S SOMETHING BEHIND ME RIGHT",
    "5bahxR6hBrg-00066-00033101-00034283.wav": "BURKE IT'S THREE NOTHING SPAIN YOU'RE KICKING IT WITH YOUR TOE NO NO YOU KICK",
    "3ZH-l7C3xry-00044-00024599-00025687.wav": "IT LOOKS FOR YOUR WEAKNESS IT LOOKS FOR WHERE YOU HAVE A DEFICIENCY MIND AND BODY IF IT",
    "xQ1V6LbJi8M-00102-00055824-00056832.wav": "I TOOK LIKE SIX HUNDRED PICTURES THE SETTINGS WERE ISO THREE THOUSAND TWO HUNDRED TEN SECONDS EXPOSURE LENS WIDE OPEN AT F THREE POINT FIVE",
    "hDpFpa0M6YU-00107-00061719-00062801.wav": "AND YOU COULD GET LOW EVEN IF YOU HAD MODERATE TO HIGH IMPORTANCE IF THERE WERE WERE MAJOR WEAKNESSES",
    "_0azrnupduM-00100-00106800-00108500.wav": "SO AND  WE HAVE CREATED A DIGITAL ARCHIVING WORKFLOW AND THAT'S THE LINK THERE IT'S ALSO IN THE CHAT AND SO IT'S AVAILABLE TO VIEW ONLINE ON COPTR'S COMMUNITY OWNED WORKFLOWS AND SO PLEASE TAKE A LOOK FEEL FREE TO USE IT AS INSPIRATION FOR YOUR OWN WORKFLOWS IF YOU'D LIKE",
    "OjZX62EXNZu-00069-00017087-00018584.wav": "IF I GOTTA CHOOSE ONE OR THE OTHER",
    "sCZcMd5R0fo-00019-00019600-00021258.wav": "SO IF WE WANT TO SIMPLIFY THE SQUARE ROOT WE WOULD WRITE IT AS A ROOT A AND HERE'S WHY",
    "iTsN4gYDQiI-00420-00355988-00357107.wav": "I'VE LEARNED THAT I JUST NEED I FEEL SO MUCH HAPPIER IN WHEN I WHEN I EXERCISE AND AND KEEP MY BUTT AND BODY MOVING",
    "Ek3fHIq1O28-00030-00028207-00029226.wav": "TWO YEARS LATER THE SAME TEENAGER ATTEMPTED TO OVERPOWER A GUARD AND ESCAPE FROM A PSYCHIATRIC WARD WHERE HE WAS BEING HELD IN ORDER TO MURDER PRINCE CHARLES WHO WAS",
    "IWS56NrfDP4-00067-00052408-00053584.wav": "YOU ARE WHICH MIGHT BE FINE AND GOOD TO SOME PLACE THAT'S BETTER GREATER HEALTHIER YOUNGER WEALTHIER",
    "SJV-LD4EC14-00299-00258223-00259240.wav": "HAVE THE SAME LICENSING AS YOU SO THEY BASICALLY HAVE TO ALSO HAVE IF YOU HAVE AN OPEN LICENSE THEY ALSO HAVE TO HAVE AN OPEN LICENSE ON THEIR WORK THAT SORT OF",
    "3hOHMDxGDSg-00052-00042105-00043137.wav": "YEAH I CAN I CAN DO THE QUICK ADMINISTRATIVE PARTS JUST WANTED TO GIVE AN UPDATE TO EVERYBODY AND HAVE SOME DISCUSSIONS HERE ARE WE HAD",
    "i2MbB4Fmc_A-00102-00106200-00107200.wav": "THAT CAN CHANGE THE ABILITY OF A LIBRARY OFFERING TO DELIVER ON A PARTICULAR OUTPUT OR EVEN AN OUTCOME THAT WAS NOT PREVIOUSLY ANTICIPATED AS WE KNOW FROM",
    "iTsN4gYDQiI-00480-00406032-00407451.wav": "VERY WARM FEELING WARM FEELING WARM AND RELAXED NOW AFTER THIS ONE HOUR OF FIRESIDE CHART WHICH REALLY FELT LIKE THIS IN THE BEGINNING I WAS A BIT A",
    "yJzm8nl_Cik-00014-00011400-00012500.wav": "A FLAT AREA AT THE SUMMIT FIRST HAS TO BE CREATED THAT IS BIG ENOUGH TO ACCOMMODATE THE E ELT AND ITS HUGE DOME",
    "ZYVx2DwjQLU-00042-00029615-00030655.wav": "SO THE THE GREATER THE LENGTH OF THE PENDULUM THE SLOWER IT'S GOING TO SWING BACK AND FORTH BUT",
    "I9A8nInjDCk-00024-00017953-00018991.wav": "SO VAGUE LANGUAGE IS A REALLY IMPORTANT PART OF EVERYDAY SPEECH OF WRITING OF COMMUNICATION OF SOCIAL INTERACTION",
    "iTsN4gYDQiI-00217-00200943-00201999.wav": "ACADEMIC INSTITUTIONS QUITE OFTEN YOU KNOW WHEN SOMEONE STRUGGLING THE SOLUTION IS TO SEND THEM ON A MINDFULNESS COURSE OR SOMETHING AND",
    "3hOHMDxGDSg-00084-00076323-00077907.wav": "KIND OF SUPPORT THAT TODAY WE SUPPORT EUROPE AND AND THE US OBVIOUSLY WITH THIS TIME ZONE AND THIS TIME MEETING MEETING TIME BUT WOULD LOVE TO KIND OF AT LEAST HAVE ONE MEETING ON THE APAC TIME ZONE WHICH MEANS ABOUT FOUR PM YOU KNOW",
    "FwONAfQHQKY-00022-00019848-00020864.wav": "APPROPRIATE RANGE FOR EACH MONTH AND WE CAN DO THAT BY CHECKING TO SEE IF EACH POINT MATCHES",
    "Tbks2pJBbdo-00595-00443800-00444872.wav": "COLON ENDING QUOTES MY IWC WORKFLOWS SEARCH TERM AND THIS TIME I'M INTERESTED IN GETTING THIS",
    "-Qydq5EtDOc-00132-00138231-00139476.wav": "IF FIVE MINUTES IS THE MOST YOU CAN DO THEN COMMIT TO FIVE MINUTES BEFORE YOU DO YOUR PROCRASTINATION ACTIVITY OKAY I'M EXHAUSTED I WANT TO TAKE A NAP WELL I'M GOING TO WORK FOR FIVE MINUTES AND THEN I'M GOING TO SEE IF I REALLY NEED IT NOW",
    "SJV-LD4EC14-00203-00173608-00174780.wav": "SO HERE FROM THIS LANDING PAGE HERE YOU CAN CLICK ON THE PENCIL OR HERE THIS IS THE START OF THE LIST OF FILES",
    "ATkvfokAgfo-00082-00075264-00076656.wav": "I MIGHT HAVE SOME POPCORN LATER JUST BECAUSE IT'S QUICK AND EASY I'M JUST IN A VERY LAZY MOOD",
    "O8_CzVPBq1g-00055-00050700-00051739.wav": "TO NPR AND THERE WAS A SEGMENT ON JUDY CHICAGO AND HOW HER WORK WAS RECEIVED THIRTY YEARS AGO AND",
    "0DlB_eflFyo-00045-00036280-00037528.wav": "HOPEFULLY COMING IN TODAY GOING FROM A T MOBILE INTERNET TO A AT AND T SIM CARD AND HOPEFULLY WE CAN",
    "uv_A3Gn6eDI-00169-00167329-00168616.wav": "I BUILT UP THESE MUSCLES BY MANY TRIPS TO THE GYM MY BRILLIANT BRAUN WILL SURELY LEAD",
    "SJV-LD4EC14-00167-00145700-00146760.wav": "SO HOW YOU DO THAT IS WE WE'RE GOING TO EDIT WE'RE GOING TO EDIT THE README FILE THAT WE CREATED IN OUR IN OUR REPOSITORY AND WE DO THAT USING THIS MARKDOWN SCRIPT",
    "0kDM9FM54m4-00040-00038121-00039357.wav": "AND NOTHING POINTS IN BOILING POINTS WE KNOW OUR TEMPERATURES SO WE TAKE THAT LINE ACROSS AND WHERE WE GET THAT IS THE TEMPERATURE THAT CORRESPONDS TO THAT MELTING POINT BOILING POINT",
    "-Qydq5EtDOc-00338-00358065-00359652.wav": "YEAH BREAK DOWN CALCULATOR YEAH DENISE YOU KNOW YOU AS A GRADUATE STUDENT GOOGLING A DISSERTATION CALCULATOR A THESIS CALCULATOR IS GOOD I'M I'M SO IF YOU I'M PRETTY SURE DOES NOT HAVE ONE BUT THERE ARE OTHER INSTITUTIONS TO DO",
    "uo9f9QJPUJ8-00013-00018316-00019344.wav": "SO THIS IS THE PRINTER LET ME MOVE THE FRONT TO THE CAMERA SO THAT YOU CAN SEE",
    "_0azrnupduM-00065-00074200-00075700.wav": "I'M AWARE THAT MANY OF THESE QUESTIONS ARE ETERNAL ANXIETIES FOR OPEN SOURCE PROJECTS AND I'M NOT ABOUT TO SOLVE THEM TODAY BUT AS A PART OF PREPARING FOR THIS LIGHTNING TALK I WAS READING UNCURLED WHICH IS A FREE ONLINE GUIDE TO OPEN SOURCE SOFTWARE DEVELOPMENT WRITTEN BY DANIEL STENBERG",
    "pEyd5AdCdKk-00326-00133340-00134359.wav": "YEAH SO THE TOP ROW SHOWS ALL SYMMETRICAL HORN SYMMETRICAL HORN SECTORS WITH THE GAIN RANGING FROM TEN TO EIGHTEEN POINT FIVE DBI",
    "iTsN4gYDQiI-00186-00179959-00181125.wav": "THAT KNOW WHO ELSE FROM THE PANEL WOULD LIKE TO GO NEXT MAYA DO YOU HAVE ANY INSIGHTS FROM MAYBE THE MASTERS YOU'RE DOING AT THE MOMENT OR ANY ANYTHING ELSE",
    "EQnDWPn44tU-00051-00033384-00035184.wav": "WELL DONE TRIXIE HE WHISPERED INTO THE NIGHT AIR YOU'VE ENGINEERED A MAGICAL LIGHT DISPLAY",
    "Tbks2pJBbdo-00388-00292512-00293536.wav": "AND AND THIS IS WHAT WE'LL SEE ONCE THIS IS FINISHED WE CAN ACTUALLY ALREADY START SETTING UP THAT",
    "fOxaq-9aUb4-00448-00258991-00260080.wav": "TO THE STARS ARTICLE ABOUT IN THIS CASE BOCA RATON'S STORYWALKS",
    "40J9HqjY7zg-00018-00013961-00015490.wav": "ALSO YOU CAN CUSTOMIZE YOUR CHART FOR EXAMPLE NOW I HAVE SELECTED POSITION STYLE AND CUSTOMIZED VALUE FORMAT OF MY CHART",
    "3hOHMDxGDSg-00178-00170205-00171672.wav": "THAT A DIFFERENT TEAM IS HAVING A PROBLEM WITH OR ON THE CLUSTER THAT I'M RUNNING SOME APP TEAM HAS TO PLAY TO A NAMESPACE AND THEY WANT TO YOU KNOW USE THIS OR YOU KNOW THEY WANT ME THE CLUSTER OPERATOR TO MAKE IT POSSIBLE FOR THEM TO USE THIS",
    "EhdI72IN7LM-00169-00086050-00087343.wav": "AND WE SEE THEN WHAT HAPPENS WHEN TEXT IS IS PASTED ON THE ON THE TERMINAL APPLICATION SO",
    "hDy39t60cD0-00192-00146192-00147240.wav": "THIS WAS PRETTY HUGE BECAUSE WE MY COLLEAGUES AND I LIKE IDENTIFIED THAT WE ARE ACTUALLY CONFRONTING WITH A LIKE THEY WERE INVESTIGATING",
    "tVotjSWqQOy-00015-00011038-00012084.wav": "PLANTS OF DIED MASSIVE AMOUNTS OF EROSION AN INCREDIBLY UNSTABLE CLIMATE AND YEAH FLASH FLOODS IN DESERT ENVIRONMENTS ON",
    "XZ1f89-ecUA-00090-00096420-00097797.wav": "COLUMBUS BEHIND A FENCE IN IT'S ACTUALLY A LOCAL STATUE THAT IS UNDER CONSIDERATION FOR EITHER CONTEXTUALIZING OR POSSIBLE REMOVAL OR SOMETHING LIKE THAT BUT",
    "y62q2Roz78u-00683-00430904-00432784.wav": "USAGE FOR A WHOLE SECOND WE'LL SAY THE MACHINE IS HEALTHY IF THE CPU USAGE IS LESS THAN SEVENTY FIVE PERCENT",
    "XZ1f89-ecUA-00175-00187569-00188808.wav": "NATHAN ASKS FIRSTLY COMMENTS IT'S GREAT IT'S GREAT TO SEE THESE MATERIALS BEING CAPTURED WITH SO MANY PARTICIPATING ORGANIZATIONS CAN YOU TALK ABOUT HOW YOU HANDLE THE CHALLENGES OF GOVERNANCE",
    "hnQ9nrNz6y8-00420-00271472-00272480.wav": "HOUR IT HAS A A MUCH GREATER IMPACT ON LONGEVITY LIKE LITERALLY LIKE FORTY GREATER LONGEVITY VERSUS",
    "NuC2EaptQpM-00057-00039712-00041200.wav": "WHETHER IT IS BUDDHA WHOEVER WE FOLLOW WHO SHOWS US MASTERY IN THIS WORLD HOW TO LIVE IN MASTERY",
    "iTsN4gYDQiI-00135-00135904-00136975.wav": "WRITING SO MANY PAYPAL SOMETIMES YOU HAVE TO LOOK THROUGH AND AND AND SEE WHAT IS IN IT WHAT WHERE IS IT GOING AND I ONE OF THE THINGS I DID WAS TO SET UP",
    "uv_A3Gn6eDI-00024-00022176-00023492.wav": "WHAT THE FUCK I DON'T REALLY HAVE ANY GOOD POKEMON FOR WATER I GUESS ELECTRIC BUT",
    "SJV-LD4EC14-00104-00100276-00101656.wav": "AND IT'S GIVEN IT THE NAME OF THE OF THE REPOSITORY SO WHAT I WANT YOU TO DO IS DO EXACTLY THAT NOW I WANT YOU JUST TO CREATE YOUR FIRST REPOSITORY AND THEN WE'LL GO OVER STARTING TO EDIT IT",
    "fevs-b2fuN8-00112-00116800-00118600.wav": "INSTITUTIONS SO WE THOUGHT MAYBE WE SHOULD REALLY TEST OUT THIS THIS FEE FOR SERVICE APPROACH WHERE PERHAPS WE CAN PACKAGE UP CURATION AND AND REALLY GET END USERS TO TO PAY FOR THAT AND AND HELP US SUSTAIN THE DATA CURATION NETWORK OF COURSE WHEN WE START",
    "1ChRbK9S8Wk-00029-00029505-00031149.wav": "IT'S GOING TO GIVE ME ONE MINUS TWO THREE FOUR SIX IT SAYS TO ROUND TO AT LEAST TWO DECIMALS",
    "Aqu2m6oB1ec-00033-00020228-00021258.wav": "THINK OF IT LIKE CASH IF A BANK ROBBER GOES AND STEALS CASH FROM THE BANK THEY WILL THEN GO TO A HUNDRED GAS STATIONS WHERE THEY'RE GOING TO SPEND THAT CASH",
    "EsP0yM0TK9c-00045-00041043-00042587.wav": "LIKE MOTHER AND DAUGHTERS I'VE ALREADY DECIDED THEIR NAMES BELLA DANIELA AND CASSANDRA",
    "5y2dBx7tASI-00115-00067674-00068718.wav": "BUT WE HAVE POST AND POST TWO WHAT WHY HOW WHY NOT CALL FIVE POINT ONE TO FIVE POINT THREE LIGHTWARDEN THE IRONY OF",
    "-Qydq5EtDOc-00169-00176547-00177972.wav": "AND ONE OF THEM IS PERFECTIONISM AND YEAH SO IF IF LIKE WHAT IS LEADING YOU TO PROCRASTINATORS YOU'RE THINKING IT'S I'VE GOT TO MAKE IT PERFECT I JUST GOT TO POLISH IT UP A LITTLE MORE OH IT'S SO OVERWHELMING BECAUSE I'M TRYING TO GET AN A PLUS",
    "qC2XKbLbRk8-00097-00068237-00069755.wav": "ACTUALLY TRYING TO GET SOME TIME SIX SECONDS FOR A DIRECTION WINCES PANEL TWO",
    "i04eTdUwCUA-00006-00010253-00012159.wav": "SHE LOVES HIGH FASHION AND IS WILLING TO TAKE RISKS WHICH IS BRILLIANT I FELT LIKE A VERY LUCKY FELLA EVERYONE KNOWS NOTTS GIRLS HAVE A REPUTATION FOR BEING AMONG THE MOST BEAUTIFUL IN THE COUNTRY",
    "-Qydq5EtDOc-00176-00183981-00186033.wav": "A LOT OF MEDITATION APPS MAYBE YOU NEED TO JUST LIKE TAKE A BRISK WALK IN THE COLD AIR OUTSIDE FOR YOU KNOW I MEAN WE'RE NOT TALKING A SUPER LONG WALK OR TALKING MAYBE A TEN MINUTE WALK MAYBE YOU EVEN WANT TO DO SOME AEROBIC EXERCISES FOR FIVE TO TEN MINUTES TO YOU KNOW GET SOME OF THE",
    "pOuiRIZ1eso-00056-00045416-00046824.wav": "DIVIDE IT PRECISELY IN HALF THE LENGTH OF THIS LINE SEGMENT",
    "UHBmdAHtEW4-00002-00001976-00003272.wav": "JUST ONE EXAMPLE OF THESE QUESTIONS UTILIZING THE INTERACTIVE BOOK FEATURE OR TOOL RATHER IN H FIVE P",
    "p4vr6m8YiqU-00013-00010514-00011634.wav": "FROM THE LOWEST LEVEL OF CREATURES TO AS YOU COME YOU WILL SEE THEY BECOME LESS AND LESS AVAILABLE",
    "XZ1f89-ecUA-00109-00116976-00118334.wav": "OHIO CITIES AND OTHER PLACES AND THE KNIGHTS OF COLUMBUS ETCETERA BUT YOU'LL NOTE IN THE TINY TEXT IN THE BOTTOM CORNER WE'VE GOT FACETS THAT",
    "sfu8nxhF9so-00070-00054000-00055112.wav": "NOT REALLY LIKE A HUMAN FEELING BEING NOT REALLY LIKE A A REAL BEING WITH YOUR OWN THOUGHTS",
    "O8_CzVPBq1g-00033-00027099-00028237.wav": "LACK OF MODEL A FAMILY MODEL FOR FOR THESE INDIVIDUALS FOR THESE FAMILIES SO IT REALLY BECAME QUITE",
    "-wP53qG0tco-00043-00026874-00027930.wav": "FEELINGS BEFORE COMING OUT I COULDN'T SPEAK ABOUT IT WITHOUT TOUCHING THE SUBJECT SO MAYBE I JUST",
    "x-hSfjB4x_y-00053-00038240-00039288.wav": "SHRINKAGE FROM RESPONSIBILITY OF THE POOR THE RICH ARE INSOLENT UNWILLING TO OBEY AND DESPOTIC THE POOR",
    "Qs7jV6Kt1yA-00037-00024122-00025148.wav": "I HAVE ALREADY DEPLOYED THE OTHER SMART CONTRACTS SO LET US COMPILE OUR PROGRAMMATICALLY CONSTRUCTED",
    "qgxSKdLY8Xo-00010-00014140-00015972.wav": "OH BOY FINISH HIM YOU GOT ONE MORE HE WENT INSIDE",
    "SJV-LD4EC14-00214-00183468-00184612.wav": "WE GO BACK WE CAN HAVE A LOOK AT THIS IF I PRESS THE EDIT WE CAN SEE IT SO THIS IS ME EDITING IT IN THE MARKDOWN SCRIPT",
    "_0azrnupduM-00022-00020900-00022400.wav": "I'LL BE HONEST MY PYTHON SKILLS JUST FAILED ME I'VE VERY RARELY HAD A REASON TO USE PYTHON IN MY DAY TO DAY WORK AND BUILDING A GRAPHICAL APPLICATION WITH PYTHON IS A VERY DIFFERENT KIND OF GOAL FROM THE MANY DATA PROCESSING AND AUTOMATION TASKS",
    "fevs-b2fuN8-00140-00147800-00149000.wav": "OTHER TIER YOU KNOW IS COMPLETELY UNTESTED AND AND THAT'S A TEAR THAT WE'RE HEARING PEOPLE ARE INTERESTED IN THAT YOU KNOW YOU'RE JUST GETTING STARTED WITH CURATION YOU MIGHT NOT EVEN HAVE A DATA CURATOR ON STAFF BUT",
    "UaRnQ8nPdR4-00201-00143679-00145055.wav": "OR MAYBE NOT OKAY MAYBE NOT COME ON COME ON GENIE",
    "d1aN-vUq8d0-00016-00007413-00009100.wav": "IS FILLED WITH AIR AT A RATE OF THREE CUBIC INCHES PER SECOND",
    "-Qydq5EtDOc-00211-00222821-00223890.wav": "AND THAT CAN GO A LONG WAY YOU KNOW EVEN IF YOU THEN PUT IT AWAY FOR A BIT THAT CAN GO A LONG WAY TOWARDS NOT HAVING THE ANXIETY AND OVERWHELM BUILD UP TO TOO MUCH AROUND THAT PROJECT",
    "yfLoti8wIQ4-00141-00096472-00097560.wav": "AND THEN THERE YOU GO SO NOW SUPER SOCIALIZER NO THAT'S FINE INVENTORY LINKED PRODUCTS UPSELLS",
    "i2MbB4Fmc_A-00076-00078600-00079900.wav": "ALLOCATIONS RESOURCES THAT MIGHT NEED TO BE REALLOCATED OR REDISTRIBUTED MIGHT INCLUDE FINANCIAL RESOURCES OF COURSE BUT MIGHT ALSO INCLUDE PERSONNEL TIME OR EFFORT IN THIS WAY",
    "lSnOuePFLzY-00012-00022968-00024684.wav": "WHEN ASKED IF THERE HAD BEEN ANY MOMENTS OF DANGER DURING FILMING WATKINS SAID IN LATER EPISODES WHEN YOU SEE THE SYSTEMA JAMES WAS GETTING PROPERLY HIT THERE WERE TIMES IN THE CUT I CAN SEE THAT WAS IN PAIN IT WASN'T ACTING",
    "DqpGHoSwBAk-00333-00216360-00217408.wav": "YOU KNOW SPICE YES DO YOU REMEMBER THREE WEEKS AGO I ASKED YOU A QUESTION I SEE HE DO YOU DO WE GET",
    "ATkvfokAgfo-00212-00195768-00197584.wav": "WEIRD THAT THEY JUST WENT BAD SO QUICKLY BUT YEAH THAT IS THE UPDATE ON MY FOOD RIGHT NOW",
    "fevs-b2fuN8-00118-00125500-00126500.wav": "YOU KNOW ONE SIZE FITS ALL APPROACH TO HOW WE DO CURATION AND THAT'S SOMETHING WE ACTUALLY HAVE TO TEACH IN OUR WORKSHOPS YOU KNOW WE TAKE A VERY PRAGMATIC APPROACH TO MAKE DATA A LITTLE BIT BETTER",
    "i2MbB4Fmc_A-00016-00017600-00019400.wav": "PROCEDURES SETTING TARGETS FOR GOALS AND INITIATIVES REALLOCATING RESOURCES SUPPORTING LIBRARY WORKERS GENERATING INNOVATION AND OR SUNSETTING SERVICES RESOURCES OR SPACES NO LONGER SERVING A PURPOSE TO MAKE WAYS FOR NEW OFFERINGS ",
    "fevs-b2fuN8-00205-00217600-00220000.wav": "THANK YOU LISA WONDERFUL TALK SO INTERESTING AND REALLY APPRECIATE YOUR YOUR YOUR HONEST DESCRIPTION OF THE CHALLENGES THAT YOU FACED AND AND HOW YOU WERE CALLED TO BRING CERTAIN KINDS OF SKILLS TO BEAR ON A THORNY PROBLEM WITH A SITUATION THAT",
    "iTsN4gYDQiI-00105-00112361-00113465.wav": "IF I GENERALIZE VERY BRIEFLY DO TEND TO FOCUS VERY MUCH ON OTHER PEOPLE AND I THINK THERE'S A WHOLE COMPONENT OF YOU KNOW TAKING CARE OF OURSELVES AS WELL",
    "-Qydq5EtDOc-00297-00313713-00314814.wav": "CRITICAL THINKING TAKES TIME YOU KNOW SO IF YOU'RE ASSIGNED AN ARGUMENTATIVE PAPER WHERE YOU HAVE TO YOU KNOW READ A BUNCH OF OTHER SOURCES AND SEE THE RELATIONSHIPS AND SEE YOU KNOW THE STRENGTHS AND WEAKNESSES AND WHICH",
    "r1LABD3dMnE-00020-00025500-00027000.wav": "I THOUGHT JHARKHAND WAS A PEACEFUL STATE BUT MY VIEWS HAVE CHANGED AFTER THIS INCIDENT HE ADDED VIDEOS OF THE ALLEGED ATTACK SHOW A LARGE CROWD BEATING HIM AND HIS SUPPORTERS",
    "-2V5jdMxRMo-00008-00005980-00007053.wav": "IS GROWING WITH EACH STEP AND THIS IS INEVITABLE AS WE HAVE ERRORS DUE TO THE APPROXIMATION",
    "-Qydq5EtDOc-00069-00072708-00073728.wav": "A LOT OF PEOPLE GET SUCKED INTO THAT AND THAT'S WHERE THEY'LL SPEND THEIR TIME OR YOUR BOSS CALLS AND SAYS YOU KNOW SOMEBODY CALLED IN SICK YOU KNOW PLEASE COME AND COVER THE SHIFT",
    "FkQfXGqx-no-00017-00028498-00030500.wav": "LEON'S WIDOW JUNE TOOK TO TO THANK VIEWERS FOR THEIR COUNTLESS MESSAGES OF SUPPORT SHE SAID THANK YOU FOLKS FOR ALL YOUR WONDERFUL MESSAGES LEON WOULD HAVE LOVED READING THEM AND HEARING HOW MUCH HE WAS LOVED",
    "plmlE78FQzu-00006-00007250-00008450.wav": "BY UNRAVELLING THE BUTTERFLIES WE DISCOVERED A WHOLE RAFT OF NEW METAMATERIAL TYPE STRUCTURE IF YOU STRUCTURE MATTER ON A FINE ENOUGH SCALE IT DOESN'T RESPOND IN THE SIMPLE WAY LIKE A BUCKET OF WATER OR",
    "hDpFpa0M6YU-00141-00083059-00084101.wav": "SO HERE THE PEAK IS IS STILL BETWEEN THIRTY AND FORTY BUT AT THIS POINT THE A QUARTER OF",
    "tYwvPd669H4-00056-00018828-00019853.wav": "AND THE ACT OF CHANGING SHAPE RECEPTOR CHANGES SHAPE",
    "vgrOHpUp_O4-00050-00041630-00042793.wav": "THE COATED GLASS ON THE PRINT BED KEEPS OBJECTS STICKING ON THE SURFACE WHILE PRINTING AND MAKES REMOVING THE PRINTS AN EASY THING BECAUSE THE GLASS PLATE CONTRACTS WHILE COOLING DOWN WHICH LOOSENES THE PLASTICS",
    "_s5BeE7H9sY-00006-00010923-00012428.wav": "METTE MARIT A FORMER WAITRESS MET HER FUTURE HUSBAND AT A MUSIC FESTIVAL IN THE NINETEEN NINETIES WHEN SHE WAS A SINGLE MOTHER AND MARRIED INTO THE ROYAL FAMILY IN TWO THOUSAND ONE",
    "AiO-SHDOG5u-00001-00001229-00002294.wav": "ZERO KNOWLEDGE PROOFS ARE EXACTLY WHAT THE NAME IMPLIES THEY ALLOW ME TO PROVE TO ANOTHER PARTY THAT I HAVE A CERTAIN PIECE OF KNOWLEDGE WITHOUT REVEALING WHAT THAT KNOWLEDGE IS",
    "iTsN4gYDQiI-00312-00274516-00276109.wav": "YEAH NO THANK YOU I MEAN I THINK YEAH IT CAN BE COUNTERINTUITIVE RIGHT LIKE BE SELFISH TO TAKE CARE OF OTHERS BUT I MEAN TO AN EXTENT YOU KNOW THAT THAT IS JUST WHAT NEEDS TO HAPPEN YOUR RESPONSIBILITY IS TO YOU KNOW KEEPING YOURSELF",
    "m3rJs0Q8bPQ-00094-00089630-00090756.wav": "YOU WOULD DO THE SAME FOR INSTAGRAM FOR INSTAGRAM YOU GO TO YOUR APPS YOU JUST CLICK IN",
    "m3rJs0Q8bPQ-00025-00022219-00023354.wav": "YOUTUBE A LANDSCAPE HERE YOU WILL NOTICE THESE VIDEOS HERE ARE TEMPLATES THAT ARE CREATED",
    "VhPdG7wMqNk-00039-00035900-00037300.wav": "SURVEYED STUDENT PARTICIPANTS AND RECENT ALUMS OF EACH OF THE THREE COMMUNITIES TO GAIN INSIGHT INTO THEIR PERCEPTION AND USE OF COMMUNITY SPACE AS WELL AS LIBRARY SPACES SERVICES AND RESOURCES TO BETTER UNDERSTAND THE IMPACT OF THE COMMUNITIES ON THE LIBRARIES FOR THE",
    "fevs-b2fuN8-00075-00081700-00082700.wav": "WE'LL TALK ABOUT SOME REALLY GREAT BENEFITS TO THAT THAT IT'S BEEN SO INTEGRAL TO THE ENTIRE PROCESS THAT WE'VE BEEN WE'VE BEEN WORKING THROUGH BUT ALSO A FEW THINGS THAT THAT MADE IT HARD SO",
    "NuC2EaptQpM-00018-00012056-00013072.wav": "OURSELVES THAT WE NEED TO CANCEL RELEASE LET GO AND THEN REMEMBER TO BREATHE SMILE AND LOVE",
    "_0azrnupduM-00054-00059900-00061700.wav": "I CAN ONLY SPEAK HERE FOR MY PERSONAL EXPERIENCE I'M COGNIZANT OF THE FACT THAT EVEN OTHERS PARTICULARLY NON CIS HETERO WHITE MEN AT MY WELL RESOURCED IVY LEAGUE WORKPLACE MAY NOT SHARE THIS EXPERIENCE BUT I DO CONSIDER MYSELF PERSONALLY LUCKY TO BE IN A WORK SITUATION THAT IS HIGHLY SUPPORTIVE OF PROFESSIONAL DEVELOPMENT",
    "fevs-b2fuN8-00208-00224100-00225500.wav": "THIS MAY BE A QUESTION FOR ANOTHER TALK BUT DO THE FACULTY OR RESEARCHERS HAVE PARALLEL CONCERNS TO THE GRANTS OFFICES ABOUT SHARING THEIR DATA WITH PEOPLE THAT THEY DON'T KNOW",
    "pNuyj0iR6gy-00008-00005462-00006620.wav": "IS LESS THAN SEVEN WE KNOW FORTY SEVEN IS LESS THAN SEVENTY FOUR SO WE ENTER FORTY SEVEN IS LESS THAN SEVENTY FOUR",
    "-Qydq5EtDOc-00036-00037134-00038184.wav": "I'M GOING TO FAIL NO MATTER WHAT I DO YOU KNOW SO IT'S THOUGHTS LIKE THIS WHERE YOU ASSESS THE TASK HAS BEEN KIND OF BEYOND YOUR CAPACITY THE UNDER CONTROL MAYBE",
    "AMmQbDReEQk-00023-00017943-00019044.wav": "FOR EXAMPLE HERE YOU CAN SEE THAT THE SYSTEM SHOW CYCLIC BEHAVIOUR AND THIS BEHAVIOUR IS FIXED BUT SYSTEM IS NOT AT A FIXED POINT",
    "fASx_786hQ8-00060-00056949-00058202.wav": "OKAY I MEAN I'LL GIVE YOU MY SWAT DUDE AND I'LL TAKE THE SHOT IF YOU",
    "-Qydq5EtDOc-00288-00302916-00304854.wav": "YOU KNOW BEING ABLE TO HAVE YOUR FULL BRAIN POWER DEVOTED TO IT BECAUSE YOU'RE RESTED AND BECAUSE YOU DON'T HAVE HALF OF YOUR BRAIN BEATING YOURSELF UP THAT'S A BIG ADVANTAGE WORKING ON SOMETHING OVER TIME YOU HAVE MORE OPPORTUNITIES TO SEEK HELP AND YOU MIGHT GET A FLASH OF INSIGHT",
    "OjZX62EXNZu-00004-00000744-00001792.wav": "BUT ANY EXCUSE TO GET SAFELY OUT OF THE HOUSE RIGHT NOW IS A WELCOME ONE",
    "EsP0yM0TK9c-00115-00108502-00110127.wav": "NICE ART SMASHES CABINETS",
    "yfLoti8wIQ4-00287-00213119-00214752.wav": "UPDATE OKAY SO IT'S NOT THERE OKAY SO NOW I'M CONNECTED CLICK",
    "xQ9DRNW1-5Q-00047-00023615-00024634.wav": "HPE SMART STORAGE ADMINISTRATOR WHICH CAN BE DOWNLOADED FROM THE OFFICIAL WEBSITE",
    "qC2XKbLbRk8-00085-00057601-00058954.wav": "THERE GOES THE HOLY ARTIFACT DEFENSIVE TOWER OF DEMIGODS DEMIGODS GETTING MORE",
    "m-Nsxr5PcYU-00313-00243216-00244264.wav": "CHECK AGAIN THE SNIP CHROMOSOMES IF I RUN THIS WE HAVE THREE ELEVEN X AND SIX SO IF I TRY TO CHECK THE MODE",
    "x_yvZ70dZEy-00016-00007358-00008502.wav": "AND THEN YOU HAVE MINUS LOG TO THE BASE THREE OF LET'S GO NINE",
    "Mz7Cp8U8Zs8-00043-00033223-00034273.wav": "I WOULD WANT AND BRING IT UP HERE NO THOSE ARE THE DARKER TONES OK SO BRING IT",
    "yfLoti8wIQ4-00237-00169128-00170984.wav": "I ALREADY HAVE THAT SO WHY IS IT NOT I HAVE IT INSTALLED MAYBE I DON'T HAVE IT INSTALLED",
    "xXADsgcd-2c-00432-00357223-00359020.wav": "BECAUSE MOST OF THE DOCUMENTS I HAD TO DO DEEP TABLES BUT ANYWAY SO PEOPLE HAVE STRUGGLED TO REPLACE ALL THAT FUNCTIONALITY IN A WAY THAT'S CROSS PLATFORM BUT THIS PARTICULAR VERSION OF THE LIBRARY THAT I'M USING HAS DONE THAT BY INCORPORATING",
    "NuC2EaptQpM-00214-00148040-00149176.wav": "THEY WENT IN AND RESOLVED SEVENTY FIVE THOUSANDS MALARIA CASES IN AFRICA WITH THE MMS WE ALSO USE",
    "Fvbz9kR7Dcg-00152-00106771-00108262.wav": "WAS A NEW DAY AND TIME FOR A CHANGE",
    "PRup4lwrk28-00003-00002532-00003550.wav": "OR WE CAN MOVE THE DECIMAL TWICE TO THE RIGHT",
    "ihFQL7FCRf8-00032-00015190-00016191.wav": "BESIDES WHOLESALERS WOULD PREFER TO USE ORIGINAL TASTE BECAUSE THEY COULD ADD THEIR OWN FLAVOURS",
    "EE5BS63YQ7A-00070-00042176-00043192.wav": "TO USE WITH THOSE BVMS THAT MIGHT ONLY SUPPORT POWER THAT IS AGAIN OVER THE ONE HUNDRED AND TEN TWENTY VOLTS THAT",
    "GyM7h_VUfS0-00006-00010007-00011781.wav": "SHE HAD KEPT TIGHT LIPPED ABOUT THE COUPLE'S BLOSSOMING RELATIONSHIP BUT SAID IN A JOINT ANNOUNCEMENT WITH THOMAS MARKLE WE ARE INCREDIBLY HAPPY FOR MEGHAN AND HARRY OUR DAUGHTER HAS ALWAYS BEEN A KIND AND LOVING PERSON",
    "Dr2uz8Uvcfo-00084-00052736-00053760.wav": "THAT IS GOING TO BE A MEMBER OF R N BY M AND IF I TAKE A TRANSPOSE A THAT WILL BE A MEMBER OF",
    "d7VPjs5hjJg-00035-00033808-00034952.wav": "ISN'T THIS CUTE SO I CAN PUT ALL OF THESE IN THERE YES OH MEDICAL EMERGENCY OH YES I DO NOT I DO NOT",
    "GyM7h_VUfS0-00001-00002538-00003868.wav": "THE COUPLE ARE SET TO TIE THE KNOT IN MAY NEXT YEAR AFTER KATE'S THIRD CHILD IS BORN MEGHAN'S MOTHER A YOGA AND SOCIAL WORKER SAYS SHE IS INCREDIBLY HAPPY",
    "iTsN4gYDQiI-00030-00039156-00041104.wav": "AROUND OPEN OPEN SIGNS OPEN RESEARCH BUT ALSO YEAH LEADERSHIP SKILLS CORE PART OF WE DO IS HOSTING A SIXTEEN WEEK LONG MENTORING PROGRAM WHERE PEOPLE CAN COME WITH THE PROJECTS ONE",
    "fevs-b2fuN8-00120-00128100-00129100.wav": "ALSO REALIZED AND THIS IS THROUGH A LOT OF OUR SATISFACTION SURVEYS THAT WE DID ON OUR OWN CURATORS WHO ARE DOING THIS WORK AS A PART OF OUR IMPLEMENTATION PHASE",
    "iYNbrH-SxLo-00041-00058067-00059282.wav": "THEY WOULD TURN YELLOW BECAUSE IT'S IT'S GETTING HOT IN HERE SUMMER MONTHS AND ALL THE GRASSES BURNED OUT",
    "m3rJs0Q8bPQ-00088-00081924-00083070.wav": "OR YOU CAN DO SOCIAL OR YOUR SOCIAL YOU CAN USE TWITTER OR YOU CAN USE INSTAGRAM HERE",
    "iTsN4gYDQiI-00096-00102386-00103629.wav": "BEING IN THOSE SITUATIONS AND THEN MOVING INTO OR ALSO TRYING TO CONSTRUCT AND TO MANAGE YOUR COMMUNITY AND TO TAKE CARE OF A COMMUNITY ONE OF THE THINGS THAT I'VE NOTICED IS THAT IT AMPLIFIED ALL THE",
    "hXIpu9hMs_u-00062-00038580-00039642.wav": "ALTERNATIVES ARE BUT THAT THAT PROCESS COMES YOU KNOW YOU KNOW YEARS YEARS LATER AFTER THATRIGOROUS PROCESS",
    "Aqu2m6oB1ec-00077-00054940-00056067.wav": "IT REMINDS ME OF THE OLD ANECDOTE OR STORY ABOUT A PEASANT AND A KING WHO PLACE A BET",
    "uv_A3Gn6eDI-00268-00261013-00262788.wav": "ALL RIGHT WHAT'S UP SEE YOU COMPLETED THE ESP EXERCISE CONGRATULATIONS NOW ON THE RIGHT",
    "kNq2bP9uWUQ-00217-00135852-00136854.wav": "LICENSED I OFTEN SHARE A I SHARE A SPREADSHEET WITH FACULTY THAT ALLOWS THEM TO SORT OF LIKE KEEP TRACK OF EVERYTHING",
    "xXADsgcd-2c-00416-00342943-00344444.wav": "BECAUSE THERE REALLY ARE AND A LOT OF A LOT OF DIFFERENT TASKS THERE'S MANY DIFFERENT WAYS TO APPROACH THE I WANT TWO INCHES OF BLAKE SPACE HERE RIGHT I JUST",
    "yxFPnxqxqQu-00024-00034000-00035300.wav": "NOT ALL OF THE CONCRETE BUT AS I HAVE MENTIONED IN A CONVERSATION WITH YOU BEFORE THE FIRST PART OF IT WILL BE POURED THE ENTIRE FOUNDATION WILL BE DIVIDED INTO THREE STAGES STAGE ONE FROM",
    "Tbks2pJBbdo-00380-00286320-00287456.wav": "E NINETY THREE TO D CHANGE IN THAT PROTEIN SO THIS IS THE INFORMATION THAT'S IN SUCH A VCF FILE AND THE NEXT",
    "i9HdESw-P0Y-00067-00062370-00063455.wav": "INTO YOUR MARRIAGE YOU ARE NOT A CANDIDATE OF A PRINCIPLE OR SO TODAY LET US PRAY FOR OUR MARRIAGE",
    "xXADsgcd-2c-00185-00143370-00144670.wav": "NOW THERE'S A WHOLE LOT MORE TO THIS IN IN THE REAL WORLD THE REAL VERSION THAT I HAVE BUT FOR THIS I WANTED TO KEEP IT SIMPLE ENOUGH SO THAT I COULD EXPLAIN EACH SECTION BUT",
    "-el4rVvWNEI-00008-00006072-00007520.wav": "ARE NOW IN THE PRESENT AND ETERNAL WITH YOU AND WITHIN YOU CARRYING YOU ON YOUR JOURNEY",
    "W1PY-i2pV1E-00020-00017511-00018544.wav": "IS ARE TRUE BETWEEN NEGATIVE SIX INCLUDING NEGATIVE SIX AND TO ONE NOT INCLUDING",
    "fevs-b2fuN8-00164-00174200-00175600.wav": "NOT WHAT I'VE BEEN TRAINED TO DO RIGHT YOU KNOW AND AND WE HAVE A LOT OF STRUCTURES THAT ARE UNIVERSITIES IN PLACE TO HELP US THROUGH THESE BUT THEY'RE REALLY NOT USED TO COLLABORATING WITH OUR PEERS LIKE THIS SO IT'S BEEN IT'S BEEN A STRUGGLE",
    "m3rJs0Q8bPQ-00098-00096988-00098510.wav": "THE WAY TO ADD A LINK ILL IS TO SIMPLY CLICK INTO THIS DOWNWARD ARROW SOCIAL",
    "xXADsgcd-2c-00178-00134558-00136404.wav": "YOU SEND IT TO DOUBLE AND THE MOST CONVENIENT WAY TO CREATE THAT WITH THIS UNDERLYING LIBRARY IS JUST TO ADD A PARAGRAPH SO THAT PARAGRAPH ACTUALLY HAS NO SPACE OF IT TO BECAUSE I DIDN'T ADD ANY TEXT OR ANYTHING TO IT BUT I SAY OH THERE'S A PARAGRAPH THERE AND THERE'S THIS MUCH SPACE AFTER IT",
    "ATkvfokAgfo-00286-00275496-00276584.wav": "BUT ANYWAYS I WAS GOING TO GET COFFEE THIS MORNING BUT LIKE ANYTHING THAT HAS CAFFEINE IN IT DOES",
    "K-ROB1hCESu-00811-00475888-00476888.wav": "I THINK SOME OF THE SOLUTIONS THAT ARE BEING PROPOSED HERE TOUCH ON",
    "d1aN-vUq8d0-00076-00042624-00044057.wav": "WE WANT TO KNOW HOW FAST IS THE BOTTOM MOVING ALONG",
    "fevs-b2fuN8-00169-00179900-00181200.wav": "WE'RE REALLY TRYING TO FIGURE OUT WHAT WE'RE DOING WHICH MEANS THAT WE'VE BEEN PIVOTING A LOT AND WE'VE ALSO BEEN THEN FORCED TO MAKE DECISIONS THAT MAY NOT BE THE RIGHT THING LONG TERM THEY MIGHT NEED TO BE EXPECTED THAT PROBABLY WILL HAVE TO EVOLVE",
    "d3xjDU-My_Q-00011-00016344-00017443.wav": "OKAY WELL YOU HAVE TO GO BACK TO THE WINDOW RIGHT THERE OKAY AND OPEN THE CURTAIN",
    "-FbjyIKzep8-00078-00050969-00051971.wav": "WELL I M GETTING ANOTHER PIECE OF TAPE IF YOU'RE DOING IF YOU'RE ADDING TAPE TO A LIGHT",
    "y_emZwTypfg-00025-00020964-00022086.wav": "SO NOW WE HAVE THE PROFILE HERE TO PUT A PICTURE THROUGH BOTH PAGE AND MAKE IT TWO MM",
    "XZ1f89-ecUA-00065-00072735-00074295.wav": "AND WITHOUT PERMISSIONS AND THEIR PARTNERSHIP THIS THIS THIS ARCHIVE COULDN'T EXIST AND YOU KNOW ONE THING THAT WE FOUND INTERESTING IS THAT OPENNESS HAS REALLY BEEN THE SELLING PART POINT TO THIS PARTNERS FOR WHOM A PRIMARY CONCERNS REALLY EXPANDING THEIR COMMUNITIES OF READERS",
    "mGcJ28GWm1y-00022-00025948-00027033.wav": "WHAT IF I HAD NEVER FOUND GOD A NINETEEN YEAR OLD SUPERSTAR WITH MANY FELONIES AND MANY CHARGES AND MANY CHANCES AND MANY PARDONS AND WHEN I GOT A CHANCE TO CHANGE YEAH",
    "VhPdG7wMqNk-00060-00056400-00057600.wav": "IT SHOULD THE COMMUNITY SHOULD MAKE SENSE THIS HELPS INTEGRATE THEM INTO THE CULTURE OF THE LIBRARIES AND CAN REDUCE PERCEPTIONS OF THEIR BEING TENANTS UNITS THAT COULD OR SHOULD BE ELSEWHERE BUT",
    "hDpFpa0M6YU-00753-00374970-00376789.wav": "WE WILL RECONVENE AT ONE FIFTEEN SO FIFTY FIVE MINUTES GO GET YOUR LUNCH AND THANK YOU FOR A GOOD MORNING",
    "3hOHMDxGDSg-00040-00035895-00037554.wav": "THERE COOL AS WELL AS THE THE LOGO WHICH HAS BEEN KIND OF SHOWING THE KIDS EVERY BUT WE SHALL WE WE HAVE A COUPLE OF FINAL TWO FINAL TRACES I THINK MATT RIGHT SO WE CAN TAKE A GARLAND FINISH THAT",
    "_0azrnupduM-00191-00202000-00203700.wav": "AS A SCRIPT BUT IN THE FORM IT WAS IMPORTANT TO ME TO PUT IT OUT THERE AS JUST SORT OF LIKE DOWNLOADABLE DOUBLE CLICK APPS FOR MAC OS WINDOWS ETCETERA AND IN ORDER TO DO THAT YOU NEED A PLACE TO HOST THOSE FILES SO THAT PEOPLE USE YOUR SERVERS THAT PEOPLE",
    "vMCu1MBdzNA-00146-00111904-00113096.wav": "ALL RIGHT NICE WE HAVE SOME OF THE RESULTS IN AND YEAH YOU ALL NAILED IT GREAT JOB",
    "q5JmbXiNXTE-00116-00051274-00052294.wav": "STATISTICS ACTUALLY ARE MUCH HIGHER IN RURAL PENNSYLVANIA AS COMPARED TO OUR URBAN URBAN COUNTERPARTS",
    "vp1y7BFTpey-00046-00031080-00032383.wav": "WHILE EATING BLEEDING GUMS OR TEETH SENSITIVITY IT IS IMPORTANT TO CONSULT WITH YOUR DENTIST",
    "ZMtcbA-aSK8-00011-00007781-00008832.wav": "EARLY TWO THOUSANDS BEACH COVER UP DRESS TUBE DRESS FROM DIESEL AND IT HAS LIKE THIS COOL BLACK PRINT ON IT",
    "SJV-LD4EC14-00056-00054640-00055716.wav": "OBVIOUSLY IT HAS THESE INTEGRATIONS THAT YOU CAN USE LIKE OLD VERSIONS OF MY WEBSITE SO I PUT DIFFERENT VERSIONS ONTO ZENODO TO ARCHIVE IT",
    "Fvbz9kR7Dcg-00000-00000057-00001652.wav": "HI EVERYONE I'M SARAH AND THIS IS BUDGETSEW WHERE WE CREATE STYLISH FASHIONABLE LOOKS",
    "d1aN-vUq8d0-00220-00110009-00112030.wav": "WATER IS DRAINING FROM A CONE SHAPED FUNNEL AT A RATE",
    "-Qydq5EtDOc-00054-00056891-00058596.wav": "SO THE IDEA BEHIND THIS IS THAT ANY ACTIVITY CAN BE CATEGORIZED IN ONE OF THESE FOUR QUADRANTS BASED ON HOW IMPORTANT IT IS AND THEN THE OTHER DIMENSION IS HOW URGENT IT IS YOU KNOW DOES IT HAVE TO BE DONE IMMEDIATELY",
    "i2MbB4Fmc_A-00001-00001700-00002800.wav": "THIS PRESENTATION IS PART OF A MODULE THAT FOCUSES ON REFLECTING COMMUNICATING AND ACTING ON THE RESULTS OF LIBRARY ASSESSMENT IT DESCRIBES POSSIBLE OUTCOMES OF THE OVERRAL ASSESSMENT",
    "idIrEq4X9HY-00001-00001271-00002466.wav": "BUILD I USED TWO FIFTY FIVE GALLON DRUMS THEY WERE USED FOR STORING CANOLA OIL AND",
    "ATkvfokAgfo-00252-00233704-00234776.wav": "SO SO I JUST GOT BACK AND IT IS TEN THIRTY I THINK FOR DINNER",
    "iTsN4gYDQiI-00021-00027141-00028552.wav": "SO WE USE THAT TO FACILITATE WRITTEN NOTE TAKING AND TO INVITE IDEAS FROM YOU WHO'VE JOINED TO LISTEN IN TODAY SO PLEASE FEEL FREE TO ADD QUESTIONS AND NOTES INTO THE PAD OR TO THE CHAT WHICHEVER YOU PREFER WE WE WILL FIND THEM DON'T WORRY",
    "y62q2Roz78u-00650-00402352-00403384.wav": "KEEP THESE THINGS IN MIND AND AUTOMATION CAN BE A VALUABLE ASSET IN YOUR TOOLBOX",
    "1mNa-aOTQQ4-00036-00039050-00040308.wav": "I'LL FLIP BACK AROUND THERE BACK IN THE MIDDLE COME ON DUDE WHERE Y'ALL AT",
    "zHHLc8crbO8-00003-00002586-00003661.wav": "HUGE ESPECIALLY WHEN YOU'RE LEARNING AS YOU GO REMEMBER IT IS ALWAYS BETTER TO HAVE ONE THING DONE THAN EIGHTY THINGS ALMOST DONE",
    "plmlE78FQzu-00011-00011950-00013250.wav": "IF FLAT SILVER IS PLACE INTO THE SCATTEROMETER ALL WE WOULD SEE IS REFLECTED GREEN LIGHT IF WE PATTERN THE SILVER SERVICE WITH A GRATING THEN WHAT WE SEE ON THE SCREEN IS SOME MISSING PORTIONS OF LIGHT",
    "i2MbB4Fmc_A-00030-00032100-00033300.wav": "THESE GOALS OR VALUES THAT BOTH UNDERPIN AND DRIVE ACTION AND THEN ASSURE THAT THERE'S ALIGNMENT FROM WHAT THE LIBRARY HOPES TO DO AND ACHIEVE THROUGH THESE ELEMENTS INCLUDING THE LIBRARIES",
    "-Qydq5EtDOc-00276-00292173-00293445.wav": "YOU KNOW MAYBE THAT IN ITSELF IS OVERWHELMING BECAUSE IT'S GOING TO TAKE YOU A WEEK SO YOU KNOW MAYBE YOU WANT TO BREAK THAT ONE DOWN FINER ON YOUR OWN BUT AT LEAST THIS TOOL WILL GET YOU STARTED ON THE PROCESS OF BREAKING DOWN",
    "SK5HEeyi14M-00077-00052991-00054014.wav": "FSK OVER HALFWAY BUT HERE COMES THE FLOOD INTERRUPTED BY THE WAY FSK IS PART",
    "UHBmdAHtEW4-00209-00161432-00162816.wav": "OKAY HERE WE GO SO FROM ZERO TO NINETY NINE PERCENT I'M GOING TO ACTUALLY TELL THEM TO REVIEW THE HANDOFF REPORT",
    "ZjvShf8A-3c-00204-00118176-00119183.wav": "LAYERS THAT GOT INSTALLED AND THEN BASICALLY IT ENDED UP CREATING MY IMAGE INSIDE OF IT",
    "txeQSfXGy68-00012-00005697-00007401.wav": "ME SO I'VE PUT THEM ALL TOGETHER AND I'M GOING TO TAKE YOU THROUGH MY EXPERIENCE AND WHAT I SAW IN EACH OF THESE VIDEOS I REALLY LOVE",
    "VDBjCrR1RPM-00019-00017592-00018612.wav": "BELVEDERE I'M THE LIQUOR STORE MY CAVALIER WELL THERE'S NO ONE LIKE ME ANYWHERE ELSE AROUND HERE",
    "nU032J1d8fY-00023-00006845-00007953.wav": "ALL RIGHT AND THEN FINALLY LET'S DO THOSE INTERCEPTS",
    "vobR23oit2y-00012-00011748-00012810.wav": "BUT IT'S A WHAT TECHNOLOGY WISE DOES IS IT MANIPULATES THE LOGARITHMS SO BASICALLY",
    "pMO0K5eFj8u-00079-00055088-00056104.wav": "THAT VERB CONJUGATION IT WAS VERY IMPORTANT TO BE TAUGHT RIGHT FROM RIGHT FROM WHEN THEY FIRST STARTED",
    "O8_CzVPBq1g-00049-00044329-00046058.wav": "REALLY OBVIOUS TO US THE IMPACT THAT WE WERE MAKING IN THESE ENVIRONMENTS AND HOW WE WERE BEING PERCEIVED",
    "iTsN4gYDQiI-00267-00243317-00244702.wav": "THIS IS ACTUALLY JUST THE THING THAT I READ ABOUT AND LIKE YOU KNOW YOU YOU THINK YOU THINK IN A I THINK YOU DON'T HAVE BURN OUT YOU YOU DON'T THINK LIKE YOU'RE IN AN OBVIOUS TOXIC ENVIRONMENT",
    "ZpGX4SprPTA-00024-00020680-00021744.wav": "TO LESS HORSEPOWER AND YOU DON'T EVEN FEEL IT I JUST FILL THE CAR UP WITH FUEL SO IT'S A FIFTY FIVE LITER",
    "Tbks2pJBbdo-00052-00038672-00039808.wav": "AND THEN PAST PASTE FETCH DATA AND I JUST PUT IN ALL THESE LINKS AND THEN FOR THESE SPECIFIC DATA",
    "Kr5LECV3cXu-00210-00088291-00089363.wav": "AS A MEASURE OF VALUE THE FUNCTIONS ARE",
    "Ihwr_fevmwo-00034-00024969-00026045.wav": "IF YOU KNOW THERE WAS THERE WAS ONE PERSON WHO WAS KIND OF THE LEAD PERSON OR THE FOCUS PERSON FOR THE SOFTWARE IMPLEMENTATION AND YOU KNOW PROBABLY",
    "xXADsgcd-2c-00162-00119191-00120306.wav": "WELL YOU BATCH IT WITH EITHER LETTER OR A FOR ALL THE OTHER WHAT'S IT WE'LL PUT IT EVENTUALLY RIGHT AND YOU SEE WE YOU KNOW THIS IS HOW YOU SET A VALUE",
    "Ci0zFttXIrM-00064-00047266-00048316.wav": "RIGHT WE HAVE FLAME AND IT WON'T STAY ON LONG ALL RIGHT SEE HOW THE FLAME JUST WENT OUT LIKE",
    "y62q2Roz78u-00282-00169904-00170984.wav": "YOU CAN CHECK THAT ONE OUT OR YOU CAN SKIP AHEAD IT'S UP TO YOU",
    "fOxaq-9aUb4-00381-00213563-00214727.wav": "SO ON A SIMILAR TOPIC THAT IS A GOOD POINT",
    "P-AWqB1QItY-00175-00093214-00094316.wav": "POINT NUMBER TWO WAS OH OH SHUT WHAT WAS POINT NUMBER TWO",
    "nKR8mWw55Og-00190-00078069-00079142.wav": "FOREWORD BELOW IS AN ARCHIVE OF ALL DOCUMENTED SHIFTS UNDERGONE BY SCP FIVE",
    "ATkvfokAgfo-00279-00269368-00270760.wav": "AND IT'S JUST REALLY GOOD SO I GO UPWARDS AND OUTWARDS DON'T DRAG DOWN THAT'S WHAT I DO I TRY TO",
    "fOxaq-9aUb4-00343-00187367-00189145.wav": "LET ME SEE OKAY OKAY COOL YEAH IF ANYONE WANTS MORE DETAILED INFORMATION THEN TALK TO AMY",
    "i2MbB4Fmc_A-00031-00033300-00034400.wav": "SERVICES RESOURCES AND SPACES PROVIDED THE KNOWLEDGE SKILLS ABILITIES AND DISPOSITIONS OF LIBRARY WORKERS THE GATHERING IN OF USERS NOT YET USERS",
    "i2MbB4Fmc_A-00064-00066600-00068000.wav": "AND OBSERVABLE AND ARE INTENDED TO COMMUNICATE WHAT MEETING AN OUTCOME LOOKS LIKE OR HOW WE CAN KNOW WHEN AN OUTCOME HAS BEEN MET WITHIN AN INDICATOR TARGETS ARE USED TO QUANTIFY AN INDICATOR OR MARK",
    "8YWBpJTbP3Y-00011-00005732-00006765.wav": "REJOICES THAT YOU CAME AND SINGS YOUR PRAISES AS IT KEEPS YOU SAFE FROM EVERY FORM OF DANGER AND OF PAIN",
    "VhPdG7wMqNk-00029-00024900-00025900.wav": "NEXT SLIDE PLEASE THE LAUNCH PAD IS OUR ENTREPRENEURSHIP AND INNOVATION SERVICE POINT AND IS LOCATED ON OUR FIRST FLOOR BIRD LIBRARY",
    "m3rJs0Q8bPQ-00078-00069894-00071502.wav": "SO WHY DON'T I JUST TAKE SOME OUT THAT ONE'S OKAY AND THEN",
    "qC2XKbLbRk8-00088-00060737-00061985.wav": "BUT THEY ARE DOING A GOOD JOB IN HOLDING THEIR OWN HOWEVER A DIRAC SEA AHEAD IN",
    "_0azrnupduM-00125-00133700-00134700.wav": "IT CHARTS A TIMELINE OF THE UNIVERSITY OF GLASGOW'S RECORD KEEPING FROM FOURTEEN FIFTY ONE WHEN IT WAS ESTABLISHED AND ITS TRANSFORMATION FROM PAPER TO DIGITAL",
    "0dK_HZwohd4-00124-00108320-00109383.wav": "HE GOT WHAT WAS COMING TO HIM BUT WHO DID GO THERE",
    "hDy39t60cD0-00259-00200688-00201888.wav": "A LOT OF ENTHUSIASTS A LOT OF USERS OF OF SOFTWARE OF LIKE MARIADB SERVERS I WANT TO POINT OUT A",
    "A06AYpw_Ory-00006-00010400-00011500.wav": "THE FILM IS DIRECTED BY SHAAD ALI AND PRODUCED BY SONY PICTURES DILJIT DOSANJH AND TAAPSEE PANNU IN LEAD ROLES IN THE FILM",
    "taiuT-89IkI-00065-00038097-00039489.wav": "JUST TO SHOW YOU WHAT THAT LOOKS LIKE ANNETTE GORDON REED HEMINGSES WE FIND",
    "-Qydq5EtDOc-00067-00070716-00071934.wav": "BUT IT HAS THAT ILLUSION ESPECIALLY IF YOU HAVE TROUBLE SETTING BOUNDARIES AND OFTEN YOU GET SUCKED INTO SPENDING TIME IN QUADRANT THREE SO AN EXAMPLE COULD BE YOU KNOW THAT YOUR PRIORITY MAYBE IS TO PREPARE FOR AN EXAM",
    "EsP0yM0TK9c-00056-00055713-00057250.wav": "THAT SOUNDED LIKE A FLOCK OF INSECTS",
    "xwBI1bwxZOY-00367-00137746-00139669.wav": "SO IF YOU TAKE A LOOK AT THE SENTENCES I WROTE HERE",
    "H8HdPQHnx5y-00487-00217648-00218762.wav": "TEAM OF OKAY WE'RE MAKING THIS CHANGE TO COMPONENT X Y AND Z WHO IS ACTUALLY CONSUMING IT OKAY I CAN SEE THAT THERE IS THESE THREE APPLICATIONS CONSUMING IT COOL",
    "BN6fXT8QSdI-00008-00012200-00013900.wav": "WRITING THIS STORY IS A CHALLENGE BECAUSE EVERYONE KNOWS THE STORY AND THE CHARACTERS IN OUR COUNTRY HISTORY WASN'T RECORDED AT THE TIME AND PEOPLE STILL THINK IT IS MYTHOLOGY AND NOT TRUE BUT IT IS OUR HISTORY",
    "7S3nwMnW1mQ-00036-00063500-00064900.wav": "JUDICIAL WATCH INCORPORATED A CONSERVATIVE NON PARTISAN EDUCATIONAL FOUNDATION PROMOTES TRANSPARENCY ACCOUNTABILITY AND INTEGRITY IN GOVERNMENT POLITICS AND THE LAW",
    "iYNbrH-SxLo-00004-00007272-00009554.wav": "YOU CAN TALK AS LOR AT THE SAME TIME",
    "AMmQbDReEQk-00063-00049911-00051105.wav": "WE HAVE A THIRD SYSTEM WHICH HAS ANOTHER TYPE OF UNSTABLE FIXED POINT CALLED SADDLE POINT ALL OF THESE THREE SYSTEMS ARE LINEAR SYSTEMS",
    "_F_NrNcJNFg-00083-00081150-00082210.wav": "THAT IS WHAT THIS CAMPAIGN IS ABOUT AND ALONG WITH THE CWA WE ARE GOING TO WIN THIS ELECTION AND WE ARE GOING TO TRANSFORM AMERICA THANK YOU ALL VERY VERY MUCH",
    "lzEsx1EMC1g-00089-00078040-00079984.wav": "ALL RIGHT SO I'VE GOT MY FIELD NOTES STAMP SET SO I'M JUST GOING TO GO WITH ONE OF THESE IMAGES",
    "AiO-SHDOG5u-00016-00011282-00012376.wav": "BEING ABLE TO ANSWER A QUESTION OF DOES A USER HAVE ENOUGH MONEY TO SEND TO ANOTHER USER WITHOUT KNOWING WHO THE USER IS OR EXACTLY HOW MUCH THEY HAVE",
    "i2MbB4Fmc_A-00084-00087100-00088200.wav": "THIS RISK TAKING CAN LEAD SOME COLLEAGUES TO BE UNCOMFORTABLE WITH ASSESSMENT THOUGH MOST WILL ACKNOWLEDGE THAT DISCOVERING A PROBLEM AND LEARNING HOW TO HELP RECTIFY IT IS INFINITELY",
    "ATkvfokAgfo-00265-00244744-00245936.wav": "AND THAT IS ALL I HAD TONIGHT FOR DINNER THE THE LAST MEAL WHATEVER SO REALLY FANCY FOOD",
    "uv_A3Gn6eDI-00221-00213492-00215963.wav": "GUYS I'M REALLY CONFUSED BY THIS",
    "iTsN4gYDQiI-00492-00415156-00416443.wav": "IT IS YOU KNOW HEARING ALL THE THE WONDERFUL ASPECTS FROM YOU SEE I LIKE JUST I HAVE ONE EYE ON THE CHAT AND THERE'S JUST SO MUCH GOOD STUFF IN THERE THAT I'M",
    "QNJPFHSbvGk-00101-00081288-00082784.wav": "URAL SIGN YEN SIGN AND PHRASES SMILEY FACE FROWN FROWNY FACE WINKY FACE HEART EMOJI",
    "hDy39t60cD0-00003-00002136-00003200.wav": "ATTACKS AND I HOPEFULLY I HOPE YOU WILL FIND IT INTERESTING AND PRETTY PRETTY RESOURCEFUL MY NAME IS DAN DEMETER",
    "-2V5jdMxRMo-00107-00094702-00095724.wav": "IS OVERWHELMING THE SOLUTION ITSELF SO AGAIN THAT H EQUALS ZERO POINT FIVE CASE THAT'S GONNA BE OUR",
    "uo9f9QJPUJ8-00021-00026012-00027327.wav": "YEAH IT SNAPS RIGHT ON THAT'S GREAT IF YOU WANT TO YOU KNOW MAYBE YOU CAN ORDER ANOTHER ONE FOR ANOTHER BUILT PLATE ONLINE TO SWAP IT OUT",
    "fevs-b2fuN8-00199-00211800-00213200.wav": "CLOSER CLOSER LOOK AT IT ANY CLOSING THOUGHTS YOU WANT TO SAY BEFORE WE OPEN IT UP LISA I GUESS I WOULD JUST ADD THAT YOU KNOW WE WE ULTIMATELY WE CHANGED OUR MINDS ABOUT SUSTAINABILITY WE THOUGHT WE COULD SUSTAIN OURSELVES LONG",
    "UpAygmmv6Zk-00014-00009936-00011484.wav": "CHAIN DRIVE INDOOR CYCLING BIKE BY SUNNY HEALTH AND FITNESS",
    "PN3p12f_0p8-00113-00073489-00074770.wav": "ONE POSITION BUT I'M JUST USING MY ELBOW SO ITS OKAY OF WHAT IS IT OK",
    "EsP0yM0TK9c-00252-00263256-00264350.wav": "I DON'T KNOW OH IT WAS A NEW TRIGGER WHAT ARE YOU BUYING THANK YOU FOR YOUR PATRONAGE",
    "5jkEa0qHI7M-00044-00029464-00030464.wav": "NOTICING THE SENSATION OF SITTING THE CREATION PARTS OF THE BODY COMING UP TO THE TORSO AND",
    "-BRrloXj0b0-00003-00001036-00002755.wav": "TIME FOR QUESTIONS SO WE WILL ADJUST THE LUNCH BREAK ACCORD ACCORDINGLY FROM HERE",
    "O8_CzVPBq1g-00084-00071297-00072687.wav": "ALL RACES REALLY ARE PROTECTIVE OF THEIR SPACE AND WE REALLY WANT TO CREATE A SACRED SPACE",
    "iTsN4gYDQiI-00324-00286587-00288203.wav": "SOMETIMES YOU KNOW THAT CAN BE IN PERSONAL RELATIONS WITH FRIENDS BUT THAT CAN ALSO BE IN PROFESSIONAL RELATIONS AND I THINK THAT IN THE CHAT SOMEBODY EARLIER SHARED YOU KNOW IN TOXIC SITUATIONS YOU CAN'T EASILY EXTRICATE YOURSELF FROM THAT SO IT'S NOT ALWAYS AN OPTION TO DO THIS",
    "3hOHMDxGDSg-00032-00030198-00031239.wav": "I PUT A LINK A LINK IN TOTAL PROFILES BUT THERE'S A COMPLETELY BLANK DOCUMENT THERE THE ONLY THING THERE'S A TITLE THAT SAYS LIKE OH TYPE VISION DOCUMENT THAT WE WAS WAS BANDIED ABOUT",
    "i2MbB4Fmc_A-00121-00125100-00126900.wav": "WHEN AN ENVIRONMENT OR USERS HAVE MOVED SUFFICIENTLY ON FROM THE INITIAL NEED WHEN OTHERS OUTSIDE THE LIBRARY MIGHT BE BETTER POSITIONED TO TAKE ON THE WORK OR WHEN ROOM MUST BE MADE FOR SERVICES RESOURCES AND SPACES THAT MEET NEW NEEDS OF COURSE",
    "j7UUgMJOAgk-00048-00037656-00038937.wav": "SO THEY HAVE A RING OF SIX CARBONS EACH WITH THE SP TO HYBRIDIZATION SO SP TWO THAT MEANS WE HAVE THREE BONDS AND THAT ONE ON HYBRIDIZE PEOPLE",
    "E9HF8izz8ak-00109-00079332-00080412.wav": "AND HAVE SOME EXCITING NEWS SO I'VE BEEN ON THE HUNT FOR THIS STARBUCKS CUP SENSE A",
    "_NIgqn87tiy-00043-00048056-00049184.wav": "IN THE MAGNETIC RACK AND THE DNA IS REMOVED FROM THE TUBE",
    "CYZcIuipG9g-00001-00000528-00001592.wav": "IT IS THE AFTER IMAGE THAT WE ARE LOOKING AT WHICH IS THE INFORMATION THAT WE SEEK",
    "JlxiAy9sAbY-00606-00287684-00288953.wav": "WE HAVE TO FACTOR IN A SEVEN PERCENT SALES TAX ON THE ROOM RENTALS",
    "iTsN4gYDQiI-00078-00084397-00085935.wav": "THAT DEVELOPS ALSO COLLECTIVE PRACTICES THAT ALLOW TO TO FIND WAYS TO WORK WITH INDIVIDUAL HARDSHIP OF SUFFERING BUT IN A WAY THAT IT IT CREATES A SOCIAL ACTION OR IT IT BECOMES USEFUL TO THE WHOLE COMMUNITY",
    "_0azrnupduM-00161-00166400-00168000.wav": "FOR NOW YOU KNOW THE THE TOOL AND THE METHODOLOGY TAKE INTO ACCOUNT A NUMBER OF DIFFERENT FACTORS AND THE PRIORITIZE PROCESSING OF COMPUTER STORAGE MEDIA ON A SCALE OF ONE TO FIVE WITH ACTIONS RECOMMENDED",
    "shjJsUifGjA-00110-00091017-00092238.wav": "STOP YOUR FOOLING WHAT'S ON YOUR MIND WE AIN'T FOOLING WE'RE DEAD SERIOUS SILENCE",
    "y62q2Roz78u-00273-00161488-00162584.wav": "MODULE HOW ABOUT WE USE THE GET FUNCTION IN THE MODULE TO CREATE A DATE OBJECT FROM A STRING",
    "-Qydq5EtDOc-00032-00032388-00033705.wav": "SO THE FIRST THING THAT TENDS TO HAPPEN IS YOU BECOME AWARE OF SOMETHING YOU NEED TO DO IN MY CASE IT'S WORKING ON THINGS LIKE TAXES AND FINANCE THAT'S TYPICALLY WHAT I PROCRASTINATE",
    "m3rJs0Q8bPQ-00105-00106069-00107133.wav": "VIDEO AND USING THE SCRIPT THE NEXT VIDEO WILL BE ADDING THE AUTOMATED VOICE OVER SO",
    "_RJ5I2tZlvg-00053-00041844-00042864.wav": "IS SOMETHING I GENUINELY WANT TO CRY OVER BECAUSE BECAUSE BECAUSE BECAUSE IT IS THIS DIESEL SKIRT",
    "pOuiRIZ1eso-00047-00035200-00036232.wav": "A RADIUS NOT THE RADIUS OF THE BIG CIRCLE OR THE RADIUS OF THE SMALL CIRCLE",
    "O8_CzVPBq1g-00046-00040824-00042334.wav": "WITH WE WERE ABLE TO THERE'S SUCH A A DIFFERENCE BETWEEN JUST BETWEEN THE OTHER PARTS OF THE COUNTRY",
    "mLj4yHUBwuy-00031-00039608-00040784.wav": "NEVER ONCE CONSIDERING HOW IN THE WORLD SHE WAS TO GET OUT AGAIN",
    "Ihwr_fevmwo-00019-00017022-00018252.wav": "NO THAT SOUNDS GOOD AND MAYBE I'M THE IF IFTHERE WAS ACTUALLY AN INTEREST IN TRYING TO MOVE THE METRICS FORWARD DURING THE CALL MAYBE THERE COULD BE AN IDENTIFIED METRIC LIKE IN THE PRIOR WEEK THAT PEOPLE COULD BRING",
    "iTsN4gYDQiI-00464-00395292-00396318.wav": "GONNA REALLY HELP MOVE THAT COMMUNITY FORWARD IT'S SOMETHING THAT COINCIDENTALLY WE JUST STARTED PLANNING A YESTERDAY SO THAT'S SOMETHING I TAKE AWAY",
    "D9oWoELjYHu-00061-00033624-00035184.wav": "THIS PLACE IS A REAL PORTAL TO HELL WHICH HOUSES THE PRESENCE OF TWO HUNDRED DIFFERENT DEMONS",
    "ROQ5k7sCN7I-00026-00026510-00028245.wav": "AND TWO X MINUS THREE LET'S ADD THREE TO BOTH SIDES WE'RE GONNA GET TWO X IS LESS THAN OR EQUAL",
    "3hOHMDxGDSg-00054-00044496-00045669.wav": "AND THE GOOD NEWS IS THAT OUR TOP BID GET ACCEPTED SO WE'LL BE YOU KNOW PLANNING TO PUT TOGETHER A PRESENTATION AND OF COURSE SHARING IT WITH THE EVERYONE ON THE",
    "_0azrnupduM-00174-00182100-00184200.wav": "HELPED US REVIEW POLICIES AND PROCEDURES AND I GUESS BY USING THE THE AFFORDANCE OF FORENSIC TOOLS WE CAN CREATE OR START TO CREATE ECONOMIES OF SCALE SO PROCESS VERY LARGE VOLUMES OF DIGITAL INFORMATION BY AUTOMATING OR SEMI AUTOMATING SOME OF THESE",
    "Ihwr_fevmwo-00269-00205629-00207521.wav": "KNOW WHY DON'T WE JUST DO THIS HERE'S THE HERE'S THE LINK TO THE GOOGLE DOC THE GOOGLE SPREADSHEET SO I'LL DO THE FIRST PASS AS A AS A GOOGLE SPREADSHEET AND THEN ONCE I'VE ONCE I'VE GOT THAT DOWN THE SECOND PASS WILL BE A PUT IT IN A TABLE IN THE REPO",
    "C_Dhd2VyZ7Q-00037-00027064-00028100.wav": "TO STEP ASIDE MY EXTERNAL CONVERSATIONS WITH OTHERS BEGAN TO CHANGE I STARTED TO",
    "m3rJs0Q8bPQ-00102-00101607-00103984.wav": "I'M HOPING THEY WILL ADD MORE BUT THOSE ARE THE TWO RIGHT NOW",
    "otKu0tzQpwg-00001-00000481-00002025.wav": "BEFORE I DO THAT YOU AND I NEED TO HAVE A QUICK LITTLE TALK WHAT IS UP EVERYBODY THIS IS CHRIS FROM THE REWIRED",
    "_qB-z2sy9Ek-00372-00210796-00211886.wav": "CHRISTIAN I WOULD HAVE NEVER KNOWN WOW THAT WAS AN EYE OPENER SO I REALIZED",
    "UaRnQ8nPdR4-00114-00077560-00078783.wav": "PLEASE PLEASE PLEASE PLEASE PLEASE NO NO NO NO NO NO",
    "4fv0g0knOZA-00097-00089223-00090441.wav": "ON THE RIGHT WILL HAVE MORE ENTROPY THEN ON THE LEFT THE LOWER PRESSURE OR PRESSURE MEANS LARGER VOLUME AND THAT ALSO MEANS HIGHER ENTROPY SO BOTH OF THE CHANGES ARE SAYING THE SAME THING",
    "yhBLxOdf1P4-00103-00068272-00069280.wav": "IT WORKS ITS VALUES AND THE KNOWLEDGE THAT IT HAS ABOUT THE LAND SO AN EXAMPLE IS OF THIS IS",
    "sCZcMd5R0fo-00030-00031789-00032878.wav": "SO B TIMES B TIMES B TIMES B ARE THE NUMBER OF B'S THAT COME OUT",
    "yn-CJ363ZzY-00069-00049930-00050995.wav": "CAR IS COMING TOGETHER TAKE A LOOK AT THAT YOU GOTTA LOVE THAT EMBLEM AND THAT WHOLE FRONT END",
    "O8nCeYr0Izy-00018-00019448-00021184.wav": "EITHER BOTH CONVERGE OR BOTH DIVERGE",
    "vobR23oit2y-00020-00019164-00020184.wav": "IN CHINESE OR LIKE A PHILIPPINES OR THAILAND OR VIETNAM AND I ALSO DON'T EXACTLY WHY EUROPEAN",
    "iTsN4gYDQiI-00284-00253859-00255169.wav": "GO ELSEWHERE AND CREATE AND IF THOSE PEOPLE IN THAT ONE PLACE DIDN'T WANT TO TO HAVE ME THE WAY I WAS AND WITH THE STRUCTURES THEY HAVE SET UP",
    "xXADsgcd-2c-00040-00033089-00034235.wav": "YES OKAY SO IT'D BE REALLY HELPFUL TO THE EXTENT THAT YOU'RE FAMILIAR WITH ELBOW WHICH IS A WAY OF DOING HTML BUT LIKE WITH LISTS",
    "su79iUkFHnE-00058-00047900-00049100.wav": "OH I JUST REALIZED I HAVE TO SHAKE THIS I'M SHAKING A CARBONATED DRINK OH MY THIS IS WORSE THIS IS WORSE OF AN IDEA THAN I THOUGHT HONESTLY",
    "-2V5jdMxRMo-00070-00059645-00060660.wav": "I PLUS ONE IS EQUAL TO Y I TIMES ONE MINUS NOT FIVE TIMES ZERO POINT TWO IS JUST GONNA BE ONE OR Y I PLUS",
    "vbGCZjFU3MA-00133-00084552-00085552.wav": "SECTION I HAVE THE PAGE NUMBER AS A B CONTINUING BUT IF I GO UP TO PAGE TWO IT'S NOW NUMBERS SO I",
}  # noqa


DICT_DEV = {
    "txeQSfXGy68-00068-00034338-00035402.wav": "AND I'D LIKE TO INCLUDE SOME OF THOSE BECAUSE WHY NOT RIGHT TO BE A FAN OF OURSELVES TO BE ENCOURAGING",
    "FHBmNaDvWiM-00524-00377256-00378516.wav": "BUT THAT'S NOT TRUE SO TAKE YOUR TIME BE PATIENT DO WHAT YOU LOVE AND AND AND ENJOY THE JOURNEY",
    "XUYT__j-lwE-00099-00078948-00080112.wav": "THIS WAS THAT WHICH WAS SPOKEN BY THE PROPHET JOEL SO THE HOLY GHOST MOVED INTO THAT WHICH WAS SPOKEN",
    "Ihwr_fevmwo-00110-00070722-00071805.wav": "EXACTLY HUNDRED PERCENT AND SO THE ANDES QUESTION IS HOW ON THAT PAGE HOW EASY IS IT TO ALLOW FOR PEOPLE TO INPUT PARAMETERS THAT ARE RELEVANT TO THEM",
    "FkQfXGqx-no-00024-00040742-00042689.wav": "SPEAKING TO THE SUNDAY PEOPLE BEFORE VALENTINES DAY TWENTY SIXTEEN LEON SAID HE ENJOYED THE FAME GOGGLEBOX BROUGHT HIM AND JUNE ADDING I'M HUGGED AND KISSED BY WOMEN IN THE STREET EVERY DAY AND NOT JUST IN LIVERPOOL",
    "_0azrnupduM-00097-00104400-00106100.wav": "WE HAVE PEOPLE WHO COME FROM AN ARCHIVING BACKGROUND A LIBRARIAN A LIBRARIANSHIP BACKGROUND AND MANY OF WHOM HAVE HAD VERY LITTLE TO DO WITH DIGITAL AND DIGITAL ARCHIVING SO THAT PROVIDES US A BIT OF A KIND OF FOUNDATION AND GROUNDWORK FOR ALL OF THIS NEXT SLIDE",
    "f7RwlHD7Dtg-00103-00069084-00070122.wav": "OH YOU KNOW THAT PSY AQUARIUS OH I CAN ACTUALLY REST I FEEL THAT'S WHERE SPIRIT IS TAKING YOU LET'S",
    "VhPdG7wMqNk-00061-00057600-00059200.wav": "BUT HAPPEN TO BE HERE THESE CONNECTIONS WITH THE COMMUNITIES AND CAMPUS NEED TO BE OUR TICKULATED IN THE LIBRARY'S OVERALL STRATEGIC PLAN AND ALONG WITH CONNECTING TO THE MISSION OF THE LIBRARIES IT IS IMPORTANT TO CONNECT WITH AND LISTEN TO THE",
    "peqws-UzqBU-00047-00026710-00027769.wav": "SEED OF THE SERPENT THAT IS SATAN'S OFF SPRING OR CHILDREN OF THE DEVIL FRIENDS",
    "VDBjCrR1RPM-00016-00014802-00015888.wav": "MY TEARDROPSY IN AMAZING JOB I GOT A GRATEFUL OF A BAD ATTITUDE I GOT A TWENTY FIVE CENT PHOTO BOOTH PICTURE",
    "qC2XKbLbRk8-00151-00121183-00122570.wav": "BUT HAS DIED HE'S BACK AGAIN AND GETTING MORE TIME STUNNED",
    "_0azrnupduM-00221-00236600-00238000.wav": "THE IDEA AND DEFINITELY THE VISION IS TO BUILD ON THESE THROUGH THE PILOT BY TRIALING AND AND TRYING DIFFERENT THINGS WORKING ALSO WITH THE COMMUNITY WITH PEERS ELSEWHERE SEE WHAT OTHERS HAVE DONE AND",
    "sWJ-14NkCHg-00012-00016422-00017620.wav": "THE STANDARD CLASSROOM ANSWER THAT FOR A NATIVE ENGLISH SPEAKER MAKES YOU SOUND A BIT LIKE A ROBOT",
    "3hOHMDxGDSg-00238-00219146-00220752.wav": "SO THIS IS AN OFFICIAL PROPOSAL SORT OF LIKE CLARIFYING THE HIGH LEVEL VISION AND AND THEN SHARING THAT WITH EVERYONE TO MAKE SURE THAT BOTH THE COMMUNITY IS AWARE OF THE EFFORTS THAT WE'RE DOING AND AGREE THAT IT IS A",
    "EsP0yM0TK9c-00068-00067947-00069373.wav": "OR POLYPETULUM VAN DER PLANKY PLANKY I DON'T KNOW ANYTHING IMPORTANT ON THIS SIDE",
    "uPWq505rHCY-00020-00016840-00017960.wav": "SO I THINK THIS IS HOW I LIKE TO SEE MY MENU WORK FOR NOW SO I STILL HAVE MY SPEED TRIP FUEL ECONOMY",
    "CYZcIuipG9g-00009-00009992-00011064.wav": "UM YOU SAW THE BLUE LIGHT IN THERE THAT'S A A UV LED TORCH FROM ALIEXPRESS AND YEAH",
    "txeQSfXGy68-00075-00039955-00040964.wav": "TO GROW AND TO BE GENTLE WITH YOURSELF AND TO LOVE I LOVE YOU",
    "XZ1f89-ecUA-00071-00079035-00080049.wav": "THERE'S AN EXTENT TO WHICH THEY CAN DO BOTH AND IN FACT SOME OF OUR PARTNERS HAVE BEEN ABLE TO LEVERAGE THEIR COMMERCIAL DIGITIZATION ACTIVITIES TO BE ABLE TO CONTRIBUTE SCANS THEIR WORK TO THE TO THE NEWS ARCHIVE",
    "nU032J1d8fY-00193-00066870-00068497.wav": "WITH MULTIPLICITY OF TWO MEANING IT TOUCHES AT ZERO ZERO SO IT'S GONNA",
    "1mNa-aOTQQ4-00045-00046614-00047767.wav": "KILLER IS GOING TO WORK SON NOT SURE HOW I GET KILLED RIGHT THERE I DON'T KNOW",
    "xIJG6z2D-44-00009-00006848-00008407.wav": "FOR EXAMPLE THIS IS YOUR BRAIN AND THIS IS THE BRAIN STEM AND HERE YOU HAVE THE BRAIN OK NOW AT ANY POINT IF YOU PICK UP ANY PARTICULAR",
    "i2MbB4Fmc_A-00085-00088300-00090000.wav": "FIND AND THEREFORE PERPETUATING PROBLEMS SOMETIMES FINDING PROBLEMS OR OPPORTUNITIES TO MAKE IMPROVEMENTS CAN RESULT IN REALLOCATION OF LIBRARY WORKER EFFORT AND ATTENTION IN SOME CASES THAT MIGHT BE DISCONCERTING IN OTHER CASES",
    "fOxaq-9aUb4-00545-00306113-00307297.wav": "TO GET TO OUR LAST TOPIC ON THE AGENDA WHICH WAS SUCCESSES AND CHALLENGES IN TWENTY TWENTY ONE",
    "plmlE78FQzu-00019-00022950-00024200.wav": "I GO AROUND FROM PLACE TO PLACE TALKING ABOUT PHYSICS AND I FIND THAT FROM THREE YEAR OLDS TO NINETY THREE YEAR OLDS THEY ARE STILL FASCINATED HOW DOES IT ALL WORK THAT'S IT ISN'T IT HOW DOES IT ALL WORK",
    "y62q2Roz78u-00670-00416088-00417384.wav": "THEN CALL THE CPU PERCENT FUNCTION WITH POINT ONE SECONDS",
    "SJV-LD4EC14-00227-00195492-00196684.wav": "OKAY SO ONCE I'VE MADE ALL MY CHANGES THERE AND I CHECK FOR THEM USING THE PREVIEW I GO DOWN I GO DOWN TO THE BOTTOM",
    "4fv0g0knOZA-00007-00005736-00006972.wav": "AND WE CAN'T TRULY MAKE AN ARREST SYSTEM BECAUSE WE CAN'T STOP THE FLOW OF ENERGY EVERYTHING RADIATES WAS CALLED BLACK BODY RADIATION EVERYTHING RADIATES",
    "Tbks2pJBbdo-00705-00523952-00525040.wav": "THIS PANGOLIN SPECIFIC INTERNAL LINEAGE NAMES IN THE OUTPUT SO DELTA IS B ONE SIX HUNDRED AND SEVENTEEN DOT TWO OF COURSE AND",
    "m3rJs0Q8bPQ-00075-00067451-00068755.wav": "WORK AND THIS LOOKS LIKE THIS MAY NEED TO HAVE OH",
    "RbWEMdPcOm4-00015-00010515-00011825.wav": "CONTACT THE TUTORING CENTER DISTANCE EDUCATION OFFICE WRITING CENTER OR VISIT ONE OF THE CAMPUS COMPUTER LABS DISTANCE",
    "5jkEa0qHI7M-00054-00038744-00040008.wav": "SHOULDER THE BICEPS AND TRICEPS THE ELBOW WRISTS THE FINGERS AND THE OTHER SIDE THE SHOULDER",
    "L8FaMU19dFu-00793-00508150-00509168.wav": "PLOT A LOT MORE RIGHT NOW ACTUALLY SO LET ME SEE THAT OH YES SO YES THE DATASET",
    "Ihwr_fevmwo-00072-00045429-00046965.wav": "SURE SO I MEAN I THINK YOU THINKING ABOUT BOTH GOMORRAH LAB AND AUGER JUST THINKING ABOUT THE UI STRUCTURE THAT BOTH OF THEM HAVE THE THE GOAL FOR GOMORRAH LAB IN MY MIND IS TO CREATE PANELS",
    "i2MbB4Fmc_A-00157-00156700-00158300.wav": "SPEAKING OF ASSESSMENT CAPACITY THE MULTIPLICITY OF POSSIBLE CONCLUSIONS TO A PARTICULAR ASSESSMENT PROJECT OR PROCESS REVEALS THAT A NUMBER OF SKILLS MIGHT BE REQUIRED OF OR USEFUL TO LIBRARY ASSESSMENT PRACTITIONERS AMONG THOSE SKILLS",
    "rmZWrxDDSy0-00026-00022896-00023984.wav": "TUNE IN WITH US TOMORROW AT TWELVE THIRTY ON TV CARIB THANK YOU AND HAVE A GREAT EVENING",
    "jtUDQ4sPRtk-00010-00005676-00006678.wav": "OF IS IN THAT CIRCLE SO WE HAVE TO TAKE PI WHICH IS THREE POINT ONE FOUR AND WE HAVE TO MULTIPLY",
    "i-4I0Gswg2I-00061-00041120-00042168.wav": "AWAY FROM SATAN THE MORE WE GO AWAY FROM GOD THE WEAKER WE ARE WITH WITH THE MORE WE GO AWAY FROM GOD",
    "Zv-XEPw03BY-00010-00004812-00005947.wav": "I'M SORT OF A BIT MIXED ABOUT THAT I GUESS BECAUSE THEY GAVE US FOUR EIGHT TWELVE SIXTEEN SEVENTEEN MY MATH IS TERRIBLE",
    "kRefZ6aGymu-00068-00037629-00038819.wav": "LIFE I MEAN LIKE WE CAN'T REALLY LIVE WITHOUT IT OR COMPUTERS SO IF SOMEONE",
    "Z6-uTG6jc5E-00014-00011616-00012618.wav": "IT'S KIND OF LIKE I DON'T KNOW I'M JUST SEEING SOMEONE LIKE A MUSIC MUSICIAN",
    "_7YTZI0xnZ8-00221-00157280-00158384.wav": "ASK YOUR QUESTION NOW THAT CHRISTINE'S DONE PRESENTING OR YOU CAN PUT IT IN THE CHAT",
    "i2MbB4Fmc_A-00114-00118400-00120000.wav": "RESULTS THESE STEPS MIGHT INCLUDE A JOINT NEEDS ASSESSMENT IDENTIFYING GAPS AND NEEDS FOR THE CONCEPTUALIZED SERVICES RESOURCES OR SPACES THAT THE PROGRAM MAY ADDRESS JOINT ARTICULATION OF PROGRAMMATIC OUTCOMES MAPPING",
    "fOxaq-9aUb4-00425-00240443-00241955.wav": "BUT WHO HAVE YOU PARTNERED WITH IN YOUR COMMUNITY TO PROVIDE FINANCIAL LITERACY SERVICES",
    "yJzm8nl_Cik-00001-00001500-00002500.wav": "PART OF THE THREE THOUSAND METRE PEAK OF CERRO ARMAZONES WAS BLASTED AWAY AS A STEP TOWARDS LEVELLING THE SUMMIT",
    "y62q2Roz78u-00238-00136248-00137432.wav": "HOW ABOUT WE CHECK THE LENGTH OF THE RESPONSE TEXT USING THE LENGTH FUNCTION",
    "hnQ9nrNz6y8-00179-00113320-00114359.wav": "SIMILAR TO GINGER AND IT HAS SOME COMPOUNDS IN IT PARADOXICAL PARADOXINE THERE'S SEVERAL",
    "4pWWqtCSiOu-00032-00031920-00033096.wav": "IS PROBABLY LIKE A HALF GUARD IS THIS WRETCHED JUMP AND I GOT IT CORRECT MY NOSE WHAT MY NOSE",
    "3PYrtkzrMqu-00014-00012092-00013202.wav": "THE DERIVATIVE SO FOR THIS ONE I HAVE LOW IS COSINE D HIGH THE DERIVATIVE OF ONE IS ZERO",
    "fevs-b2fuN8-00165-00175600-00176900.wav": "AND THAT'S BEEN FASCINATING ALSO IN TERMS OF ONE THING THAT'S NOT SO GREAT IS IS LOSING OUT ON ON TALENT SO SAY WE HAVE YOU KNOW AN ORGANIZATION THAT'S IN THE DATA CURATION NETWORK AND THEY HAVE TO EXIT OR THEY WERE NEVER ABLE TO COME IN TO BEGIN WITH SO",
    "xXADsgcd-2c-00197-00153019-00154677.wav": "DISCRIMINATING USES LIKE EVERYWHERE THAT THAT'S JUST LIKE IT RECORDS WHICH I DON'T USE IT THIS BUT YOU KNOW I COULD HAVE BUT BUT TO ME THE ENTIRE WORLD IS MADE UP OF THOSE THINGS IT FUNCTIONS THAT DO STUFF TO THOSE THINGS",
    "iTsN4gYDQiI-00339-00301786-00303475.wav": "BUT I ALSO WANTED TO POINT OUT I LOVE YOUR PHRASING OF YOU KNOW RECIPROCITY OF YOU KNOW LIKE OKAY I AM GIVING THIS JOB OR THIS PERSON YOU KNOW THIS MUCH AND I THOUGHT YOU WERE GOING TO SAY I EXPECT TO RECEIVE SOMETHING SIMILAR FROM THEM BUT ALSO YOU MENTIONED",
    "5bahxR6hBrg-00001-00000495-00001778.wav": "MIGHT IT MAKE YOU HAPPY LET'S CHECK IT OUT EVERYBODY SCOTT AND JEFF WE'RE",
    "4fv0g0knOZA-00042-00042078-00043386.wav": "AND IT WON'T CHANGE THE MACROSCOPIC PROPERTIES OF PRESSURE AND TEMPERATURE SO WE HAVE A LOT OF DIFFERENT MICROSTATES COME WITH GASES THAT WE CAME THE MACROSCOPIC PROPERTIES THAT WE'RE LOOKING AT",
    "xXADsgcd-2c-00223-00176452-00178443.wav": "THAT'S A DISCRIMINATING YOU GET NOW WE CAN FAKE THAT IN THE OLD WORLD WE CAN HAVE AN I PAVEMENT INTERFACE AND LIKE THAT'S COOL WHEN I DO THAT WHEN I WRITE IN C SHARP THAT SORT OF STUFF BUT I JUST KIND OF SEE THE WORLD THAT WAY AND SO THIS SORT OF SIMPLE FUNCTIONAL PROGRAMMING HAS ALWAYS APPEALED TO ME",
    "3hOHMDxGDSg-00143-00132357-00134484.wav": "DEPLOYS THE COLLECTOR AND CONFIGURE THEM ACCORDING TO THE USERS SPECIFIED IN THE USER INTERFACE THAT IT'S PRETTY BASIC WILL BE HOPEFULLY WILL SUPPORT MORE FEATURES LIKE A SAMPLING AND FILTERING WHICH WILL MAKE IT ON SKIN A LITTLE BIT MORE COMPLEX AND JUST REACT APPLICATION THAT",
    "3hOHMDxGDSg-00155-00146823-00148560.wav": "WHICH COLLECTOR WE SHOULD TALK TO AND POLITICAL GROUP IS A IS A PART OF THE COLLECTORS PIPELINE THAT RESPONSIBLE TO ACHIEVE A COMMON GOAL EITHER COLLECT LOGS OR METRICS AND SEND THEM AS THE GATEWAY TO ONE OF THE VANDAL",
    "3hOHMDxGDSg-00093-00084480-00086037.wav": "AND LAST BUT NOT LEAST WE HAVE THE NEW GOOGLE DRIVE FOLDER FINALLY AND A LINK THESE BOOK MARKET YOU KNOW THAT WILL BE OUR GO TO FOLDER FOR ALL THE DIFFERENT DOCS AND PRESENTATIONS THAT WE ARE WORKING ON INCLUDING WHITE PAPERS AND ANYTHING ELSE THAT WE WANT TO WORK ON FROM THE TAG",
    "W1PY-i2pV1E-00005-00003571-00004736.wav": "FIRST WE'LL SOLVE THE INEQUALITIES AND FLOAT THEM ABOVE THE NUMBER LINE THEN WE'LL USE",
    "hD9mU-4RQm4-00242-00103804-00104941.wav": "AFTER LOCKDOWN SO THE CHOICE WAS LIMITED BUT IT WASN'T EXPENSIVE AND IT WAS FINE",
    "gApY05Jtm8o-00141-00083516-00084539.wav": "UM SO LOOKING AT THIS SLIDE THE JUST THE STRICTLY ENVIRONMENTAL ASPECT OF TELEWORK",
    "6h7YN83tGo4-00186-00168344-00169360.wav": "IT ON HIS DEVICE TEST HIS OWN SCENARIOS CAN REPORT IT BACK AND WE WILL BE HAPPY FOR THAT",
    "uv_A3Gn6eDI-00308-00291801-00293704.wav": "GOOD THING MINES",
    "nFgtCRh-ZaM-00000-00000003-00001892.wav": "SO AFTER WE TERMINATOR SEED IN WET PAPER TOWEL IN IN THE HEATING PAD YOU WANTED TO CREATE CRACK THROUGH",
    "GyM7h_VUfS0-00007-00011931-00013538.wav": "TO SEE HER UNION WITH HARRY WHO SHARES THE SAME QUALITIES IS A SOURCE OF GREAT JOY FOR US AS PARENTS WE WISH THEM A LIFETIME OF HAPPINESS AND ARE VERY EXCITED FOR THEIR FUTURE TOGETHER",
    "OYqoclzWw5A-00002-00000506-00001813.wav": "MY NAME IS NIK PETERSSON AND I WILL BE REVIEWING AN EXAMPLE OF A PORTLAND STATE UNIVERSITY LIBRARY WEBSITE THAT USES SCREENSHOTS",
    "sCZcMd5R0fo-00028-00028902-00030006.wav": "THIS TELLS US HOW MANY BASES COME OUT SIDE THE RADICAL AND HOW MANY REMAIN INSIDE",
    "taiuT-89IkI-00150-00098268-00099443.wav": "SERVICES RESEARCH GUIDES AND MORE THE URL IS LIB DOT LAW DOT UW DOT EDU",
    "SJV-LD4EC14-00045-00042440-00043540.wav": "VERSION CONTROL SYSTEM WHICH BASICALLY IS MORE ADVANCED THAN GOOGLE BECAUSE GOOGLE YOU KNOW WHEN YOU ARE WORKING ON A DOCUMENT AND YOU'RE TRACKING THE CHANGES SO THAT'S VERSION CONTROL IS SHARING",
    "fevs-b2fuN8-00056-00063100-00064500.wav": "SO HOW CAN WE DO THIS SUSTAINABLY SO THE GRANT WHEN WE ALL KNOW GRANTS AND THAT'S HOW THEY WORK SO WHAT WE'RE GOING TO DO WHEN IT WAS OVER BECAUSE WE KNEW THAT THE DATA STEWARDSHIP WITHOUT THE OVER AND WE WOULD NEED TO CONTINUE TO CONTINUE MOVING ON SO",
    "3PYrtkzrMqu-00043-00037846-00039774.wav": "IS SECANT X TANGENT X MINUS HIGH D LOW WOULD BE ONE OVER LOW SQUARED SO IF I CLEAN",
    "idIrEq4X9HY-00017-00019104-00020301.wav": "SOME STEEL PLATE AND BENT IT AND SEEMS TO WORK PRETTY GOOD AND MORE OF THOSE",
    "EsP0yM0TK9c-00078-00074461-00075730.wav": "ALL RIGHT AND THEN ONE TWO THREE ONE TWO",
    "-FbjyIKzep8-00057-00034742-00036411.wav": "OKAY LET'S SEE THE RESULT OKAY THAT LINE NEEDS A LITTLE DARKENING BUT I THINK",
    "fevs-b2fuN8-00027-00030200-00032400.wav": "ONE OF THE THEMES THAT SEEMS TO BE SHOWING UP IN THE CNI SYNCHRONOUS SESSIONS THAT WE HAVE GOING THIS WEEK IS THE ONE OF SIX SUSTAINABILITY WE HAVE A LOT OF PROJECTS WHO ARE LOOKING AT PATHWAYS TO GENUINE SUSTAINABILITY AND THOSE INVOLVED TYPICALLY UM",
    "n1dzetzSEC4-00027-00018488-00019512.wav": "RESTOCKED SPENT THE NIGHT AND THEN WENDED OUR WAY SOUTH TO THE BORDER WITH MALI NOW IT",
    "j7UUgMJOAgk-00025-00019839-00020994.wav": "SIGMA AND TWO PI TWO OR THE THREE THE TRIPLE BOND AUCTION HAS ONE SIGMA AND PI A TOTAL OF THE DOUBLE BONDS HYDROGEN HAS ONE SIGNAL ON",
    "SJV-LD4EC14-00059-00057544-00058591.wav": "JUST TO BE AWARE THAT YOU CAN ADD SPACE USING GITHUB AND LET'S AND GET ON WITH MAKING YOUR FIRST REPOSITORY SO WHAT I MEAN BY REPOSITORY",
    "iTsN4gYDQiI-00039-00050646-00052003.wav": "I'VE WORKED WITH THE THE TOURING WAY I'VE WORKED WITH OS AND THEY ARE ALL WONDERFUL WONDERFUL COMMUNITIES THAT HAVE THE TOPIC OF WELL BEING AND SELF CARE",
    "BSojiZdoneI-00140-00085744-00086984.wav": "TRY TO GET THE NOTE TO RING OUT NICELY AND IT WILL ALMOST BE A CHORD MELODY ARRANGEMENT OF THIS SONG",
    "LdHusFnwlQ0-00008-00017936-00019939.wav": "THE ENGAGEMENT OF PERSONS WITH DISABILITIES FOR MONITORING AND THE INTER GOVERNMENTAL COORDINATION FOR IMPLEMENTATION WAS DEVELOPED TO SUPPORT HOLISTIC APPROACH OF POST EARTHQUAKE REFORM",
    "R5folQ3Zdsc-00006-00003108-00004202.wav": "BEAUTIFUL RAZOR NICE WEDGE HORN SCALES BEAUTIFUL CHECK THIS OUT ISN'T THAT MARVELOUS",
    "iTsN4gYDQiI-00108-00115510-00118159.wav": "AND ONLY WHEN I REALIZED THAT THAT WAS THAT WAS THAT THAT WAS THE UNDERLYING ASSUMPTION DID IT DID I REALLY COME TO THIS CONCLUSION THAT I CAN'T IT'S JUST NOT POSSIBLE NO MATTER HOW GOOD I MANAGE MY TIME HOW EFFECTIVE I GET OR HOW WHEN HOW MANY WINDOWS I HAVE OPEN I COULD JUST FOLLOW ALONG AND SAVE STUFF IN MY YOUTUBE WATCH LATER AND THAT WAS A REALLY REALLY HELPFUL",
    "iTsN4gYDQiI-00205-00193293-00194427.wav": "YEAH AND ALSO KIND OF YEAH THEY CONSTRUCT WHAT IS IMPLICIT AGAIN COMING BACK TO COMMUNITIES I THINK WHAT WAS FOR ME USEFUL TO OBSERVE AND THEN LEARN IS THAT",
    "dCCdLM7IH-c-00012-00010200-00011400.wav": "COUNCIL MEMBERS PRAY FOR US TO NOTICE THEM AND TO NOTICE THE VALUE OF THEIR GIFTS OF GUIDANCE",
    "Aqu2m6oB1ec-00081-00058348-00059652.wav": "BECAUSE BY THE FINAL SQUARE OF THE CHESSBOARD THE DEBT IS EIGHTEEN MILLION BILLION TRILLION GRAINS OF RICE",
    "ZLNLj7eM27u-00047-00035130-00036210.wav": "AND THERE FEED COMES OUT AND YOU CAN SEE THERE YOU'LL SEE ALL THAT PLATING HAS COME OFF",
    "l6KMeTvfkLA-00283-00238512-00239584.wav": "I FEEL LIKE IT WAS LIKE EIGHTEEN BEFORE SO I'LL TAKE THAT AS A WIN",
    "y62q2Roz78u-00419-00265864-00267584.wav": "WRITE AND EXECUTE TO RUN THE FILE DIRECTLY WE WANT OUR FILE TO BE EXECUTABLE THIS IS HOW WE DO IT",
    "fevs-b2fuN8-00257-00270400-00271400.wav": "SO THAT'S THAT'S REALLY THAT CHALLENGE THERE IS TRYING TO FIGURE OUT HOW TO HOW TO WORK IN THE CONTEXT OF FIGURING OUT SUSTAINABILITY WHILE YOU'RE STILL KIND OF JIGGLING AROUND",
    "j7UUgMJOAgk-00038-00029529-00030813.wav": "WOULD INVOLVE FOR CARBON ATOMS SO WE HAVE BEEN SEPARATED BY A SINGLE BUTTON THAT'S OUR SMALLEST CONJUGATED HOW KING SO ONE THREE JUDAH DYING IS ONE OF THESE COMPOUNDS",
    "m-Nsxr5PcYU-00119-00090232-00091232.wav": "VECTORS SO WHAT I'M GOING TO DO INITIALLY IS I'M GOING TO CREATE A NEW VECTOR VECTOR CALLED SNP GENES",
    "m3rJs0Q8bPQ-00089-00083070-00085331.wav": "IS AN EXAMPLE OF TWITTER YOU DO THE CONTROL V YOU PUT YOUR TWITTER LINK YOU CLICK FETCH",
    "iTsN4gYDQiI-00432-00365352-00366618.wav": "SEEN AS VERY COOL TO BE SUPER BUSY AND NOT HAVE TIME FOR YOU KNOW NOT NOT HAVE ANY FREE TIME WHEN I STARTED MY CAREER AND I THINK WE'RE NOW",
    "VhPdG7wMqNk-00078-00074900-00077000.wav": "APPROPRIATE RESEARCH METHODOLOGIES SO IF A LIBRARY WERE TO COMMIT TO UNDERTAKING THIS TYPE OF RESEARCH LONG TERM WE WOULD RECOMMEND FULLY INVESTING IN A NECESSARY STAFFING EXPERTISE AND SKILLS TO DEVELOPMENT SO THANKS SO MUCH TO ARL FOR THIS OPPORTUNITY AND WE WILL ANSWER YOUR QUESTIONS DURING THE Q AND A",
    "Gdx6Vx2S3d8-00017-00008713-00009713.wav": "WE HOPE YOU ENJOYED THAT IMDC INFORMATIONAL MINUTE THANKS FOR WATCHING AND REMEMBER THE IMDC IS LOCATED IN ANDRUSS LIBRARY IN ROOM TWO HUNDRED AND SIX",
    "EsP0yM0TK9c-00256-00265874-00266976.wav": "NEXT TIME WE'LL SLAP THAT MASK IN THERE I GUESS RUN TO THE HALL OF PLEASURE WHY WOULD",
    "-2V5jdMxRMo-00061-00051522-00052700.wav": "THAT INITIAL SLOPE DOWN TO Y OVER TWO OR POINT FIVE THE NEXT STEP WILL GO TO POINT TWO FIVE THE NEXT STEP",
    "XZ1f89-ecUA-00056-00065805-00066873.wav": "THIS INCLUDES PARTICIPATION FROM SMALL ACADEMIC LIBRARIES WHO MIGHT NOT OTHERWISE HAVE THE RESOURCES OR OPPORTUNITY TO PARTICIPATE IN LARGE SCALE DIGITIZATION PROJECTS BUT ALSO AS AS",
    "vKHDcopIhd0-00260-00097618-00099060.wav": "WHEN YOU SEE THAT THERE ARE NOT OTHERS TO ANALYZE THAT THERE ARE NOT EXTERNAL MOTIVES YOU FOCUS ON YOUR OWN MOTIVE YOUR OWN DESIRE FOR PEACE",
    "xXADsgcd-2c-00332-00276314-00277852.wav": "RAP SOMETHING YOU KNOW YOU HAVE SOME REALLY COMPLEX UNDERLYING LIBRARY THAT YOU WANT TO USE AND AND YOU WANT TO MAKE IT JUST A LITTLE SIMPLER FOR MAKE THE EASY STUFF EASY NOW I KNOW",
    "uv_A3Gn6eDI-00138-00134838-00137400.wav": "THINK I SAID EARLIER ICE IS NOT EFFECTIVE AGAINST ROCK SO IT IS EFFECTIVE AGAINST GROUND RIGHT",
    "4yY9E0fVdXM-00420-00236450-00237469.wav": "SO WITH ONLINE VIDEO SHORTER IS BETTER IN SORT OF VIDEO EXPERIENCE",
    "vOnJ9dm3ZhQ-00046-00029602-00030672.wav": "TO MEN AND WOMEN WHO WANT TO HAVE CHILDREN BUT ARE NOT ABLE THE PRESSURE CAN BE CRUSHING AND",
    "lQpOwSzoIro-00002-00002494-00003660.wav": "DETERMINE WHAT THE PATTERN IS THERE IS MORE THAN ONE WAY TO DESCRIBE THE",
    "ZmSflzkqIUA-00303-00211512-00212567.wav": "FOR THIS PROJECT INTO THIS CLUSTER AND I WILL TAKE A QUICK LOOK AT THEM BUT BEFORE I DO THAT",
    "VhPdG7wMqNk-00021-00017900-00019300.wav": "NEXT SLIDE PLEASE SINCE THEY PICTURE IS WORTH A ONE THOUSAND WORDS I'M GONNA TRY TO SHOW YOU BEFORE AND AFTER SHOTS OF WHAT THESE SPACES LOOKED LIKE TO MAKE THEM A LITTLE BIT MORE REAL AND GIVE A SENSE OF WHAT THE COMMUNITIES ACTUALLY LOOK",
    "LdHusFnwlQ0-00014-00029986-00031986.wav": "HARDSHIP HE IS SUPPORTING THOUSANDS OF PEOPLE BY HIS ENGAGEMENT IN SUPPORTING PERSONS WITH DISABILITIES BEING ASSIGNED BY ADRAD",
    "Kr5LECV3cXu-00845-00372798-00374053.wav": "TALKS ABOUT THE WAY IN WHICH DIFFERENT FORMS OF MONEY WERE PUSHED OUT THROUGH HISTORICAL EVOLUTION",
    "1mNa-aOTQQ4-00018-00020364-00021537.wav": "COME ON MAN WHERE Y'ALL SITTING WHERE Y'ALL SITTING MAN LING MEDICAID WILL",
    "pZl0IgYswbM-00074-00029680-00031183.wav": "LESS THAN TWO WEEKS HOPE THAT HELPS SMASH THAT LIKE BUTTON",
    "x_yvZ70dZEy-00023-00011403-00012646.wav": "BUT WHAT WE'RE GOING TO GET IS LOG TO THE BASE THREE OF NINE MINUS LOG TO THE BASE THREE OF THREE",
    "GYZ9K9Bxzey-00035-00018408-00019465.wav": "IT HAS ONE HUNDRED AND SIXTY TWO PEOPLE EIGHTY ANIMALS ANGELS AND ABOUT FOUR HUNDRED AND FIFTY SMALLER OBJECTS",
    "iTsN4gYDQiI-00329-00292533-00293576.wav": "I WASN'T AS AS CONSISTENT WITH IT FOR MYSELF BUT IN NOVEMBER OR OCTOBER LAST YEAR I SAID OKAY I'M GONNA DO THIS I'M GONNA TRY AND BE BETTER AT IT",
    "SJV-LD4EC14-00075-00071248-00072348.wav": "SO JUST CONSIDER THAT BUT IF IT'S YOUR OWN WORK IF YOU'RE DOING YOUR OWN PHD I THINK IT'S ABSOLUTELY FINE FOR IT TO BE BIT TO BE OPEN AND THAT'S JUST MY OPINION THOUGH",
    "u72Fqq4AUBE-00003-00002632-00004120.wav": "ONE TWO THREE FOUR FIVE SIX AND SEVEN SORRY AND EIGHT AND NINE OH THERE ARE HOURS ONE",
    "m7vABybvvGM-00004-00002043-00003345.wav": "WHO MIGHT BE STRUGGLING WITH WEIGHT ISSUES WHAT'S UP EVERYBODY",
    "xXADsgcd-2c-00290-00235561-00237076.wav": "UM EVERYTHING IS EASY UNTIL YOU GET TO THE PART WHERE YOU NEED TO MAKE THAT ONE WORD BOLD IN THE MIDDLE OF THE SENTENCE AND THEN YOU HAVE TO MAKE SOME DECISIONS ABOUT HOW YOU WANT TO SHOW THAT THE REST OF IT'S ALL PRETTY STRAIGHTFORWARD IT REALLY IS",
    "o4dZFc0zYpE-00008-00003022-00004061.wav": "AND BULLISH BAR FOLLOWED BY A BEARISH BAR LOCATED AT THE TOP",
    "P3bNTQbDJyE-00019-00015632-00016672.wav": "STOCKFISH LOVES NINETY SEVEN KNIGHT F FIVE YOU CAN PLAY LET'S SEE THE POSITION STOCKFISH FIFTEEN VERSUS STOCKFISH FIFTEEN",
    "i2MbB4Fmc_A-00092-00095300-00096300.wav": "PRACTITIONERS CAN HELP COLLEAGUES ARTICULATE WHAT OUTCOMES THEY HOPE TO CONTRIBUTE TO LIBRARY WORKERS CAN ALSO BE MORE MINDFUL ABOUT THE OUTCOMES THAT THE LIBRARY WOULD LIKE TO CONTRIBUTE TO",
    "x-hSfjB4x_y-00223-00157000-00158216.wav": "SUPER INTENDENTS LIKE THE OFFICES OF THE GENERAL OR SUBSECTIONS OF THAT OR INSPECTORS OF",
    "ejB39NMNzLY-00031-00024342-00025374.wav": "I IT'S SO EASY IT'S JUST FREQUENCIES I ALWAYS EXPERIENCE THIS AS AN INNER BATH OR INNER",
    "EsP0yM0TK9c-00105-00097550-00099712.wav": "HEY THAT'S NOT HOW WE'RE FIGHTING THIS A DREAM THIS IS A DREAM I DON'T WANT TO DIE",
    "-Qydq5EtDOc-00285-00299310-00300414.wav": "I'VE ACTUALLY CONVENED WORKSHOPS WHERE IT'S ACTUALLY MATH PROFESSORS PRESENTING AND THEY ARE THE FIRST TO SAY THAT SOMETIMES YOU JUST NEED A BREAK WHEN YOU'RE DOING QUANTITATIVE WORK",
    "_0azrnupduM-00160-00165000-00166400.wav": "UM WHICH OF THESE SHOULD BE PROCESSED FIRST AND WITHIN WHAT TIMEFRAME WE'VE SUBMITTED A PAPER ON THIS TO THIS YEAR'S IPRES SO IF IT GETS ACCEPTED YOU WILL PROBABLY HEAR MORE ABOUT IT BUT",
    "XZ1f89-ecUA-00007-00015516-00016536.wav": "I'M THE DIRECTOR OF CNI CLIFF LYNCH AND I'LL BE INTRODUCING THIS RATHER BRIEFLY A COUPLE OF LOGISTICAL THINGS",
    "idIrEq4X9HY-00025-00026394-00027854.wav": "THERE WE GO HERE'S MY BRISKET IT'S COOKING GOOD",
    "sfu8nxhF9so-00073-00056488-00057488.wav": "WOMEN THEMSELVES ARE GOING TO BOTOX CLINICS TO BASICALLY CANCEL THEIR OWN FACIAL EXPRESSIONS",
    "xwBI1bwxZOY-00129-00043669-00044776.wav": "AND I WANT TO SUBTRACT TEN POINT FIVE AND Q THREE",
    "xXADsgcd-2c-00074-00057629-00059596.wav": "HOWEVER THERE IS A A LIBRARY DOTNET CALLED MICRO DOC THAT WAS WRITTEN LIKE TWENTY YEARS AGO LIKE IT'S SERIOUSLY THAT OLD AND IT GIVES YOU A SORT OF A WORD PROCESSOR LIKE FEEL TO GENERATE A PDF IT ACTUALLY GENERATES ALL SORTS OF YOU CAN GENERATE OTHER STUFF TOO BUT",
    "4XgJcRAQDIg-00280-00188128-00189224.wav": "YEAH WE HAVE TO BUILD BUILD THOSE IN WHEN THEY DON'T EXIST",
    "Ihwr_fevmwo-00216-00170310-00171678.wav": "I THINK THAT'S A GOOD START AND KNOWING HOW THE OTHER WORKING GROUP STARTED HAVING A LIST OF FIVE ITEMS IN A FOCUS AREA IT'S A GOOD CONTRIBUTION TO THE REPOSITORY AND BUT STARTING PLACE",
    "Ts9uYB2ZfRA-00001-00000696-00001984.wav": "EGGS AND HAM BY THE INFAMOUS DR SEUSS OKAY LET US BEGIN HERE GREEN EGGS AND HAM I AM SAM",
    "lzEsx1EMC1g-00125-00114496-00116383.wav": "AND I'LL SAVE THAT LITTLE NAME PLATE I'M SURE I CAN USE THIS IN ANOTHER LITTLE PROJECT SOMEWHERE",
    "xtBcPuJ02rI-00041-00030276-00031347.wav": "TO HAVE ON HEALTH AT THIS LEVEL OF CARE AND WITH SUCH IMPACT FOR SUCH SMALL INVESTMENTS",
    "fOxaq-9aUb4-00593-00338594-00339637.wav": "THAT'S ONE PIECE OF ADVICE THAT YOU WOULD SHARE WITH OTHER LIBRARY STAFF",
    "-Qydq5EtDOc-00189-00200916-00202365.wav": "THAT'S A GOOD ONE TOO BUT WHAT IF YOU KNOW YOU YOU REALLY THE WHOLE NEXT WEEK IS FILLED UP WITH LIKE A TON OF URGENT THINGS YOU KNOW AND YOU'RE LIKE HOW CAN I I DON'T EVEN KNOW IF I'M GOING TO HAVE TIME FOR MY QUADRANT ONE THINGS HOW CAN I MAKE TIME FOR MY QUADRANT TWO THINGS",
    "uv_A3Gn6eDI-00175-00173759-00175379.wav": "MY MASSIVE MASCULATURE MISLED ME",
    "u72Fqq4AUBE-00000-00000162-00001249.wav": "QUESTION ONE OF TOPIC ONE THIS PROBLEM IS ABOUT A SURVEY AND IT'S ABOUT GET",
    "-Qydq5EtDOc-00112-00117693-00119571.wav": "YEAH AND YOU KNOW AND IF IF YOU'RE YOU KNOW TRYING TO STUDY FOR AN EXAM LAST MINUTE AND YOU'RE KIND OF MISERABLE AND YOU'RE REMEMBERING YOUR PAST FAILURES YOU KNOW THAT'S BRAIN POWER THAT YOU'RE NOT DEVOTING TO THE TASK AT HAND SO YOU KNOW YOUR FOCUS IS SACRIFICED AS WELL",
    "ulYpqCUJN5k-00987-00511242-00512484.wav": "LITTLE ABOUT THE STRATEGIES YOU USED OR YOU USE AIRBNB IS VERY OFTEN SEEN",
    "-Qydq5EtDOc-00254-00269325-00270507.wav": "THAT BIG WORKSHOP I WENT TO IT WAS SORT OF SUGGESTED THAT INSTEAD OF NECESSARILY MAKING REWARDS CONTINGENT JUST IF YOU KNOW YOU'RE GOING INTO A PERIOD WHERE YOU'RE GOING TO BE WORKING ON A LOT OF THINGS THAT ARE HARD FOR YOU",
    "sgwXrgXOicy-00047-00032866-00034163.wav": "WE CAN TAKE THE TABLE OF DISTANCE FALLEN AND GO FOR LONGER TIMES AND AND SO YOU SEE THE",
    "EsP0yM0TK9c-00156-00150795-00151941.wav": "TO KILL THESE THINGS BUT ALSO NOT KILL MYSELF OH SNIPER RIFLE HOLY SHIT",
    "7L4GmXPjLCy-00110-00061890-00062920.wav": "THE FIRST AND ONLY EVER AUDIT OF THE MET OFFICE'S HADLEY CENTER'S THE CLIMATE RESEARCH UNIT'S HADCRUT GLOBAL AVERAGE SURFACE TEMPERATURE DATA SET",
    "pEyd5AdCdKk-00528-00239822-00240870.wav": "HERE OUR TWISTPORT ECOSYSTEM ENABLES ZERO LOSS TRANSMISSION FROM OF THE RF SIGNAL FROM RADIO TO THE ANTENNA",
    "olEfAZKJu08-00016-00016369-00017972.wav": "HOWEVER THIS ISN'T THE FIRST TIME THE ROYALS HAVE TURNED TO THE FORMER SPICE GIRL FOR WARDROBE INSPIRATION MRS BECKHAM'S QUINCY TOTE IS THE DUCHESS OF CAMBRIDGE'S BAG OF CHOICE FOR WIMBLEDON HAVING CARRIED IT TO THE TENNIS CHAMPIONSHIPS ON THREE SEPARATE OCCASIONS",
    "uty0cU4W4TI-00035-00021987-00023074.wav": "SENSE OF RELIEF THAT THE BURNS NO LONGER ME IT'S LIKE UP TO SOURCE ENERGY TO REPLENISH",
    "-Qydq5EtDOc-00046-00049149-00050556.wav": "BUT THEN AT SOME POINT YOU BECOME AWARE OF IT AGAIN AND WHEN YOU BECOME AWARE OF IT AGAIN IT CAN GET WORSE BECAUSE YOU KNOW NOW YOU KNOW MAYBE IT'S THE NEXT DAY OR MAYBE IT'S SOME LATE AT NIGHT AND YOU'RE EXHAUSTED",
    "-Qydq5EtDOc-00034-00034407-00035475.wav": "AND THEN IT HAS TO DO WITH HOW YOU ASSESS THAT TASK THE NEXT STEP LIKE DO YOU FEEL GOOD OR DO YOU FEEL BAD WHAT DO YOU THINK ABOUT I SHOULD STUDY FOR THE EXAM DO YOU",
    "-FReKYSxT74-00001-00001203-00002223.wav": "WITH SKULKER I THINK YEAH LET'S GO LET'S GET IT GUYS YOU IN HERE WE GOT TO GO WE",
    "XZ1f89-ecUA-00142-00157551-00159084.wav": "IN HER IN HER REMARKS IS THAT WE ARE REALLY VERY INTERESTED IN EXPANDING INCLUSION WITHIN THE COLLECTION AND MAKING SURE THAT THE WIDE RANGE OF VOICES THAT ARE PART OF THE CATHOLIC COMMUNITY IN NORTH AMERICA ARE REPRESENTED WITHIN THE CATHOLIC NEWS ARCHIVES",
    "Ihwr_fevmwo-00014-00012600-00013713.wav": "TAKING A LOOK AT IT COULD BE A VARIETY OF THINGS BUT YOU KNOW IT COULD BE TAKING A LOOK AT LIKE A DATA SCHEMA AND ACTUALLY SEEING WHAT WAS AVAILABLE IN A SCHEMA AND WHAT YOU CAN AND CAN'T DO",
    "E9HF8izz8ak-00063-00041514-00042522.wav": "OH FROZEN FRUITS I USE MY SMOOTHIES I GOT TWO STRAWBERRIES TWO PEACHES",
    "4yY9E0fVdXM-00485-00278770-00280070.wav": "UM AND AS FAR AS SORT OF THE THE CHEAP AS FAR AS CHEAPER AUDIO CONFERENCING YOU KNOW FOR THE ORGANIZATION WE'VE DONE FOR OUR FREE WEBINARS",
    "IWS56NrfDP4-00052-00041680-00042816.wav": "THE RIGHT WORDS GRANT YOU ENTRANCE INTO KNOWING THE HOLY GHOST AS GOD IF YOU DON'T LISTEN",
    "dQp_cT_Gbey-00732-00457360-00458368.wav": "GO THROUGH JUST A TINY BIT OF THE OF THE HISTORY HERE THIS IS WHAT WAS SAID TO TO VOTERS IN NINETEEN SIXTY EIGHT",
    "UHBmdAHtEW4-00012-00009864-00011384.wav": "WHICH WE HAVE A VIDEO THAT IS PROVIDED AND I CAN CLICK ON PLAY TO PLAY THE HANDOFF REPORT",
    "iTsN4gYDQiI-00367-00320092-00321341.wav": "WE SHOULD ALSO TRY TO REMIND THEM THAT THERE ARE BLUE THINGS IN THEIR LIVES SO THERE ARE OTHER THINGS THAT WE VALUE IN THEM THAT WE KNOW THEY HAVE THAT SKILL FOR THAT EXPERTISE",
    "Tbks2pJBbdo-00361-00270840-00272016.wav": "IN PRODUCTION THEN YEAH AFTER THAT THOSE FINAL VCF FILES THEN WE STILL GET ANNOTATED WITH",
    "n1dzetzSEC4-00199-00122959-00124783.wav": "NO IT'S NOT NIGERIA IT'S NIGER SO NEXT TIME INTO NIGER PARK W LIONS HIPPOS AND PARAFFIN LAMPS",
    "IWS56NrfDP4-00138-00107056-00108104.wav": "WRONG WORDS KEEP YOU OUT YOU KNOW WHERE MOST OF THE CHURCH IS THEY'RE OUT IF RIGHT WORDS",
    "fevs-b2fuN8-00111-00115300-00116800.wav": "WE THOUGHT LIKE YOU KNOW THERE'S A LOT OF MEMBERSHIP FATIGUE OUT THERE AND ALL OF THE EXPERTS WE'RE TALKING TO ARE SORT OF WARNING US THAT IT'S REALLY DIFFICULT TO BECOME YET ANOTHER ORGANIZATION THAT YOU KNOW GOES HATTON AND IT ASKS FOR MEMBERSHIP FEES FROM A VARIETY",
    "3hOHMDxGDSg-00159-00150828-00152100.wav": "YOU KNOW SOMETIMES IT'S HARD TO USE A VPN I GUESS YOU KNOW YOU HAVE TO BE ON THE RIGHT YOU KNOW LIKE LINUX VERSION AND STUFF LIKE THAT I GUESS ARE THERE ANY LIMITATIONS THAT YOU CURRENTLY KIND OF FACE WITH",
    "m3rJs0Q8bPQ-00087-00080738-00081924.wav": "AND THIS IS WHAT A QUESTION WILL LOOK LIKE",
    "ZYVx2DwjQLU-00029-00021129-00022476.wav": "SO IN IN THAT EXAMPLE IT'S VERY NOTICEABLE WHEN THE BALL IS COMING STRAIGHT AT YOU WHEN I",
    "0Wbd9kQ7g-u-00003-00088544-00089802.wav": "GUILT CANNOT BE REAL FOR OURSELVES OR FOR ANYONE INSANE IDEAS HAVE NO REAL RELATIONSHIPS FOR THAT IS WHY THEY ARE INSANE",
    "XKT-UY6hjXY-00155-00096637-00097703.wav": "AND CHOOSE THOSE COLORS TO WORK WITH SO NOW WHAT I HAVE CHOSEN IS A SQUARED",
    "GyM7h_VUfS0-00008-00013688-00014949.wav": "MEGHAN WHO IS CURRENTLY INSTALLED IN NOTTINGHAM COTTAGE WITH PRINCE HARRY COMPLETED HER FIRST ROYAL ENGAGEMENT LAST WEEK IN THE WEST MIDLANDS",
    "K-ROB1hCESu-00695-00406696-00407816.wav": "WELL I COMPLETELY AGREE WITH THE IDEA THAT WE NEED SORT OF BRIDGE PEOPLE LIKE THE DATA GUIDES",
    "fevs-b2fuN8-00129-00137000-00138900.wav": "AND THERE'S YOU KNOW A LOT OF THINGS THAT OUR UNIVERSITY HAS BEEN ABLE TO PROVIDE FOR US THAT WE JUST WEREN'T ABLE OR MAYBE TO TRY TO START DOING LIKE TAXES AND YOU KNOW COLLECTING THE ACTUAL MONEY AND AND ALL OF THE THE OVERSIGHT THAT NEEDS TO GO INTO",
    "ROQ5k7sCN7I-00037-00039111-00040132.wav": "AS YOU MAY RECALL WITH A COMPOUND AND INEQUALITY YOU HAVE TO ONLY TAKE THE PART OF THE COMPOUND INEQUALITY",
    "JlxiAy9sAbY-00632-00304073-00305181.wav": "MECHANISMS FOR SUPPORTING THEIR COMMUNITIES NEED FOR CONNECTION PARTICULARLY AS THEY'VE BEEN",
    "XKT-UY6hjXY-00183-00125345-00126647.wav": "HERE TOO THAT WE CAN BRING A LITTLE OF ATTENTION TO MAYBE WHAT IF WE PUT",
    "iam5PuKoYZA-00138-00042346-00043697.wav": "THANK YOU FOR WATCHING THIS VIDEO AND HAVE A GOOD DAY",
    "m-Nsxr5PcYU-00121-00092008-00093560.wav": "AND I'M GOING TO PUT A FEW GENES HERE LET'S SAY O X T R A C T N THREE A R AND O P R M ONE",
    "O8_CzVPBq1g-00078-00068322-00069394.wav": "IT WAS QUITE ILLUMINATING TO FIND OUT THAT WE WE WE POSSESS ALL OF THE JUDGMENT AND FEAR THAT EXIST",
    "_0azrnupduM-00164-00170200-00172200.wav": "AND TO BE FAIR MOST OF THE FORENSIC PROCESSING THAT THAT WE DO IS QUITE ENERGY HUNGRY SO WE HAVE DEVELOPED A COUPLE OF AND WE'RE STILL DEVELOPING METHODS TO MINIMIZE THE ENERGY FOOTPRINT FROM FROM THE PROCESS",
    "J977VFMgKBY-00058-00040844-00042243.wav": "IS ENTIRELY IN YOUR HANDS THE CLEANLINESS OF THE FORK IS NOT ENTIRELY IN YOUR HANDS THIS IS TRUE UM",
    "fevs-b2fuN8-00080-00086100-00087100.wav": "WE'LL GIVE MORE DETAILS ABOUT THAT IN A FEW MINUTES BUT IT FELT KIND OF AWKWARD IN THE MIDDLE OF IT BECAUSE THE SUSTAINABILITY IS IS SOMEWHAT OF A RESEARCH PROJECT IN AND OF ITSELF AND WE WERE JUST TRYING TO FIGURE IT OUT",
    "taiuT-89IkI-00130-00084138-00086139.wav": "THE TOP B E V E R I D G E ABRAHAM LINCOLN AND HERE WE HAVE THE FOUR VOLUME SET",
    "yxFPnxqxqQu-00028-00039400-00041100.wav": "TODAY IN FACT AFTER ARRIVING TO THE SITE WE CAN SEE A HUGE AMOUNT OF WORK THAT IS BEING CARRIED OUT IN DIFFERENT AREAS THERE ARE SPECIALISTS ENGAGED IN CONCRETE WORKS SOME WORKERS ARE PREPARING THE FORMWORK THE TRUCK IS GETTING UNLOADED ACTUALLY CAN YOU TELL US IN MORE DETAIL WHAT KINDS OF WORK ARE BEING",
    "uVG4Y3Hm3Vu-00065-00050336-00051391.wav": "BE SPIRITUAL AND TO TUNE INTO WHATEVER YOUR BELIEF SYSTEM IS IS OKAY BECAUSE IT'S YOURS AND",
    "iTsN4gYDQiI-00032-00041878-00043460.wav": "AND THE RESIDENT FELLOWS ARE KIND OF A NEW EXPERIMENTAL THING I WAS KINDLY INVITED TO BE PART OF THE FIRST COHORT AND I STARTED THIS IN IN OCTOBER LAST YEAR",
    "QNJPFHSbvGk-00104-00084464-00085584.wav": "GIVE YOU AN OPPORTUNITY TO BE ABLE TO SEE WHAT YOU ARE THINKING WITHOUT USING YOUR KEYBOARD",
    "txeQSfXGy68-00069-00035427-00036632.wav": "MY THEORY IS IS LIKE WHEN WE'RE IN LOVE AND WE JUST SEE THAT A PERSON'S POTENTIAL AND WANT THEM TO BE THEIR BEST",
    "fevs-b2fuN8-00177-00189300-00190600.wav": "SO WE HAD A LOT OF REALLY STRONG LEARNING CURVE TO REALLY TACKLE SOME OF THE SUSTAINABILITY ISSUES YEAH MARKET ANALYSIS OKAY WHAT IS THAT AGAIN WITH THESE WERE JUST THINGS THAT YOU KNOW EVEN WITH THE CONSULTANT THEY WERE STILL HARD KIND OF IN THE SUMMARY OF THE PAPER CONTRACT",
    "v9tvvU9fFgy-00045-00026432-00027439.wav": "LOOK A BIT WEIRD BUT WE'LL COME BACK TO THE EYES LET'S START THE SKIN I AM LOW KEY BREAKING OUT",
    "XZ1f89-ecUA-00077-00084699-00086349.wav": "GREAT PARTNERSHIPS IN REGARDING GOVERNANCE RIGHT SO WE'VE REALLY BEEN ENGAGED NOT ONLY WITHIN CRA BUT ALSO ENGAGING ENGAGING SCHOLARS AND OUR PUBLISHING PARTNERS AND HELPING TO MAKE POLICY AND STRATEGIC DECISIONS STRATEGIC DIRECTION DECISIONS FOR THE ARCHIVES",
    "i2MbB4Fmc_A-00087-00090400-00092200.wav": "TO IN ANY CASE WHEN ASSESSMENT IMPACTS LIBRARY WORKERS JOB ROLES OR RESPONSIBILITIES THOSE CHANGES SHOULD BE ADEQUATELY RESOURCED SUPPORTED BY PROFESSIONAL DEVELOPMENT AND REWARDED ASSESSMENT CAN ALSO HELP LIBRARIES REALIZE",
    "SK5HEeyi14M-00179-00172181-00173381.wav": "WERE OUTPLAYED THE FSK WAS DEFINITELY OUTPLAYED BUT THAT WAS A GOOD SR PADDLE",
    "SzuxfcBJn6g-00014-00009544-00011384.wav": "OR SOMETHING LOGIC PRO X LET'S GO I DID FIND SOME PRETTY A DOPE SAMPLE AND SPLICE THAT I LIKE",
    "su79iUkFHnE-00043-00033684-00035024.wav": "THERE WE GO I'M ON MY LAPTOP NOW IT'S TIME TO MAKE THE HUEL SODA I STILL CAN'T",
    "uv_A3Gn6eDI-00230-00222031-00223792.wav": "SHE'S A DOCTOR RITA DOCTOR RITA",
    "w8jXvrLmwM4-00029-00027000-00028536.wav": "AND ACT FAST AND HERE IS THE REAL CHALLENGE WHAT I THINK YOUR YOUR DISCONTENT YOUR POSSIBLE I WOULD SAY LACK OF TRUE BUT CERTAINLY LOSING TRUST",
    "_0azrnupduM-00211-00226200-00227300.wav": "SO WE HAVE ANOTHER QUESTION FOR LEO AND EMMA SO HOW DID YOU DETERMINE WHAT KIND OF APPRAISAL CRITERIA AND OR METHODOLOGY TO USE IN THE PROCESSING PHASE",
    "FkQfXGqx-no-00047-00081668-00083380.wav": "LEON'S UNIQUE PERSONALITY AND SHARP WIT ENDEARED HIM TO FANS OF THE SHOW AS HE CONTRIBUTED FULLY TO GOGGLEBOX'S REPUTATION AS A PROGRAMME FULL OF WARM HUMOUR AND UNVARNISHED OPINION",
    "uNYK59pQaBg-00009-00015479-00017837.wav": "THE TWO THOUSAND SEVEN HUNDRED AND SIXTY ONE SQUARE FEET PROPERTY BOASTS THREE BEDROOMS STUNNING VIEWS OF BOTH THE HUDSON RIVER AND CENTRAL PARK ROUND THE CLOCK BUTLER SERVICE A POOL AND A PRIVATE MOVIE THEATRE WITH LEASES ON THE BUILDING AVERAGING AROUND TWENTY EIGHT THOUSAND DOLLARS A MONTH",
    "4yY9E0fVdXM-00577-00335420-00337040.wav": "UM IS NOT SO I GUESS SORT OF IN A BROAD SCOPE IDEALLY WE ALSO PROVIDE REPORTS WHICH ARE PDF FILES",
    "fevs-b2fuN8-00201-00214000-00215100.wav": "THAT THAT'S A GOOD THING I THINK YOU DO HAVE TO STAY TRUE TO LIKE YOUR YOUR VALUES AND AND WE VALUE OUR PEOPLE AND HOW OUR INDIVIDUALS FEEL ABOUT THE WORK THAT THEY'RE DOING WE COULD",
    "fevs-b2fuN8-00137-00144000-00145200.wav": "AMONGST OUR PARTNERS AND WE WANT TO CONTINUE TO YOU KNOW EXPAND ON THAT SO WE ARE CREATING THREE TIERS OF MEMBERSHIP RIGHT NOW ONE OF",
    "SJV-LD4EC14-00293-00252336-00253344.wav": "COVER DATA AND DOCUMENTATION SO I WOULD RECOMMEND THE CC BY FOUR POINT ZERO LICENSE BUT YOU ALSO NEED ONE IF YOU'RE GOING TO HAVE ANY CODE AS WELL",
    "iTsN4gYDQiI-00460-00389848-00390876.wav": "YEAH I DON'T KNOW I'D LIKE TO PASS THE MIC TO WHOEVER'S TAKING THE LONG THE LONGEST NOT TO SPEAK BUT I DON'T KNOW WHO THAT IS ANYMORE I THINK IT MIGHT BE CHRIS",
    "VhPdG7wMqNk-00057-00053800-00054800.wav": "SPACE FOR STUDENTS WHILE THESE CLEARLY WELCOME AND PARTICIPATE IN THESE COMMUNITIES THEY ALSO POINT OUT THE LOSS OF THIS OPEN SPACE AT THE SAME",
    "lSnOuePFLzY-00000-00001102-00002458.wav": "WE'LL SEE HIM GETTING OUT OF A TAXI IN A TUXEDO EMERGING FROM THE SEA IN SWIMMING TRUNKS FLIRTING WITH GLAMOROUS WOMEN AND TAKING DOWN CORRUPT MAFIA BOSSES",
    "aGwHs661YrI-00241-00152664-00153760.wav": "ME WHO CARES WHAT OTHER PEOPLE THINK WHO CARES WHAT THE WORLD THINKS WHO CARES WHAT KIND OF BOXES",
    "ZpGX4SprPTA-00006-00005712-00006720.wav": "SWITCH BACK TO THIS ONE POINT THREE PETROL MACHINE WITH ONLY ONE HUNDRED AND FORTY HORSEPOWER I WAS AFRAID IT WOULD",
    "O8_CzVPBq1g-00041-00034845-00036119.wav": "THIS FINALLY IS A SERIES CALLED JD'S LESBIAN UTOPIA AND IT STARTED IN NEW YORK AND WE TRAVELED",
    "qLZvCXRZOFy-00111-00055905-00056905.wav": "IT KEEPS YOU WALKING BY SIGHT MY STUFF MEANS I'M SAFE",
    "1QPFu-914pc-00148-00112656-00113728.wav": "OR DOWN DOESN'T REALLY MATTER BUT IT REALLY HELPS AND THAT JUST LEAVES IT FOR ME TO SAY THANK YOU",
    "ExAqRr_Jr10-00151-00099856-00100984.wav": "MY HAND UP SO I CAN'T SEE YOUR FACE OH MY GOD OH THERE'S LAYLA LAYLA IS LIKE WHAT THE HELL MANA",
    "_0azrnupduM-00136-00141700-00142800.wav": "THANK YOU OKAY SO ONE OF THE AREAS THAT WE'RE FOCUSING ON IN THE PILOT IS COLLECTIONS DEVELOPMENT WHICH YOU MIGHT REMEMBER COMES AT THE BEGINNING OF THE DIGITAL ARCHIVING WORKFLOW",
    "kNq2bP9uWUQ-00038-00028704-00029789.wav": "MY WORK RIGHT NOW REALLY CENTERS AROUND CURATION OF RESOURCES FOR INSTRUCTORS AND ALSO THINKING",
    "m3rJs0Q8bPQ-00036-00031464-00032931.wav": "THAT AND COME BACK OVER TO YOUR IN VIDEO AND CLICK COPY AND PASTE THIS INFORMATION IN ADD",
    "uo9f9QJPUJ8-00127-00118702-00119806.wav": "LIKE THIS THIS ONE IS THINNER SO THIS DRAWER ONLY FITS THE ONE OF THEIR ONLY FITS THEIR FILAMENT",
    "JlxiAy9sAbY-00292-00133243-00134313.wav": "WAS TALKING ABOUT HOW THEY WERE GOING CASHLESS AND HOW MUCH MONEY THE COUNTY IS SAVING ON BEING CASHLESS",
    "i2MbB4Fmc_A-00105-00109800-00111400.wav": "ACHIEVED HOORAY THAT'S EXCITING BUT IT'S ALSO NOT THE END ONE WAY TO EXTEND ACHIEVEMENT OF OUTCOMES FROM ONE SUCCESSFUL PROJECT TO A LARGER IMPACT IS TO SYSTEM SYSTEMATIZE THAT SUCCESS ONE CAN LOOK FOR LESSONS LEARNED THAT CAN",
    "Xj9uPAc2a68-00006-00004640-00005816.wav": "AGE OF TWENTY AFTER KING PHILLIP THE SECOND OF MACEDONIA WAS ASSASSINATED IN THREE THREE SIX BC ALEXANDER WENT ON TO RULE",
    "lzEsx1EMC1g-00157-00147312-00148584.wav": "YEAH LET'S JUST PUT A LITTLE BIT ON THERE WE JUST NEED SOMETHING GOING ON DON'T WE",
    "EhdI72IN7LM-00121-00058646-00060554.wav": "RESTOCK AND I GO TO THE BUILD THE BUILD FOLDER AND I NINJA INSTALL IT NOTHING VERY COMPLICATED",
    "PRup4lwrk28-00001-00000630-00001822.wav": "PERCENT LITERALLY MEANS PER ONE HUNDRED",
}  # noqa
