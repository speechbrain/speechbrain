# -*- coding: utf-8 -*-
"""
Data preparation.

Download and resample, use ``download_vctk`` below.
https://datashare.is.ed.ac.uk/handle/10283/2791

Authors:
 * Szu-Wei Fu, 2020
 * Peter Plantinga, 2020
"""

import os
import json
import string
import urllib
import shutil
import logging
import tempfile
import torchaudio
from torchaudio.transforms import Resample
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
LEXICON_URL = "http://www.openslr.org/resources/11/librispeech-lexicon.txt"
TRAIN_JSON = "train.json"
TEST_JSON = "test.json"
VALID_JSON = "valid.json"
SAMPLERATE = 16000
TRAIN_SPEAKERS = [
    "p226",
    "p287",
    "p227",
    "p228",
    "p230",
    "p231",
    "p233",
    "p236",
    "p239",
    "p243",
    "p244",
    "p250",
    "p254",
    "p256",
    "p258",
    "p259",
    "p267",
    "p268",
    "p269",
    "p270",
    "p273",
    "p274",
    "p276",
    "p277",
    "p278",
    "p279",
    "p282",
    "p286",
]
# Lexicon missing entries
MISSING_LEXICON = {
    "CRUCIALLY": "K R UW SH AH L IY",
    "PAEDOPHILES": "P EH D OW F AY L S",
    "MR": "M IH S T ER",
    "BBC": "B IY B IY S IY",
    "EUPHORIC": "Y UW F AO R IH K",
    "RACISM": "R EY S IH S M",
    "MP": "EH M P IY",
    "RESTRUCTURING": "R IY S T R AH K CH ER IH NG",
    "OSAMA": "OW S AH M AH",
    "GUITARIST": "G IH T AA R IH S T",
    "BLUESHE": "B L UW SH IY",
    "FLANKER": "F L AY N K ER",
    "SADDAM": "S AA D AA M",
    "COVERUP": "K UH V ER UH P",
    "FBI": "EH F B IY AY",
    "PREEMPTIVE": "P R IY EH M P T IH V",
    "FOURYEAR": "F AO R Y IY R",
    "XRAY": "EH K S R AY",
    "TALIBAN": "T AE L IH B AA N",
    "SUPERIMPOSITION": "S UW P ER IH M P OW S IH SH AH N",
    "GUIDELINES": "G AY D L AY N S",
    "FINALISED": "F AY N AH L AY Z D",
    "HALFTIME": "H AE F T AY M",
    "WINGERS": "W IH NG ER Z",
    "GM": "J IY EH M",
    "MCGREGOR": "M AH K G R EH G AO R",
    "TWODAY": "T UW D EY",
    "DATABASE": "D EY T AH B EY S",
    "TELECOM": "T EH L AH K AO M",
    "SHORTTERM": "SH AO R T ER M",
    "SHORTFALL": "SH AO R T F AH L",
    "MCCALL": "M AH K AH L",
    "HEADTEACHER": "H EH D T IY CH ER",
    "TAKEOVER": "T EY K OW V ER",
    "ONETHIRD": "W AH N TH ER D",
    "TV": "T IY V IY",
    "SCREENPLAY": "S K R IY N P L EY",
    "YUGOSLAV": "Y UW G OW S L AA V",
    "HIBS": "HH IH B Z",
    "DISPOSALS": "D IH S P OW S AH L Z",
    "MODERNISATION": "M AA D ER N AH Z EY SH AH N",
    "REALLIFE": "R IY L AY F",
    "ONEYEAR": "W AH N Y IY R",
    "GRASSROOTS": "G R AE S R UW T S",
    "ARNIE": "AH R N IY",
    "PARTTIME": "P AH R T AY M",
    "SHORTLIST": "SH AO R T L IH S T",
    "OUTPERFORMED": "OW T P ER F AO R M D",
    "LONGTERM": "L AO NG T ER M",
    "DAYTODAY": "D EY T UW D EY",
    "MCPHERSON": "M AH K F ER S AH N",
    "OUTSOURCING": "OW T S AO R S IH NG",
    "FULLSCALE": "F UH L S K EY L",
    "SERGIO": "S ER J IY OW",
    "HENMAN": "HH EH N M AA N",
    "MCLEOD": "M AH K L IY AO D",
    "TIMESCALE": "T AY M S K EY L",
    "REFURBISHMENT": "R IY F UH R B IH SH M AH N T",
    "LINEUP": "L AY N UH P",
    "DOWNBEAT": "D OW N B IY T",
    "MANDELA": "M AE N D EH L AH",
    "UNDERAGE": "UH N D ER EY J",
    "MCNAUGHTON": "M AH K N AW T AH N",
    "MICKELSON": "M IH K L S AH N",
    "THREEQUARTERS": "TH R IY K AO R T ER Z",
    "WEBSITE": "W EH B S AY T",
    "BLUEITS": "B L UW IH T S",
    "CEASEFIRE": "S IY S F AY R",
    "FULLTIME": "F UH L T AY M",
    "DOCHERTY": "D AH K ER T IY",
    "RUNNERUP": "R UH N ER AH P",
    "DOWNTURN": "D OW N T ER N",
    "EUROS": "Y ER OW S",
    "FOOTANDMOUTH": "F UH T AE N D M OW TH",
    "HIGHLIGHTED": "HH AY L AY T AH D",
    "MIDFIELD": "M IH D F IY L D",
    "MCKENZIE": "M AH K EH N Z IY",
    "BENCHMARK": "B EH N CH M AA R K",
    "MCCONNELL": "M AH K AW N EH L",
    "UPGRADING": "UH P G R EY D IH NG",
    "BLUNKETT": "B L UH N K AH T",
    "RETHINK": "R IY TH IH N K",
    "UPBEAT": "AH P B IY T",
    "TELECOMS": "T EH L AH K AO M Z",
    "APARTHEID": "AH P AH R T HH AY D",
    "AIRDRIE": "EY R D R IY",
    "RETHINK": "R IY TH IH N K",
    "HELPLINE": "HH EH L P L AY N",
    "CLEARCUT": "K L IY R K UH T",
}


def prepare_voicebank(
    data_folder, save_folder, valid_speaker_count=2, skip_prep=False
):
    """
    Prepares the json files for the Voicebank dataset.

    Expects the data folder to be the same format as the output of
    ``download_vctk()`` below.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Voicebank dataset is stored.
    save_folder : str
        The directory where to store the json files.
    valid_speaker_count : int
        The number of validation speakers to use (out of 28 in train set).
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> data_folder = '/path/to/datasets/Voicebank'
    >>> save_folder = 'exp/Voicebank_exp'
    >>> prepare_voicebank(data_folder, save_folder)
    """

    if skip_prep:
        return

    # Setting ouput files
    save_json_train = os.path.join(save_folder, TRAIN_JSON)
    save_json_valid = os.path.join(save_folder, VALID_JSON)
    save_json_test = os.path.join(save_folder, TEST_JSON)

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_test, save_json_valid):
        logger.info("Preparation completed in previous run, skipping.")
        return

    train_clean_folder = os.path.join(
        data_folder, "clean_trainset_28spk_wav_16k"
    )
    train_noisy_folder = os.path.join(
        data_folder, "noisy_trainset_28spk_wav_16k"
    )
    train_txts = os.path.join(data_folder, "trainset_28spk_txt")
    test_clean_folder = os.path.join(data_folder, "clean_testset_wav_16k")
    test_noisy_folder = os.path.join(data_folder, "noisy_testset_wav_16k")
    test_txts = os.path.join(data_folder, "testset_txt")

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Additional checks to make sure the data folder contains Voicebank
    check_voicebank_folders(
        train_clean_folder,
        train_noisy_folder,
        train_txts,
        test_clean_folder,
        test_noisy_folder,
        test_txts,
    )

    logger.debug("Creating lexicon...")
    lexicon = create_lexicon(os.path.join(data_folder, "lexicon.txt"))
    logger.info("Creating json files for noisy VoiceBank...")

    logger.debug("Collecting files...")
    extension = [".wav"]
    valid_speakers = TRAIN_SPEAKERS[:valid_speaker_count]
    wav_lst_train = get_all_files(
        train_noisy_folder, match_and=extension, exclude_or=valid_speakers,
    )
    wav_lst_valid = get_all_files(
        train_noisy_folder, match_and=extension, match_or=valid_speakers,
    )
    wav_lst_test = get_all_files(test_noisy_folder, match_and=extension)

    logger.debug("Creating json files for noisy VoiceBank...")
    create_json(
        wav_lst_train, save_json_train, train_clean_folder, train_txts, lexicon
    )
    create_json(
        wav_lst_valid, save_json_valid, train_clean_folder, train_txts, lexicon
    )
    create_json(
        wav_lst_test, save_json_test, test_clean_folder, test_txts, lexicon
    )


def skip(*filenames):
    """
    Detects if the Voicebank data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def remove_punctuation(a_string):
    """Remove all punctuation from string"""
    return a_string.translate(str.maketrans("", "", string.punctuation))


def create_lexicon(lexicon_save_filepath):
    """
    Creates the lexicon object, downloading if it hasn't been done yet.

    Arguments
    ---------
    lexicon_save_filepath : str
        Path to save the lexicon when downloading
    """
    if not os.path.isfile(lexicon_save_filepath):
        download_file(LEXICON_URL, lexicon_save_filepath)

    # Iterate lexicon file and add the first pronunciation in the file for
    # each word to our lexicon dictionary
    lexicon = MISSING_LEXICON
    delayed_words = {}
    for line in open(lexicon_save_filepath):
        line = line.split()
        phns = " ".join(p.strip("012") for p in line[1:])

        # Don't add words with punctuation until we can be sure they won't
        # overwrite words without punctuation.
        clean_word = remove_punctuation(line[0])
        if clean_word != line[0] and clean_word not in delayed_words:
            delayed_words[clean_word] = phns
        elif clean_word == line[0] and clean_word not in lexicon:
            lexicon[clean_word] = phns

    # Add words with punctuation if they won't overwrite non-punctuated words
    for word, phns in delayed_words.items():
        if word not in lexicon:
            lexicon[word] = phns

    return lexicon


def create_json(wav_lst, json_file, clean_folder, txt_folder, lexicon):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files.
    json_file : str
        The path of the output json file
    clean_folder : str
        The location of parallel clean samples.
    txt_folder : str
        The location of the transcript files.
    """
    logger.debug(f"Creating json lists in {json_file}")

    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in wav_lst:  # ex:p203_122.wav

        # Example wav_file: p232_001.wav
        noisy_path, filename = os.path.split(wav_file)
        _, noisy_dir = os.path.split(noisy_path)
        _, clean_dir = os.path.split(clean_folder)
        noisy_rel_path = os.path.join("{data_root}", noisy_dir, filename)
        clean_rel_path = os.path.join("{data_root}", clean_dir, filename)

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Read text
        snt_id = filename.replace(".wav", "")
        with open(os.path.join(txt_folder, snt_id + ".txt")) as f:
            word_string = f.read()
        word_string = remove_punctuation(word_string).strip().upper()
        phones = [
            phn for word in word_string.split() for phn in lexicon[word].split()
        ]

        # Remove duplicate phones
        phones = [i for i, j in zip(phones, phones[1:] + [None]) if i != j]
        phone_string = " ".join(phones)

        json_dict[snt_id] = {
            "noisy_wav": noisy_rel_path,
            "clean_wav": clean_rel_path,
            "length": duration,
            "words": word_string,
            "phones": phone_string,
        }

    # Writing the json lines
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def check_voicebank_folders(*folders):
    """Raises FileNotFoundError if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"the folder {folder} does not exist (it is expected in "
                "the Voicebank dataset)"
            )


def download_vctk(destination, tmp_dir=None, device="cpu"):
    """Download dataset and perform resample to 16000 Hz.

    Arguments
    ---------
    destination : str
        Place to put final zipped dataset.
    tmp_dir : str
        Location to store temporary files. Will use `tempfile` if not provided.
    device : str
        Passed directly to pytorch's ``.to()`` method. Used for resampling.
    """
    dataset_name = "noisy-vctk-16k"
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    final_dir = os.path.join(tmp_dir, dataset_name)

    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    if not os.path.isdir(final_dir):
        os.mkdir(final_dir)

    prefix = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/"
    noisy_vctk_urls = [
        prefix + "clean_testset_wav.zip",
        prefix + "noisy_testset_wav.zip",
        prefix + "testset_txt.zip",
        prefix + "clean_trainset_28spk_wav.zip",
        prefix + "noisy_trainset_28spk_wav.zip",
        prefix + "trainset_28spk_txt.zip",
    ]

    zip_files = []
    for url in noisy_vctk_urls:
        filename = os.path.join(tmp_dir, url.split("/")[-1])
        zip_files.append(filename)
        if not os.path.isfile(filename):
            logger.info("Downloading " + url)
            with urllib.request.urlopen(url) as response:
                with open(filename, "wb") as tmp_file:
                    logger.info("... to " + tmp_file.name)
                    shutil.copyfileobj(response, tmp_file)

    # Unzip
    for zip_file in zip_files:
        logger.info("Unzipping " + zip_file)
        shutil.unpack_archive(zip_file, tmp_dir, "zip")
        os.remove(zip_file)

    # Move transcripts to final dir
    shutil.move(os.path.join(tmp_dir, "testset_txt"), final_dir)
    shutil.move(os.path.join(tmp_dir, "trainset_28spk_txt"), final_dir)

    # Downsample
    dirs = [
        "noisy_testset_wav",
        "clean_testset_wav",
        "noisy_trainset_28spk_wav",
        "clean_trainset_28spk_wav",
    ]

    downsampler = Resample(orig_freq=48000, new_freq=16000)

    for directory in dirs:
        logger.info("Resampling " + directory)
        dirname = os.path.join(tmp_dir, directory)

        # Make directory to store downsampled files
        dirname_16k = os.path.join(final_dir, directory + "_16k")
        if not os.path.isdir(dirname_16k):
            os.mkdir(dirname_16k)

        # Load files and downsample
        for filename in get_all_files(dirname, match_and=[".wav"]):
            signal, rate = torchaudio.load(filename)
            downsampled_signal = downsampler(signal.view(1, -1).to(device))

            # Save downsampled file
            torchaudio.save(
                os.path.join(dirname_16k, filename[-12:]),
                downsampled_signal[0].cpu(),
                sample_rate=16000,
                channels_first=False,
            )

            # Remove old file
            os.remove(filename)

        # Remove old directory
        os.rmdir(dirname)

    logger.info("Zipping " + final_dir)
    final_zip = shutil.make_archive(
        base_name=final_dir,
        format="zip",
        root_dir=os.path.dirname(final_dir),
        base_dir=os.path.basename(final_dir),
    )

    logger.info(f"Moving {final_zip} to {destination}")
    shutil.move(final_zip, os.path.join(destination, dataset_name + ".zip"))
