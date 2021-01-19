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
import csv
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
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
VALID_CSV = "valid.csv"
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


def prepare_voicebank(data_folder, save_folder, valid_speaker_count=2):
    """
    Prepares the csv files for the Voicebank dataset.

    Expects the data folder to be the same format as the output of
    ``download_vctk()`` below.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Voicebank dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    valid_speaker_count : int
        The number of validation speakers to use (out of 28 in train set).

    Example
    -------
    >>> data_folder = '/path/to/datasets/Voicebank'
    >>> save_folder = 'exp/Voicebank_exp'
    >>> prepare_voicebank(data_folder, save_folder)
    """

    # Setting ouput files
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_test = os.path.join(save_folder, TEST_CSV)
    save_csv_valid = os.path.join(save_folder, VALID_CSV)

    # Check if this phase is already done (if so, skip it)
    if skip(save_csv_train, save_csv_test, save_csv_valid):
        print("Preparation completed in previous run, skipping.")
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
    print("Creating csv files for noisy VoiceBank...")

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

    logger.debug("Creating csv files for noisy VoiceBank...")
    create_csv(
        wav_lst_train, save_csv_train, train_clean_folder, train_txts, lexicon
    )
    create_csv(
        wav_lst_valid, save_csv_valid, train_clean_folder, train_txts, lexicon
    )
    create_csv(
        wav_lst_test, save_csv_test, test_clean_folder, test_txts, lexicon
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

    lexicon = MISSING_LEXICON
    for line in open(lexicon_save_filepath):
        line = line.split()
        phns = " ".join(p.strip("012") for p in line[1:])
        lexicon[remove_punctuation(line[0])] = phns

    return lexicon


def create_csv(wav_lst, csv_file, clean_folder, txt_folder, lexicon):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files.
    csv_file : str
        The path of the output csv file
    clean_folder : str
        The location of parallel clean samples.
    txt_folder : str
        The location of the transcript files.
    """
    logger.debug(f"Creating csv lists in {csv_file}")

    csv_lines = [["ID", "duration"]]
    csv_lines[0].extend(["noisy_wav", "noisy_wav_format", "noisy_wav_opts"])
    csv_lines[0].extend(["clean_wav", "clean_wav_format", "clean_wav_opts"])
    csv_lines[0].extend(["wrd", "wrd_format", "wrd_opts"])
    csv_lines[0].extend(["phn", "phn_format", "phn_opts"])
    csv_lines[0].extend(["biphn", "biphn_format", "biphn_opts"])

    # Processing all the wav files in the list
    for wav_file in wav_lst:  # ex:p203_122.wav

        # Example wav_file: p232_001.wav
        snt_id = os.path.basename(wav_file).replace(".wav", "")
        clean_wav = os.path.join(clean_folder, snt_id + ".wav")

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Read text
        snt_id = os.path.basename(wav_file).replace(".wav", "")
        with open(os.path.join(txt_folder, snt_id + ".txt")) as f:
            words = f.read()
        words = remove_punctuation(words).strip().upper()
        phones = " ".join([lexicon[word] for word in words.split()])

        biphones = zip(["<B>"] + phones.split(), phones.split() + ["<E>"])
        biphones = [phn1 + phn2 for phn1, phn2 in biphones]

        # Composition of the csv_line
        csv_line = [snt_id, str(duration)]
        csv_line.extend([wav_file, "wav", ""])
        csv_line.extend([clean_wav, "wav", ""])
        csv_line.extend([words, "string", ""])
        csv_line.extend([phones, "string", ""])
        csv_line.extend([biphones, "string", ""])

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    print(f"{csv_file} successfully created!")


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
            print("Downloading " + url)
            with urllib.request.urlopen(url) as response:
                with open(filename, "wb") as tmp_file:
                    print("... to " + tmp_file.name)
                    shutil.copyfileobj(response, tmp_file)

    # Unzip
    for zip_file in zip_files:
        print("Unzipping " + zip_file)
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
        print("Resampling " + directory)
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

    print("Zipping " + final_dir)
    final_zip = shutil.make_archive(
        base_name=final_dir,
        format="zip",
        root_dir=os.path.dirname(final_dir),
        base_dir=os.path.basename(final_dir),
    )

    print(f"Moving {final_zip} to {destination}")
    shutil.move(final_zip, os.path.join(destination, dataset_name + ".zip"))
