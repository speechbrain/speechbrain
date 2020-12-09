"""
Data preparation.

Download: https://catalog.ldc.upenn.edu/LDC93S1

Authors
* Others
* Elena Rastorgueva 2020
"""

import os
import csv
import logging
from speechbrain.utils.data_utils import get_all_files

from speechbrain.data_io.data_io import (
    read_wav_soundfile,
    load_pkl,
    save_pkl,
    read_kaldi_lab,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_timit_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
SAMPLERATE = 16000


def prepare_timit(
    data_folder,
    splits,
    save_folder,
    kaldi_ali_tr=None,
    kaldi_ali_dev=None,
    kaldi_ali_test=None,
    kaldi_lab_opts=None,
    phn_set="39",
    uppercase=False,
):
    """
    repares the csv files for the TIMIT dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original TIMIT dataset is stored.
    splits : list
        List of splits to prepare from ['train', 'dev', 'test']
    save_folder : str
        The directory where to store the csv files.
    kaldi_ali_tr : dict, optional
        Default: 'None'
        When set, this is the directiory where the kaldi
        training alignments are stored.  They will be automatically converted
        into pkl for an easier use within speechbrain.
    kaldi_ali_dev : str, optional
        Default: 'None'
        When set, this is the path to directory where the
        kaldi dev alignments are stored.
    kaldi_ali_te : str, optional
        Default: 'None'
        When set, this is the path to the directory where the
        kaldi test alignments are stored.
    phn_set : {60, 48, 39}, optional,
        Default: 39
        The phoneme set to use in the phn label.
        It could be composed of 60, 48, or 39 phonemes.
    uppercase : bool, optional
        Default: False
        This option must be True when the TIMIT dataset
        is in the upper-case version.

    Example
    -------
    >>> from recipes.TIMIT.timit_prepare import prepare_timit
    >>> data_folder = 'datasets/TIMIT'
    >>> splits = ['train', 'dev', 'test']
    >>> save_folder = 'TIMIT_prepared'
    >>> prepare_timit(data_folder, splits, save_folder)
    """
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "kaldi_ali_tr": kaldi_ali_tr,
        "kaldi_ali_dev": kaldi_ali_dev,
        "kaldi_ali_test": kaldi_ali_test,
        "save_folder": save_folder,
        "phn_set": phn_set,
        "uppercase": uppercase,
    }

    # Getting speaker dictionary
    dev_spk, test_spk = _get_speaker()

    # Avoid calibration sentences
    avoid_sentences = ["sa1", "sa2"]

    # Setting file extension.
    extension = [".wav"]

    # Checking TIMIT_uppercase
    if uppercase:
        avoid_sentences = [item.upper() for item in avoid_sentences]
        extension = [item.upper() for item in extension]
        dev_spk = [item.upper() for item in dev_spk]
        test_spk = [item.upper() for item in test_spk]

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_opt = os.path.join(save_folder, OPT_FILE)
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_dev = os.path.join(save_folder, DEV_CSV)
    save_csv_test = os.path.join(save_folder, TEST_CSV)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.debug("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains TIMIT
    _check_timit_folders(uppercase, data_folder)

    msg = "\tCreating csv file for the TIMIT Dataset.."
    logger.debug(msg)

    # Creating csv file for training data
    if "train" in splits:

        # Checking TIMIT_uppercase
        if uppercase:
            match_lst = extension + ["TRAIN"]
        else:
            match_lst = extension + ["train"]

        wav_lst_train = get_all_files(
            data_folder, match_and=match_lst, exclude_or=avoid_sentences,
        )

        create_csv(
            wav_lst_train,
            save_csv_train,
            uppercase,
            data_folder,
            phn_set,
            kaldi_lab=kaldi_ali_tr,
            kaldi_lab_opts=kaldi_lab_opts,
        )

    # Creating csv file for dev data
    if "dev" in splits:

        # Checking TIMIT_uppercase
        if uppercase:
            match_lst = extension + ["TEST"]
        else:
            match_lst = extension + ["test"]

        wav_lst_dev = get_all_files(
            data_folder,
            match_and=match_lst,
            match_or=dev_spk,
            exclude_or=avoid_sentences,
        )

        create_csv(
            wav_lst_dev,
            save_csv_dev,
            uppercase,
            data_folder,
            phn_set,
            kaldi_lab=kaldi_ali_dev,
            kaldi_lab_opts=kaldi_lab_opts,
        )

    # Creating csv file for test data
    if "test" in splits:

        # Checking TIMIT_uppercase
        if uppercase:
            match_lst = extension + ["TEST"]
        else:
            match_lst = extension + ["test"]

        wav_lst_test = get_all_files(
            data_folder,
            match_and=match_lst,
            match_or=test_spk,
            exclude_or=avoid_sentences,
        )

        create_csv(
            wav_lst_test,
            save_csv_test,
            uppercase,
            data_folder,
            phn_set,
            kaldi_lab=kaldi_ali_test,
            kaldi_lab_opts=kaldi_lab_opts,
        )

    # saving options
    save_pkl(conf, save_opt)


def _get_phonemes():

    # This dictionary is used to conver the 60 phoneme set
    # into the 48 one
    from_60_to_48_phn = {}
    from_60_to_48_phn["sil"] = "sil"
    from_60_to_48_phn["aa"] = "aa"
    from_60_to_48_phn["ae"] = "ae"
    from_60_to_48_phn["ah"] = "ah"
    from_60_to_48_phn["ao"] = "ao"
    from_60_to_48_phn["aw"] = "aw"
    from_60_to_48_phn["ax"] = "ax"
    from_60_to_48_phn["ax-h"] = "ax"
    from_60_to_48_phn["axr"] = "er"
    from_60_to_48_phn["ay"] = "ay"
    from_60_to_48_phn["b"] = "b"
    from_60_to_48_phn["bcl"] = "vcl"
    from_60_to_48_phn["ch"] = "ch"
    from_60_to_48_phn["d"] = "d"
    from_60_to_48_phn["dcl"] = "vcl"
    from_60_to_48_phn["dh"] = "dh"
    from_60_to_48_phn["dx"] = "dx"
    from_60_to_48_phn["eh"] = "eh"
    from_60_to_48_phn["el"] = "el"
    from_60_to_48_phn["em"] = "m"
    from_60_to_48_phn["en"] = "en"
    from_60_to_48_phn["eng"] = "ng"
    from_60_to_48_phn["epi"] = "epi"
    from_60_to_48_phn["er"] = "er"
    from_60_to_48_phn["ey"] = "ey"
    from_60_to_48_phn["f"] = "f"
    from_60_to_48_phn["g"] = "g"
    from_60_to_48_phn["gcl"] = "vcl"
    from_60_to_48_phn["h#"] = "sil"
    from_60_to_48_phn["hh"] = "hh"
    from_60_to_48_phn["hv"] = "hh"
    from_60_to_48_phn["ih"] = "ih"
    from_60_to_48_phn["ix"] = "ix"
    from_60_to_48_phn["iy"] = "iy"
    from_60_to_48_phn["jh"] = "jh"
    from_60_to_48_phn["k"] = "k"
    from_60_to_48_phn["kcl"] = "cl"
    from_60_to_48_phn["l"] = "l"
    from_60_to_48_phn["m"] = "m"
    from_60_to_48_phn["n"] = "n"
    from_60_to_48_phn["ng"] = "ng"
    from_60_to_48_phn["nx"] = "n"
    from_60_to_48_phn["ow"] = "ow"
    from_60_to_48_phn["oy"] = "oy"
    from_60_to_48_phn["p"] = "p"
    from_60_to_48_phn["pau"] = "sil"
    from_60_to_48_phn["pcl"] = "cl"
    from_60_to_48_phn["q"] = ""
    from_60_to_48_phn["r"] = "r"
    from_60_to_48_phn["s"] = "s"
    from_60_to_48_phn["sh"] = "sh"
    from_60_to_48_phn["t"] = "t"
    from_60_to_48_phn["tcl"] = "cl"
    from_60_to_48_phn["th"] = "th"
    from_60_to_48_phn["uh"] = "uh"
    from_60_to_48_phn["uw"] = "uw"
    from_60_to_48_phn["ux"] = "uw"
    from_60_to_48_phn["v"] = "v"
    from_60_to_48_phn["w"] = "w"
    from_60_to_48_phn["y"] = "y"
    from_60_to_48_phn["z"] = "z"
    from_60_to_48_phn["zh"] = "zh"

    # This dictionary is used to conver the 60 phoneme set
    from_60_to_39_phn = {}
    from_60_to_39_phn["sil"] = "sil"
    from_60_to_39_phn["aa"] = "aa"
    from_60_to_39_phn["ae"] = "ae"
    from_60_to_39_phn["ah"] = "ah"
    from_60_to_39_phn["ao"] = "aa"
    from_60_to_39_phn["aw"] = "aw"
    from_60_to_39_phn["ax"] = "ah"
    from_60_to_39_phn["ax-h"] = "ah"
    from_60_to_39_phn["axr"] = "er"
    from_60_to_39_phn["ay"] = "ay"
    from_60_to_39_phn["b"] = "b"
    from_60_to_39_phn["bcl"] = "sil"
    from_60_to_39_phn["ch"] = "ch"
    from_60_to_39_phn["d"] = "d"
    from_60_to_39_phn["dcl"] = "sil"
    from_60_to_39_phn["dh"] = "dh"
    from_60_to_39_phn["dx"] = "dx"
    from_60_to_39_phn["eh"] = "eh"
    from_60_to_39_phn["el"] = "l"
    from_60_to_39_phn["em"] = "m"
    from_60_to_39_phn["en"] = "n"
    from_60_to_39_phn["eng"] = "ng"
    from_60_to_39_phn["epi"] = "sil"
    from_60_to_39_phn["er"] = "er"
    from_60_to_39_phn["ey"] = "ey"
    from_60_to_39_phn["f"] = "f"
    from_60_to_39_phn["g"] = "g"
    from_60_to_39_phn["gcl"] = "sil"
    from_60_to_39_phn["h#"] = "sil"
    from_60_to_39_phn["hh"] = "hh"
    from_60_to_39_phn["hv"] = "hh"
    from_60_to_39_phn["ih"] = "ih"
    from_60_to_39_phn["ix"] = "ih"
    from_60_to_39_phn["iy"] = "iy"
    from_60_to_39_phn["jh"] = "jh"
    from_60_to_39_phn["k"] = "k"
    from_60_to_39_phn["kcl"] = "sil"
    from_60_to_39_phn["l"] = "l"
    from_60_to_39_phn["m"] = "m"
    from_60_to_39_phn["ng"] = "ng"
    from_60_to_39_phn["n"] = "n"
    from_60_to_39_phn["nx"] = "n"
    from_60_to_39_phn["ow"] = "ow"
    from_60_to_39_phn["oy"] = "oy"
    from_60_to_39_phn["p"] = "p"
    from_60_to_39_phn["pau"] = "sil"
    from_60_to_39_phn["pcl"] = "sil"
    from_60_to_39_phn["q"] = ""
    from_60_to_39_phn["r"] = "r"
    from_60_to_39_phn["s"] = "s"
    from_60_to_39_phn["sh"] = "sh"
    from_60_to_39_phn["t"] = "t"
    from_60_to_39_phn["tcl"] = "sil"
    from_60_to_39_phn["th"] = "th"
    from_60_to_39_phn["uh"] = "uh"
    from_60_to_39_phn["uw"] = "uw"
    from_60_to_39_phn["ux"] = "uw"
    from_60_to_39_phn["v"] = "v"
    from_60_to_39_phn["w"] = "w"
    from_60_to_39_phn["y"] = "y"
    from_60_to_39_phn["z"] = "z"
    from_60_to_39_phn["zh"] = "sh"

    return from_60_to_48_phn, from_60_to_39_phn


def _get_speaker():

    # List of test speakers
    test_spk = [
        "fdhc0",
        "felc0",
        "fjlm0",
        "fmgd0",
        "fmld0",
        "fnlp0",
        "fpas0",
        "fpkt0",
        "mbpm0",
        "mcmj0",
        "mdab0",
        "mgrt0",
        "mjdh0",
        "mjln0",
        "mjmp0",
        "mklt0",
        "mlll0",
        "mlnt0",
        "mnjm0",
        "mpam0",
        "mtas1",
        "mtls0",
        "mwbt0",
        "mwew0",
    ]

    # List of dev speakers
    dev_spk = [
        "fadg0",
        "faks0",
        "fcal1",
        "fcmh0",
        "fdac1",
        "fdms0",
        "fdrw0",
        "fedw0",
        "fgjd0",
        "fjem0",
        "fjmg0",
        "fjsj0",
        "fkms0",
        "fmah0",
        "fmml0",
        "fnmr0",
        "frew0",
        "fsem0",
        "majc0",
        "mbdg0",
        "mbns0",
        "mbwm0",
        "mcsh0",
        "mdlf0",
        "mdls0",
        "mdvc0",
        "mers0",
        "mgjf0",
        "mglb0",
        "mgwt0",
        "mjar0",
        "mjfc0",
        "mjsw0",
        "mmdb1",
        "mmdm2",
        "mmjr0",
        "mmwh0",
        "mpdf0",
        "mrcs0",
        "mreb0",
        "mrjm4",
        "mrjr0",
        "mroa0",
        "mrtk0",
        "mrws1",
        "mtaa0",
        "mtdt0",
        "mteb0",
        "mthc0",
        "mwjg0",
    ]

    return dev_spk, test_spk


def skip(splits, save_folder, conf):
    """
    Detects if the timit data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
        "test": TEST_CSV,
    }
    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def create_csv(
    wav_lst,
    csv_file,
    uppercase,
    data_folder,
    phn_set,
    kaldi_lab=None,
    kaldi_lab_opts=None,
    kaldi_lab_dir=None,
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    uppercase : bool
        Whether this is the uppercase version of timit.
    data_folder : str
        The location of the data.
    kaldi_lab : str, optional
        Default: None
        The path of the kaldi labels (optional).
    kaldi_lab_opts : str, optional
        Default: None
        A string containing the options used to compute the labels.

    Returns
    -------
    None
    """

    # Adding some Prints
    msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    logger.debug(msg)

    # Reading kaldi labels if needed:
    snt_no_lab = 0
    missing_lab = False

    if kaldi_lab is not None:

        lab = read_kaldi_lab(kaldi_lab, kaldi_lab_opts,)

        if not os.path.exists(kaldi_lab_dir):
            os.makedirs(kaldi_lab_dir)

    csv_lines = [
        [
            "ID",
            "duration",
            "wav",
            "wav_format",
            "wav_opts",
            "spk_id",
            "spk_id_format",
            "spk_id_opts",
            "phn",
            "phn_format",
            "phn_opts",
            "wrd",
            "wrd_format",
            "wrd_opts",
            "ground_truth_phn_ends",
            "ground_truth_phn_ends_format",
            "ground_truth_phn_ends_opts",
        ]
    ]

    if kaldi_lab is not None:
        csv_lines[0].append("kaldi_lab")
        csv_lines[0].append("kaldi_lab_format")
        csv_lines[0].append("kaldi_lab_opts")

    # Processing all the wav files in the list
    for wav_file in wav_lst:

        # Getting sentence and speaker ids
        spk_id = wav_file.split("/")[-2]
        snt_id = wav_file.split("/")[-1].replace(".wav", "")
        snt_id = spk_id + "_" + snt_id

        if kaldi_lab is not None:
            if snt_id not in lab.keys():
                missing_lab = False
                msg = (
                    "\tThe sentence %s does not have a corresponding "
                    "kaldi label" % (snt_id)
                )

                logger.debug(msg)
                snt_no_lab = snt_no_lab + 1
            else:
                snt_lab_path = os.path.join(kaldi_lab_dir, snt_id + ".pkl")
                save_pkl(lab[snt_id], snt_lab_path)

            # If too many kaldi labels are missing rise an error
            if snt_no_lab / len(wav_lst) > 0.05:
                err_msg = (
                    "Too many sentences do not have the "
                    "corresponding kaldi label. Please check data and "
                    "kaldi labels (check %s and %s)." % (data_folder, kaldi_lab)
                )
                logger.error(err_msg, exc_info=True)

        if missing_lab:
            continue

        # Reading the signal (to retrieve duration in seconds)
        signal = read_wav_soundfile(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Retrieving words and check for uppercase
        if uppercase:
            wrd_file = wav_file.replace(".WAV", ".WRD")
        else:
            wrd_file = wav_file.replace(".wav", ".wrd")
        if not os.path.exists(os.path.dirname(wrd_file)):
            err_msg = "the wrd file %s does not exists!" % (wrd_file)
            raise FileNotFoundError(err_msg)

        words = [line.rstrip("\n").split(" ")[2] for line in open(wrd_file)]
        words = " ".join(words)

        # Retrieving phonemes
        if uppercase:
            phn_file = wav_file.replace(".WAV", ".PHN")
        else:
            phn_file = wav_file.replace(".wav", ".phn")

        if not os.path.exists(os.path.dirname(phn_file)):
            err_msg = "the wrd file %s does not exists!" % (phn_file)
            raise FileNotFoundError(err_msg)

        # Getting the phoneme and ground truth ends lists from the phn files
        phonemes, ends = get_phoneme_lists(phn_file, phn_set)

        # Composition of the csv_line
        csv_line = [
            snt_id,
            str(duration),
            wav_file,
            "wav",
            "",
            spk_id,
            "string",
            "",
            str(phonemes),
            "string",
            "",
            str(words),
            "string",
            "label:False",
            str(ends),
            "string",
            "label:False",
        ]

        if kaldi_lab is not None:
            csv_line.append(snt_lab_path)
            csv_line.append("pkl")
            csv_line.append("")

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    _write_csv(csv_lines, csv_file)
    msg = "\t%s sucessfully created!" % (csv_file)
    logger.debug(msg)


def get_phoneme_lists(phn_file, phn_set):
    """
    Reads the phn file and gets the phoneme list & ground truth ends list.
    """

    phonemes = []
    ends = []

    for line in open(phn_file):
        end, phoneme = line.rstrip("\n").replace("h#", "sil").split(" ")[1:]

        # Getting dictionaries for phoneme conversion
        from_60_to_48_phn, from_60_to_39_phn = _get_phonemes()

        # Removing end corresponding to q if phn set is not 61
        if phn_set != "61":
            if phoneme == "q":
                end = ""

        # Converting phns if necessary
        if phn_set == "48":
            phoneme = from_60_to_48_phn[phoneme]
        if phn_set == "39":
            phoneme = from_60_to_39_phn[phoneme]

        # Appending arrays
        if len(phoneme) > 0:
            phonemes.append(phoneme)
        if len(end) > 0:
            ends.append(end)

    if phn_set != "61":
        # Filtering out consecutive silences by applying a mask with `True` marking
        # which sils to remove
        # e.g.
        # phonemes          [  "a", "sil", "sil",  "sil",   "b"]
        # ends              [   1 ,    2 ,    3 ,     4 ,    5 ]
        # ---
        # create:
        # remove_sil_mask   [False,  True,  True,  False,  False]
        # ---
        # so end result is:
        # phonemes ["a", "sil", "b"]
        # ends     [  1,     4,   5]

        remove_sil_mask = [True if x == "sil" else False for x in phonemes]

        for i, val in enumerate(remove_sil_mask):
            if val is True:
                if i == len(remove_sil_mask) - 1:
                    remove_sil_mask[i] = False
                elif remove_sil_mask[i + 1] is False:
                    remove_sil_mask[i] = False

        phonemes = [
            phon for i, phon in enumerate(phonemes) if not remove_sil_mask[i]
        ]
        ends = [end for i, end in enumerate(ends) if not remove_sil_mask[i]]

    # Convert to e.g. "a sil b", "1 4 5"
    phonemes = " ".join(phonemes)
    ends = " ".join(ends)

    return phonemes, ends


def _write_csv(csv_lines, csv_file):
    """
    Writes on the specified csv_file the given csv_files.
    """
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)


def _check_timit_folders(uppercase, data_folder):
    """
    Check if the data folder actually contains the TIMIT dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain TIMIT dataset.
    """

    # Creating checking string wrt to lower or uppercase
    if uppercase:
        test_str = "/TEST/DR1"
        train_str = "/TRAIN/DR1"
    else:
        test_str = "/test/dr1"
        train_str = "/train/dr1"

    # Checking test/dr1
    if not os.path.exists(data_folder + test_str):
        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the TIMIT dataset)" % (data_folder + test_str)
        )
        raise FileNotFoundError(err_msg)

    # Checking train/dr1
    if not os.path.exists(data_folder + train_str):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the TIMIT dataset)" % (data_folder + train_str)
        )
        raise FileNotFoundError(err_msg)
