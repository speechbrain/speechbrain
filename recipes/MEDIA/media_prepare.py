"""
Data preparation.
Download: See README.md

Author
------
Gaelle Laperriere 2021
"""

import xml.dom.minidom as DOM
from tqdm import tqdm
import logging
import subprocess
import csv
import os
import glob
import re

logger = logging.getLogger(__name__)


def prepare_media(
    data_folder,
    save_folder,
    skip_wav=True,
    method="slu",
    task="full",
    skip_prep=False,
):
    """
    Prepares the csv files for the MEDIA dataset.
    Both following repositories are nesseray for transcriptions
    and annotations (S0272) and audio (E0024).
    https://catalogue.elra.info/en-us/repository/browse/ELRA-S0272/
    https://catalogue.elra.info/en-us/repository/browse/ELRA-E0024/

    Arguments
    ---------
    data_folder: str
        Path where folders S0272 and E0024 are stored.
    save_folder: str
        Path where the csvs and preprocessed wavs will be stored.
    skip_wav: bool, optional
        Skip the wav files storing if already done before.
    method: str, optional
        Used only for 'slu' task.
        Either 'full' or 'relax'.
        'full' Keep specifiers in concepts.
        'relax' Remove specifiers from concepts.
    task: str, optional
        Either 'slu' or 'asr'.
        'slu' Parse SLU data.
        'asr' Parse ASR data.
    skip_prep: bool, optional
        If True, skip data preparation.

    """

    if skip_prep:
        return

    os.makedirs(save_folder)
    os.makedirs(save_folder + "/wav")
    os.makedirs(save_folder + "/csv")

    if task == "slu":
        logger.info("Processing SLU " + method + " Media Dataset")
    elif task == "asr":
        logger.info("Processing ASR Media Dataset")
    else:
        raise ValueError("Parameter task must be 'asr' or 'slu'")

    xmls = {
        "media_lot1.xml": "train",
        "media_lot2.xml": "train",
        "media_lot3.xml": "train",
        "media_lot4.xml": "train",
        "media_testHC.xml": "test",
        "media_testHC_a_blanc.xml": "dev",
    }

    wav_paths = glob.glob(data_folder + "/S0272/**/*.wav", recursive=True)
    channels, filenames = get_channels("./channels.csv")
    unused_dialogs = get_unused_dialogs(data_folder)
    concepts_full, concepts_relax = get_concepts_full_relax(
        "./concepts_full_relax.csv"
    )

    write_first_row(save_folder)

    # Wavs.
    if not (skip_wav):
        logger.info("Processing wavs")
        for wav_path in tqdm(wav_paths):
            filename = wav_path.split("/")[-1][:-4]
            channel = get_channel(filename, channels, filenames)
            split_audio_channels(wav_path, filename, channel, save_folder)

    # Train, Dev, Test.
    for xml in xmls:
        logger.info(
            "Processing file "
            + str(list(xmls.keys()).index(xml) + 1)
            + "/"
            + str(len(xmls))
        )
        root = get_root(
            data_folder + "/E0024/MEDIA1FR_00/MEDIA1FR/DATA/" + xml, 0,
        )
        parse(
            root, channels, filenames, save_folder, method, task, xmls[xml],
        )

    # Test2.
    logger.info("Processing files for test2")
    for filename in tqdm(unused_dialogs):
        root = get_root(
            data_folder
            + "/E0024/MEDIA1FR_00/MEDIA1FR/DATA/semantizer_files/"
            + filename
            + "_HC.xml",
            1,
        )
        parse_test2(
            root,
            channels,
            filenames,
            save_folder,
            method,
            task,
            filename,
            concepts_full,
            concepts_relax,
        )


def parse(
    root, channels, filenames, save_folder, method, task, corpus,
):
    """
    Parse data for the train, dev and test csv files of the Media dataset.
    Files are stored in MEDIA1FR_00/MEDIA1FR/DATA/.
    They are the original xml files used by the community for train, dev and test.

    Arguments
    ---------
    root: Document
        Object representing the content of the Media xml document being processed.
    channels: list of str
        Channels (Right / Left) of the stereo recording to keep.
    filenames: list of str
        Linked IDs of the recordings, for the channels to keep.
    save_folder: str
        Path where the csvs and preprocessed wavs will be stored.
    method: str
        Either 'full' or 'relax'.
    task: str
        Either 'asr' or 'slu'.
    corpus: str
        'train', 'dev' or 'test'.
    """

    for dialogue in tqdm(root.getElementsByTagName("dialogue")):

        speaker_name = get_speaker(dialogue)
        filename = dialogue.getAttribute("id")
        channel = get_channel(filename, channels, filenames)

        for turn in dialogue.getElementsByTagName("turn"):
            if turn.getAttribute("speaker") == "spk":

                time_beg = turn.getAttribute("startTime")
                time_end = turn.getAttribute("endTime")

                sentences = parse_sentences(
                    turn, time_beg, time_end, method, task
                )

                append_data(
                    save_folder,
                    channel,
                    filename,
                    speaker_name,
                    sentences,
                    corpus,
                )


def parse_test2(
    root,
    channels,
    filenames,
    save_folder,
    method,
    task,
    filename,
    concepts_full,
    concepts_relax,
):
    """
    This function prepares the data for the test2 csv files of the Media dataset.
    "Laperrière et al. The Spoken Language Understanding MEDIA Benchmark Dataset in the Era of Deep Learning: data updates, training and evaluation tools, LREC 2022" (https://aclanthology.org/2022.lrec-1.171) made the decision to make a new corpus named "test2".
    These xml files are structured differently from the original ones, explaining special functions for the test2.
    They are xml files made after the first dataset release, and have never been used before this recipe.
    This new corpus can be used as a second inference corpus, as the original test.
    Files are stored in /E0024/MEDIA1FR_00/MEDIA1FR/DATA/semantizer_files/.

    Arguments
    ---------
    root: Document
        Object representing the content of the Media xml document being processed.
    channels: list of str
        Channels (Right / Left) of the stereo recording to keep.
    filenames: list of str
        Linked IDs of the recordings, for the channels to keep.
    save_folder: str
        Path where the csvs and preprocessed wavs will be stored.
    method: str
        Either 'full' or 'relax'.
    task: str
        Either 'asr' or 'slu'.
    filename: str
        Name of the Media recording.
    concepts_full: list of str
        Concepts in method full.
    concepts_relax: list of str
        Concepts equivalent in method relax.
    """

    speaker_id, speaker_name = get_speaker_test2(root)
    channel = get_channel(filename, channels, filenames)

    for turn in root.getElementsByTagName("Turn"):
        if turn.getAttribute("speaker") == speaker_id:

            time_beg = turn.getAttribute("startTime")
            time_end = turn.getAttribute("endTime")

            sentences = parse_sentences_test2(
                turn,
                time_beg,
                time_end,
                method,
                concepts_full,
                concepts_relax,
                task,
            )

            if (
                filename == "70"
                and sentences[len(sentences) - 1][3] == "344.408"
            ):
                sentences[len(sentences) - 1][3] = "321.000"

            append_data(
                save_folder,
                channel,
                filename,
                speaker_name,
                sentences,
                "test2",
            )


def append_data(
    save_folder, channel, filename, speaker_name, sentences, corpus,
):
    """
    Make the csv corpora using data retrieved previously for one Media file.

    Arguments
    ---------
    save_folder: str
        Path where the csvs and preprocessed wavs will be stored.
    channel: str
        Channel (Right / Left) of the stereo recording to keep.
    filename: str
        Name of the Media recording.
    speaker_name: str
        Name of the speaker who said the given sentences.
    sentences: dictionnary of str
        Sentences previously parsed.
    corpus: str
        Either 'train', 'dev', 'test', or 'test2'.
    """

    data = []

    # Retrieve other necessary information
    out = subprocess.Popen(
        ["soxi", "-D", save_folder + "/wav/" + channel + filename + ".wav"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout, stderr = out.communicate()
    wav_duration = str("%.2f" % float(stdout))
    wav = save_folder + "/wav/" + channel + filename + ".wav"
    IDs = get_IDs(speaker_name, sentences, channel, filename)

    # Append data list
    for n in range(len(sentences)):
        f1 = float(sentences[n][3])
        f2 = float(sentences[n][2])
        duration = str("%.2f" % (f1 - f2))
        if (
            float(wav_duration) >= f1
            and float(duration) != 0.0
            and sentences[n][0] != ""
        ):
            data.append(
                [
                    IDs[n],
                    duration,
                    sentences[n][2],
                    sentences[n][3],
                    wav,
                    "wav",
                    speaker_name,
                    "string",
                    sentences[n][0],
                    "string",
                    sentences[n][1],
                    "string",
                ]
            )

    # Write data
    if data is not None:
        path = save_folder + "/csv/" + corpus + ".csv"
        SB_file = open(path, "a")
        writer = csv.writer(SB_file, delimiter=",")
        writer.writerows(data)
        SB_file.close()


def parse_sentences(turn, time_beg, time_end, method, task):
    """
    Get the sentences spoken by the speaker (not the "Compère" aka Woz).

    Arguments:
    -------
    turn: list of Document
        The current turn node.
    time_beg: str
        Time (s) at the beginning of the turn.
    time_end: str
        Time (s) at the end of the turn.
    method: str
        Either 'full' or 'relax'.
    task: str
        Either 'asr' or 'slu'.

    Returns
    -------
    dictionnary of str
    """

    has_speech = False
    sentences = [["", "", time_beg, time_end]]
    concept_open = False
    sync_waiting = False
    time = None
    n = 0  # Number of segments in the turn

    # For each semAnnotation in the Turn
    for semAnnotation in turn.getElementsByTagName("semAnnotation"):
        # We only process HC
        if semAnnotation.getAttribute("withContext") == "false":
            # For each sem
            for sem in semAnnotation.getElementsByTagName("sem"):
                # Check concept
                concept = sem.getAttribute("concept")
                specif = sem.getAttribute("specif")
                if method == "full" and specif != "null":
                    concept += specif

                # For each transcription in the Turn
                for transcription in sem.getElementsByTagName("transcription"):
                    # Check for sync or text
                    for node in transcription.childNodes:

                        # Check transcription
                        if (
                            node.nodeType == node.TEXT_NODE
                            and node.data.replace("\n", "").replace(" ", "")
                            != ""
                        ):
                            (
                                sentences,
                                has_speech,
                                sync_waiting,
                                concept_open,
                            ) = process_text_node(
                                node,
                                sentences,
                                sync_waiting,
                                has_speech,
                                concept,
                                concept_open,
                                task,
                                n,
                                time_end,
                            )

                        # Check Sync times
                        if node.nodeName == "Sync":
                            (
                                sentences,
                                has_speech,
                                sync_waiting,
                                time,
                                n,
                            ) = process_sync_node(
                                node,
                                sentences,
                                sync_waiting,
                                has_speech,
                                concept_open,
                                task,
                                n,
                                time,
                                time_end,
                            )

                if task == "slu":
                    (
                        sentences,
                        concept,
                        concept_open,
                        has_speech,
                        sync_waiting,
                        n,
                    ) = process_semfin_node(
                        sentences,
                        sync_waiting,
                        has_speech,
                        concept,
                        concept_open,
                        n,
                        time,
                        time_end,
                    )

    sentences = clean_last_sentence(sentences)
    return sentences


def parse_sentences_test2(
    turn, time_beg, time_end, method, concepts_full, concepts_relax, task,
):
    """
    Get the sentences spoken by the speaker (not the "Compère" aka Woz).

    Arguments:
    -------
    turn: list of Document
        All the xml following nodes present in the turn.
    time_beg: str
        Time (s) at the beginning of the turn.
    time_end: str
        Time (s) at the end of the turn.
    method: str
        Either 'full' or 'relax'.
    concepts_full: list of str
        Concepts in method full.
    concepts_relax: list of str
        Concepts equivalent in method relax.
    task: str
        Either 'asr' or 'slu'.

    Returns
    -------
    dictionnary of str
    """

    sentences = [["", "", time_beg, time_end]]
    n = 0  # Number of segments in the turn
    concept = "null"
    has_speech = False
    concept_open = False
    sync_waiting = False
    time = None

    # For each node in the Turn
    for node in turn.childNodes:

        # Check concept
        if task == "slu" and node.nodeName == "SemDebut":
            concept = node.getAttribute("concept")
            if method == "relax":
                concept = get_concept_relax(
                    concept, concepts_full, concepts_relax
                )

        # Check transcription
        if (
            node.nodeType == node.TEXT_NODE
            and node.data.replace("\n", "") != ""
        ):
            (
                sentences,
                has_speech,
                sync_waiting,
                concept_open,
            ) = process_text_node(
                node,
                sentences,
                sync_waiting,
                has_speech,
                concept,
                concept_open,
                task,
                n,
                time_end,
            )

        # Save audio segment
        if task == "slu" and node.nodeName == "SemFin":
            (
                sentences,
                concept,
                concept_open,
                has_speech,
                sync_waiting,
                n,
            ) = process_semfin_node(
                sentences,
                sync_waiting,
                has_speech,
                concept,
                concept_open,
                n,
                time,
                time_end,
            )

        if node.nodeName == "Sync":
            sentences, has_speech, sync_waiting, time, n = process_sync_node(
                node,
                sentences,
                sync_waiting,
                has_speech,
                concept_open,
                task,
                n,
                time,
                time_end,
            )

    sentences = clean_last_sentence(sentences)

    return sentences


def process_text_node(
    node,
    sentences,
    sync_waiting,
    has_speech,
    concept,
    concept_open,
    task,
    n,
    time_end,
):
    # Add a new concept, when speech following
    if task == "slu" and concept != "null" and not concept_open:
        sentences[n][0] += "<" + concept + "> "
        sentences[n][1] += "<" + concept + "> _ "
        concept_open = True
    sentence = normalize_sentence(node.data)
    sentences[n][0] += sentence + " "
    sentences[n][1] += " ".join(list(sentence.replace(" ", "_"))) + " _ "
    sentences[n][3] = time_end
    has_speech = True
    sync_waiting = False
    return sentences, has_speech, sync_waiting, concept_open


def process_sync_node(
    node,
    sentences,
    sync_waiting,
    has_speech,
    concept_open,
    task,
    n,
    time,
    time_end,
):
    # If the segment has no speech yet
    if not (has_speech):
        # Change time_beg for the last segment
        sentences[n][2] = node.getAttribute("time")
    # If the segment has speech and sync doesn't cut a concept
    elif task == "asr" or (task == "slu" and not concept_open):
        # Change time_end for the last segment
        sentences[n][3] = node.getAttribute("time")
        sentences.append(["", "", sentences[n][3], time_end])
        has_speech = False
        n += 1
    else:
        sync_waiting = True
        time = node.getAttribute("time")
    return sentences, has_speech, sync_waiting, time, n


def process_semfin_node(
    sentences,
    sync_waiting,
    has_speech,
    concept,
    concept_open,
    n,
    time,
    time_end,
):
    # Prevent adding a closing concept
    # If Sync followed by SemFin generate a new segment without speech yet
    if concept_open:
        sentences[n][0] += "> "
        sentences[n][1] += "> _ "
    concept = "null"  # Indicate there is no currently open concept
    concept_open = False
    if sync_waiting:
        sentences[n][3] = time
        sentences.append(["", "", time, time_end])
        has_speech = False
        sync_waiting = False
        n += 1
    return sentences, concept, concept_open, has_speech, sync_waiting, n


def clean_last_sentence(sentences):
    for n in range(len(sentences)):
        if sentences[n][0] != "":
            sentences[n][0] = sentences[n][0][:-1]  # Remove last ' '
            sentences[n][1] = sentences[n][1][:-3]  # Remove last ' _ '
        else:
            del sentences[n]  # Usefull for last appended segment
    return sentences


def normalize_sentence(sentence):
    # Apostrophes
    sentence = sentence.replace(" '", "'")  # Join apostrophe to previous word
    sentence = sentence.replace("'", "' ")  # Detach apostrophe to next word
    # Specific errors
    sentence = sentence.replace("gc'est", "c'est")
    sentence = sentence.replace("a-t- il", "a-t-il")
    sentence = sentence.replace("' un parking", "un parking")
    sentence = sentence.replace("bleu marine", "bleu-marine")
    sentence = sentence.replace("Saint-jacques", "Saint-Jacques")
    sentence = sentence.replace("Mont-de-Marsan", "Mont-De-Marsan")
    sentence = sentence.replace("Mont de Marsan", "Mont-De-Marsan")
    # Particular characters
    sentence = re.sub(r"^'", "", sentence)
    sentence = re.sub(r"\(.*?\)", "*", sentence)  # Replace (...) with *
    sentence = re.sub(r"[^\w\s'-><_]", "", sentence)  # Punct. except '-><_
    # Numbers correction
    sentence = sentence.replace("dix-", "dix ")
    sentence = sentence.replace("vingt-", "vingt ")
    sentence = sentence.replace("trente-", "trente ")
    sentence = sentence.replace("quarante-", "quarante ")
    sentence = sentence.replace("cinquante-", "cinquante ")
    sentence = sentence.replace("soixante-", "soixante ")
    sentence = sentence.replace("quatre-", "quatre ")
    # Spaces
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"^\s+", "", sentence)
    sentence = re.sub(r"\s+$", "", sentence)
    # Specific
    sentence = sentence.replace("c' est", "c'est")  # Re-join this word
    return sentence


def write_first_row(save_folder):
    for corpus in ["train", "dev", "test", "test2"]:
        SB_file = open(save_folder + "/csv/" + corpus + ".csv", "w")
        writer = csv.writer(SB_file, delimiter=",")
        writer.writerow(
            [
                "ID",
                "duration",
                "start_seg",
                "end_seg",
                "wav",
                "wav_format",
                "spk_id",
                "spk_id_format",
                "wrd",
                "wrd_format",
                "char",
                "char_format",
            ]
        )
        SB_file.close()


def split_audio_channels(path, filename, channel, save_folder):
    """
    Split the stereo wav Media files from the dowloaded dataset.
    Keep only the speaker channel.

    Arguments:
    -------
    path: str
        Path of the original Media file without the extension ".wav" nor ".trs".
    filename: str
        Name of the Media recording.
    channel: str
        "R" or "L" following the channel of the speaker in the stereo wav file.
    save_folder: str
        Path where the csvs and preprocessed wavs will be stored.
    """

    channel_int = "1"
    if channel == "R":
        channel_int = "2"
    path = path.replace("1 ", "'1 ")
    path = path.replace("2 ", "'2 ")
    path = path.replace(" 2", " 2'")
    os.system(
        "sox "
        + path
        + " "
        + save_folder
        + "/wav/"
        + channel
        + filename
        + "_8khz.wav remix "
        + channel_int
    )
    os.system(
        "sox -G "
        + save_folder
        + "/wav/"
        + channel
        + filename
        + "_8khz.wav -r 16000 "
        + save_folder
        + "/wav/"
        + channel
        + filename
        + ".wav 2>/dev/null"
    )
    os.system("rm " + save_folder + "/wav/" + channel + filename + "_8khz.wav")


def get_root(path, id):
    with open(path, "rb") as f:
        text = f.read()
        text2 = text.decode("ISO-8859-1")
        tree = DOM.parseString(text2)
        root = tree.childNodes[id]
    return root


def get_speaker(dialogue):
    speaker = dialogue.getAttribute("nameSpk")
    speaker = normalize_speaker(speaker)
    return speaker


def get_speaker_test2(root):
    for speaker in root.getElementsByTagName("Speaker"):
        if speaker.getAttribute("name")[0] == "s":
            speaker_id = speaker.getAttribute("id")
            speaker_name = speaker.getAttribute("name")
            speaker_name = normalize_speaker(speaker_name)
            return speaker_id, speaker_name


def normalize_speaker(speaker):
    speaker = speaker.replace("-", "_")
    speaker = speaker.replace("#", "_")
    speaker = speaker.replace("__", "_1_")
    speaker = speaker.replace("speaker1", "speaker")
    speaker = speaker.replace("108730", "1_08730")
    speaker = speaker.replace("087301123", "08730_1123")
    speaker = speaker.replace("087301457", "08730_1457")
    speaker = speaker.replace(".", "")
    speaker = speaker.replace("speaker_08730_1394", "speaker_1_08730_1394")
    speaker = speaker.replace("speaker_08730_1399", "speaker_1_08730_1399")
    speaker = speaker.replace("speaker_08730_37", "speaker_1_08730_37")
    speaker = speaker.replace("speaker_08730_400", "speaker_1_08730_400")
    speaker = speaker.replace("_8730", "_08730")
    speaker = speaker.replace("_0873", "_08730")
    speaker = speaker.replace("_08737", "_08730")
    speaker = speaker.replace("21_08730", "1_08730")
    speaker = speaker.replace("058730", "08730")
    speaker = speaker.replace("2_08730", "1_08730")
    speaker = speaker.replace("speaker_08730_846", "speaker_1_08730_846")
    speaker = speaker.replace("speaker_8730_270", "speaker_1_08730_270")
    return speaker


def get_channels(path):
    channels = []
    filenames = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            channels.append(row[0])
            filenames.append(row[1])
    return channels, filenames


def get_channel(filename, channels, filenames):
    channel = channels[filenames.index(filename)]
    return channel


def get_concepts_full_relax(path):
    concepts_full = []
    concepts_relax = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            concepts_full.append(row[0])
            concepts_relax.append(row[1])
    return concepts_full, concepts_relax


def get_concept_relax(concept, concepts_full, concepts_relax):
    for c in concepts_full:
        if (c[-1] == "*" and concept[: len(c) - 1] == c[:-1]) or concept == c:
            return concepts_relax[concepts_full.index(c)]
    return concept


def get_unused_dialogs(data_folder):
    # Used dialogs
    proc = subprocess.Popen(
        "egrep -a '<dialogue' "
        + data_folder
        + "/E0024/MEDIA1FR_00/MEDIA1FR/DATA/media_*.xml "
        + "| cut -d' ' -f2 "
        + "| cut -d'"
        + '"'
        + "' -f2",
        stdout=subprocess.PIPE,
        shell=True,
    )
    used_dialogs = [str(i)[2:-3] for i in proc.stdout.readlines()]
    # All dialogs
    all_dialogs = glob.glob(data_folder + "/S0272/**/*.wav", recursive=True)
    all_dialogs = [e.split("/")[-1][:-4] for e in all_dialogs]
    # Unused dialogs
    unused_dialogs = []
    for dialog in all_dialogs:
        if dialog not in used_dialogs:
            unused_dialogs.append(dialog)

    return unused_dialogs


def get_IDs(speaker_name, sentences, channel, filename):
    IDs = []
    for sentence in sentences:
        IDs.append(
            channel
            + filename
            + "#"
            + speaker_name
            + "#"
            + sentence[2]
            + "_"
            + sentence[3]
        )
    return IDs
