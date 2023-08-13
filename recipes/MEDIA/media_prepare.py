"""
Data preparation.
Download:
https://catalogue.elra.info/en-us/repository/browse/ELRA-S0272/
https://catalogue.elra.info/en-us/repository/browse/ELRA-E0024/
https://www.dropbox.com/sh/y7ab0lktbylz647/AADMsowYHmNYwaoL_hQt7NMha?dl=0
See README.md for more info.

Author
------
Gaelle Laperriere 2023
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
    channels_path,
    concepts_path,
    skip_wav=True,
    method="slu",
    task="full",
    skip_prep=False,
    process_test2=False,
):
    """
    Prepares the csv files for the MEDIA dataset.
    Both following repositories are necessary for transcriptions
    and annotations (S0272) and audio (E0024).
    https://catalogue.elra.info/en-us/repository/browse/ELRA-S0272/
    https://catalogue.elra.info/en-us/repository/browse/ELRA-E0024/

    Arguments
    ---------
    data_folder: str
        Path where folders S0272 and E0024 are stored.
    save_folder: str
        Path where the csvs and preprocessed wavs will be stored.
    channels_path: str
        Path of the channels.csv file downloaded via https://www.dropbox.com/sh/y7ab0lktbylz647/AADMsowYHmNYwaoL_hQt7NMha?dl=0
    concepts_path: str
        Path of the concepts_full_relax.csv file downloaded via https://www.dropbox.com/sh/y7ab0lktbylz647/AADMsowYHmNYwaoL_hQt7NMha?dl=0
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
    process_test2: bool, optional
        It True, process test2 corpus
    """

    if skip_prep:
        return

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(save_folder + "/wav"):
        os.makedirs(save_folder + "/wav")
    if not os.path.exists(save_folder + "/csv"):
        os.makedirs(save_folder + "/csv")

    if skip(
        save_folder + "/csv/train.csv",
        save_folder + "/csv/dev.csv",
        save_folder + "/csv/test.csv",
    ):
        logger.info("Csv files already exist, skipping data preparation!")
        return

    if task == "slu":
        if method == "relax" or method == "full":
            logger.info("Processing SLU " + method + " Media Dataset")
        else:
            raise ValueError("Parameter method must be 'full' or 'relax'")
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

    train_data = []
    dev_data = []
    test_data = []
    test2_data = []

    wav_paths = glob.glob(data_folder + "/S0272/**/*.wav", recursive=True)
    channels, filenames = get_channels(channels_path)

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
            "Processing xml file "
            + str(list(xmls.keys()).index(xml) + 1)
            + "/"
            + str(len(xmls))
        )
        root = get_root(
            data_folder + "/E0024/MEDIA1FR_00/MEDIA1FR/DATA/" + xml, 0,
        )
        data = parse(
            root, channels, filenames, save_folder, method, task, xmls[xml],
        )
        if xmls[xml] == "train":
            train_data.extend(data)
        elif xmls[xml] == "dev":
            dev_data.extend(data)
        elif xmls[xml] == "test":
            test_data.extend(data)

    # Test2.
    if process_test2:
        unused_dialogs = get_unused_dialogs(data_folder)
        concepts_full, concepts_relax = get_concepts_full_relax(concepts_path)
        logger.info("Processing xml files for test2")
        for filename in unused_dialogs:
            root = get_root(
                data_folder
                + "/E0024/MEDIA1FR_00/MEDIA1FR/DATA/semantizer_files/"
                + filename
                + "_HC.xml",
                1,
            )
            test2_data.extend(
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
            )

    append_data(save_folder, train_data, "train")
    append_data(save_folder, dev_data, "dev")
    append_data(save_folder, test_data, "test")
    if process_test2:
        append_data(save_folder, test2_data, "test2")


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the MEDIA data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    ):
        skip = True

    return skip


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

    Returns
    -------
    list
        all informations needed to append the data in SpeechBrain csv files.
    """

    data = []

    for dialogue in root.getElementsByTagName("dialogue"):

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

                data.append([channel, filename, speaker_name, sentences])

    return data


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

    data = []

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

            data.append([channel, filename, speaker_name, sentences])

    return data


def append_data(save_folder, data, corpus):
    """
    Make the csv corpora using data retrieved previously for one Media file.

    Arguments
    ---------
    save_folder: str
        Path where the csvs and preprocessed wavs will be stored.
    data: list
        channel, filename, speaker_name, sentences
    corpus: str
        Either 'train', 'dev', 'test', or 'test2'.
    """

    logger.info("Preparing " + corpus + ".csv")

    to_append = []

    for line in tqdm(data):
        channel, filename, speaker_name, sentences = line

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

        # Fill to_append list
        for n in range(len(sentences)):
            f1 = float(sentences[n][3])
            f2 = float(sentences[n][2])
            duration = str("%.2f" % (f1 - f2))
            if (
                float(wav_duration) >= f1
                and float(duration) != 0.0
                and sentences[n][0] != ""
            ):
                to_append.append(
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

    # Write to_append
    if to_append is not None:
        write_first_row(save_folder, corpus)
        path = save_folder + "/csv/" + corpus + ".csv"
        SB_file = open(path, "a")
        writer = csv.writer(SB_file, delimiter=",")
        writer.writerows(to_append)
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

    sentences = clean_last_char(sentences)
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

    sentences = clean_last_char(sentences)

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
    """
    Parse text nodes from the xml files of MEDIA.

    Arguments
    ---------
    node: Node
        Node of the xml file.
    sentences: dictionnary of str
        All sentences being extracted from the turn.
    sync_waiting: bool
        Used to keep track of sync nodes, to cut blank audio signal.
        True if a sync node has been processed without text in it.
        False if no sync nodes have been processed, or text has been parsed after one.
    has_speech: bool
        Used to keep track of the existency of speech in the turn's sentence.
        True if speech is present in the turn.
        False if no speech is present yet in the turn.
    concept: str
        Concept of the node being processed.
        Will be "null" if no concept is linked to this node.
    concept_open: bool
        Used to know if a concept has been used but not its closing tag ">".
        True if closing tag not seen yet and concept has been used.
        False if clossing tag put or concept not used.
    task: str, optional
        Either 'slu' or 'asr'.
        'slu' Parse SLU data.
        'asr' Parse ASR data.
    n: int
        Used to keep track of the number of sentences in the turn.
    time_end: str
        Last time given by the turn, after last speech.

    Returns
    -------
    dictionnary of str, bool, bool, bool
    """

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
    """
    Parse sync nodes from the xml files of MEDIA.

    Arguments
    ---------
    node: Node
        Node of the xml file.
    sentences: dictionnary of str
        All sentences being extracted from the turn.
    sync_waiting: bool
        Used to keep track of sync nodes, to cut blank audio signal.
        True if a sync node has been processed without text in it.
        False if no sync nodes have been processed, or text has been parsed after one.
    has_speech: bool
        Used to keep track of the existency of speech in the turn's sentence.
        True if speech is present in the turn.
        False if no speech is present yet in the turn.
    concept_open: bool
        Used to know if a concept has been used but not its closing tag ">".
        True if closing tag not seen yet and concept has been used.
        False if clossing tag put or concept not used.
    task: str, optional
        Either 'slu' or 'asr'.
        'slu' Parse SLU data.
        'asr' Parse ASR data.
    n: int
        Used to keep track of the number of sentences in the turn.
    time: str
        Current time given by the sync node.
    time_end: str
        Last time given by the turn, after last speech.

    Returns
    -------
    dictionnary of str, bool, bool, str, int
    """

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
    """
    Parse SemFin nodes from the xml files of MEDIA.

    Arguments
    ---------
    sentences: dictionnary of str
        All sentences being extracted from the turn.
    sync_waiting: bool
        Used to keep track of sync nodes, to cut blank audio signal.
        True if a sync node has been processed without text in it.
        False if no sync nodes have been processed, or text has been parsed after one.
    has_speech: bool
        Used to keep track of the existency of speech in the turn's sentence.
        True if speech is present in the turn.
        False if no speech is present yet in the turn.
    concept: str
        Concept of the node being processed.
        Will be "null" if no concept is linked to this node.
    concept_open: bool
        Used to know if a concept has been used but not its closing tag ">".
        True if closing tag not seen yet and concept has been used.
        False if clossing tag put or concept not used.
    task: str, optional
        Either 'slu' or 'asr'.
        'slu' Parse SLU data.
        'asr' Parse ASR data.
    n: int
        Used to keep track of the number of sentences in the turn.
    time: str
        Current time given by the sync node.
    time_end: str
        Last time given by the turn, after last speech.

    Returns
    -------
    dictionnary of str, str, bool, bool, bool, int
    """

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


def clean_last_char(sentences):
    """
    Clean the sentences by deleting their last characters.

    Arguments
    ---------
    sentences: dictionnary of str
        All sentences being extracted from the turn.

    Returns
    -------
    dictionnary of str
    """

    for n in range(len(sentences)):
        if sentences[n][0] != "":
            sentences[n][0] = sentences[n][0][:-1]  # Remove last ' '
            sentences[n][1] = sentences[n][1][:-3]  # Remove last ' _ '
        else:
            del sentences[n]  # Usefull for last appended segment
    return sentences


def normalize_sentence(sentence):
    """
    Normalize and correct a sentence of the turn.

    Arguments
    ---------
    sentence: str
        A sentence being extracted from the turn.

    Returns
    -------
    str
    """

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


def write_first_row(save_folder, corpus):
    """
    Write the first row of the csv files.

    Arguments
    ---------
    save_folder: str
        Path where the csvs and preprocessed wavs will be stored.
    corpus : str
        Either 'train', 'dev', 'test', or 'test2'.
    """

    SB_file = open(save_folder + "/csv/" + corpus + ".csv", "w")
    writer = csv.writer(SB_file, delimiter=",")
    writer.writerow(
        [
            "ID",
            "duration",
            "start",
            "stop",
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
    """
    Get the root of an xml file.

    Arguments
    ---------
    path: str
        The path of the xml file.
    id: int
        id of the node to extract, different considering the xml format.

    Returns
    -------
    Node
    """

    with open(path, "rb") as f:
        text = f.read()
        text2 = text.decode("ISO-8859-1")
        tree = DOM.parseString(text2)
        root = tree.childNodes[id]
    return root


def get_speaker(dialogue):
    """
    Get the name of the speaker of a dialogue.

    Arguments
    ---------
    dialogue: Node
        The node where the speaker information is stored.

    Returns
    -------
    str
    """

    speaker = dialogue.getAttribute("nameSpk")
    speaker = normalize_speaker(speaker)
    return speaker


def get_speaker_test2(root):
    """
    Get the name of the speaker of a whole xml file, for the test2 xml structure.

    Arguments
    ---------
    root: Node
        The node where the speaker information is stored.

    Returns
    -------
    str, str
    """

    for speaker in root.getElementsByTagName("Speaker"):
        if speaker.getAttribute("name")[0] == "s":
            speaker_id = speaker.getAttribute("id")
            speaker_name = speaker.getAttribute("name")
            speaker_name = normalize_speaker(speaker_name)
            return speaker_id, speaker_name


def normalize_speaker(speaker):
    """
    Normalize and correct the speaker name.

    Arguments
    ---------
    speaker: str
        Initial name of the speaker as given by the xml file.

    Returns
    -------
    str
    """

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
    """
    Get the channels (Right / Left) from the stereo audio files where the speaker (not the WoZ) speak.

    Arguments
    ---------
    path: str
        Path of the channels csv file given with this recipe.
        Can be dowloaded from https://www.dropbox.com/sh/y7ab0lktbylz647/AADMsowYHmNYwaoL_hQt7NMha?dl=0

    Returns
    -------
    list of str, list of str
    """

    channels = []
    filenames = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            channels.append(row[0])
            filenames.append(row[1])
    return channels, filenames


def get_channel(filename, channels, filenames):
    """
    Get the channel (Right / Left) of a transcription linked audio.

    Arguments
    ---------
    filename: str
        Name of the Media recording.
    channels: list of str
        Channels (Right / Left) of the stereo recording to keep.
    filenames: list of str
        Linked IDs of the recordings, for the channels to keep.

    Returns
    -------
    str
    """

    channel = channels[filenames.index(filename)]
    return channel


def get_concepts_full_relax(path):
    """
    Put the corresponding MEDIA relax concepts from their full version in lists from the concepts csv file.

    Arguments
    ---------
    path: str
        Path of the channels csv file given with this recipe.
        Can be dowloaded from https://www.dropbox.com/sh/y7ab0lktbylz647/AADMsowYHmNYwaoL_hQt7NMha?dl=0

    Returns
    -------
    list of str, list of str
    """

    concepts_full = []
    concepts_relax = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            concepts_full.append(row[0])
            concepts_relax.append(row[1])
    return concepts_full, concepts_relax


def get_concept_relax(concept, concepts_full, concepts_relax):
    """
    Get the corresponding MEDIA relax concept from its full version.

    Arguments
    ---------
    concept: str
        Concept of the node being processed.
    concepts_full: list of str
        Concepts in method full.
    concepts_relax: list of str
        Concepts equivalent in method relax.

    Returns
    -------
    str
    """

    for c in concepts_full:
        if (c[-1] == "*" and concept[: len(c) - 1] == c[:-1]) or concept == c:
            return concepts_relax[concepts_full.index(c)]
    return concept


def get_unused_dialogs(data_folder):
    """
    Get the dialogs to be process for the test2 new corpus.

    Arguments
    ---------
    data_folder: str
        Path where folders S0272 and E0024 are stored.

    Returns
    -------
    list of str
    """

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
    """
    Get the usual IDs used with MEDIA to put in the new SpeechBrain csv.

    Arguments
    ---------
    speaker_name: str
        Speaker name of the turn, already normalized.
    sentences: dictionnary of str
        All sentences being extracted from the turn.
    channel: str
        "R" or "L" following the channel of the speaker in the stereo wav file.
    filename: str
        Name of the Media recording.

    Returns
    -------
    list of str
    """

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
