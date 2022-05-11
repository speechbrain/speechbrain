# -*- coding: utf-8 -*-

"""
Data preparation.
Download: See README.md

Author
------
Gaelle Laperriere 2021
"""

import xml.dom.minidom as DOM
from tqdm import tqdm
import subprocess
import num2words
import argparse
import csv
import os
import glob
import re

def parse(
    root, 
    channels, 
    filenames, 
    wav_folder, 
    csv_folder, 
    method, 
    task, 
    corpus
):
    """
    Prepares the data for the csv files of the Media dataset.

    Arguments
    ---------
    root : Document
        Object representing the content of the Media xml document being processed.
    channels : list of str
        Channels (Right / Left) of the stereo recording to keep.
    filenames : list of str
        Linked IDs of the recordings, for the channels to keep.
    wav_folder : str
        Path where the wavs will be stored.
    csv_folder : str
        Path where the csvs will be stored.
    method : str
        Either 'full' or 'relax'.
    task : str
        Either 'asr' or 'slu'.
    corpus : str
        'train', 'dev' or 'test'.

    Returns
    -------
    list of str
    """

    for dialogue in tqdm(root.getElementsByTagName("dialogue")):

        data = []
        speaker_name = get_speaker(dialogue)
        filename = dialogue.getAttribute("id")
        channel = get_channel(filename, channels, filenames)

        for turn in dialogue.getElementsByTagName("turn"):
            if turn.getAttribute("speaker") == "spk":

                time_beg = turn.getAttribute("startTime")
                time_end = turn.getAttribute("endTime")

                if task == "slu":
                    sentences = parse_sentences_slu(
                        turn, 
                        time_beg, 
                        time_end, 
                        method
                    )
                else:
                    sentences = parse_sentences_asr(
                        turn, 
                        time_beg, 
                        time_end
                    )

                out = subprocess.Popen(
                    [
                        "soxi", 
                        "-D", 
                        wav_folder + "/" + channel + filename + ".wav"
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                stdout, stderr = out.communicate()
                wav_duration = str("%.2f" % float(stdout))
                wav = wav_folder + "/" + channel + filename + ".wav"
                IDs = get_IDs(speaker_name, sentences, channel, filename)

                # Append data
                for n in range(len(sentences)):
                    f1 = float(sentences[n][3])
                    f2 = float(sentences[n][2])
                    duration = str("%.2f" % (f1 - f2))
                    if (
                        float(wav_duration) >= f1 
                        and float(duration) != 0.0 
                        and sentences[n][0]!= ""
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
                                "string"
                            ]
                        )

        if data is not None:
            append_data(csv_folder + "/" + corpus + ".csv", data)

    return data

def parse_bis(
    root, 
    channels,
    filenames, 
    wav_folder, 
    csv_folder, 
    method, 
    task, 
    filename, 
    concepts_full, 
    concepts_relax
):
    """
    Prepares the data for the csv files of the Media dataset.

    Arguments
    ---------
    root : Document
        Object representing the content of the Media xml document being processed.
    channels : list of str
        Channels (Right / Left) of the stereo recording to keep.
    filenames : list of str
        Linked IDs of the recordings, for the channels to keep.
    wav_folder : str
        Path where the wavs will be stored.
    csv_folder : str
        Path where the csvs will be stored.
    method : str
        Either 'full' or 'relax'.
    task : str
        Either 'asr' or 'slu'.
    filename : str
        Name of the Media recording.
    concepts_full : list of str
        Concepts in method full.
    concepts_relax : list of str
        Concepts equivalent in method relax.

    Returns
    -------
    list of str
    """

    data = []
    speaker_id, speaker_name = get_speaker_bis(root)
    channel = get_channel(filename, channels, filenames)

    for turn in root.getElementsByTagName("Turn"):
        if turn.getAttribute("speaker") == speaker_id:

            time_beg = turn.getAttribute("startTime")
            time_end = turn.getAttribute("endTime")

            if task == "slu":
                sentences = parse_sentences_slu_bis(
                    turn, 
                    time_beg, 
                    time_end, 
                    method, 
                    concepts_full, 
                    concepts_relax
                )
            else:
                sentences = parse_sentences_asr_bis(
                    turn, 
                    time_beg, 
                    time_end
                )

            if filename == "70" and sentences[len(sentences)-1][3] == "344.408":
                sentences[len(sentences)-1][3] = "321.000"

            out = subprocess.Popen(
                ["soxi", "-D", wav_folder + "/" + channel + filename + ".wav"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            stdout, stderr = out.communicate()
            wav_duration = str("%.2f" % float(stdout))
            wav = wav_folder + "/" + channel + filename + ".wav"
            IDs = get_IDs(speaker_name, sentences, channel, filename)

            # Append data
            for n in range(len(sentences)):
                f1 = float(sentences[n][3])
                f2 = float(sentences[n][2])
                duration = str("%.2f" % (f1 - f2))
                if (
                    float(wav_duration) >= f1 
                    and float(duration) != 0.0 
                    and sentences[n][0]!= ""
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
                            "string"
                        ]
                    )

    if data is not None:
        append_data(csv_folder + "/test_bis.csv", data)

    return data

def parse_sentences_asr(
    turn, 
    time_beg, 
    time_end
):
    """
    Get the sentences spoken by the speaker (not the "Compère" aka Woz).

    Arguments:
    -------
    turn : list of Document
        The current turn node.
    time_beg : str
        Time (s) at the beggining of the turn.
    time_end : str
        Time (s) at the end of the turn.

    Returns
    -------
    dictionnary of str
    """

    sentences = [["", "", time_beg, time_end]]
    has_speech = False
    n = 0  # Number of segments in the turn

    # For each transcription in the Turn
    for transcription in turn.getElementsByTagName("transcription"):
        # Check for sync or text
        for node in transcription.childNodes:

            # Check transcription
            if (
                node.nodeType == node.TEXT_NODE 
                and node.data.replace("\n", "").replace(" ", "") != ""
            ):
                sentence = normalize_sentence(node.data)
                sentences[n][0] += sentence + " "
                sentences[n][1] += (" ".join(list(sentence.replace(" ", "_"))) + " _ ")
                sentences[n][3] = time_end
                has_speech = True

            # Check Sync times
            if node.nodeName == "Sync":
                # If the segment has no speech yet
                if not (has_speech):
                    # Change time_beg for the last segment
                    sentences[n][2] = node.getAttribute("time")
                # If the segment has speech
                else:
                    # Change time_end for the last segment
                    sentences[n][3] = node.getAttribute("time")
                    sentences.append(["", "", sentences[n][3], time_end])
                    has_speech = False
                    n += 1

    for n in range(len(sentences)):
        if sentences[n][0] != "":
            sentences[n][0] = sentences[n][0][:-1]  # Remove last ' '
            sentences[n][1] = sentences[n][1][:-3]  # Remove last ' _ '
        else:
            del sentences[n]  # Usefull for last appended segment

    return sentences

def parse_sentences_slu(
    turn, 
    time_beg, 
    time_end, 
    method
):
    """
    Get the sentences spoken by the speaker.

    Arguments:
    -------
    turn : list of Document
        The current turn node.
    time_beg : str
        Time (s) at the beginning of the turn.
    time_end : str
        Time (s) at the end of the turn.
    method : str
        Either 'full' or 'relax'.

    Returns
    -------
    dictionnary of str
    """

    has_speech = False
    sentences = [["", "", time_beg, time_end]]
    concept_open = False
    sync_waiting = False
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

                # Check for sync or text
                for transcription in sem.getElementsByTagName("transcription"):
                    for node in transcription.childNodes:

                        # Check transcription
                        if (
                            node.nodeType == node.TEXT_NODE 
                            and node.data.replace("\n", "").replace(" ", "") != ""
                        ):
                            # Add a new concept, when speech following
                            if concept != "null" and not concept_open:
                                if method == "relax" or specif == "null":
                                    sentences[n][0] += "<" + concept + "> "
                                    sentences[n][1] += "<" + concept + "> _ "
                                elif method == "full" and specif != "null":
                                    sentences[n][0] += "<" + concept + specif + "> "
                                    sentences[n][1] += "<" + concept + specif + "> _ "
                                concept_open = True
                            sentence = normalize_sentence(node.data)
                            sentences[n][0] += sentence + " "
                            sentences[n][1] += (" ".join(list(sentence.replace(" ", "_"))) + " _ ")
                            sentences[n][3] = time_end
                            has_speech = True
                            sync_waiting = False

                        # Check Sync times
                        if node.nodeName == "Sync":
                            # If the segment has no speech yet
                            if not (has_speech):
                                # Change time_beg for the last segment
                                sentences[n][2] = node.getAttribute("time")
                            # If the segment has speech and sync doesn't cut a concept
                            elif not concept_open:
                                # Change time_end for the last segment
                                sentences[n][3] = node.getAttribute("time")
                                sentences.append(["", "", sentences[n][3], time_end])
                                has_speech = False
                                n += 1
                            else:
                                sync_waiting = True
                                time = node.getAttribute("time")

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

    for n in range(len(sentences)):
        if sentences[n][0] != "":
            sentences[n][0] = sentences[n][0][:-1]  # Remove last ' '
            sentences[n][1] = sentences[n][1][:-3]  # Remove last ' _ '
        else:
            del sentences[n]  # Usefull for last appended segment

    return sentences

def parse_sentences_asr_bis(
    turn, 
    time_beg, 
    time_end
):
    """
    Get the sentences spoken by the speaker (not the "Compère" aka Woz).

    Arguments:
    -------
    nodes : list of Document
        All the xml following nodes present in the turn.
    time_beg : str
        Time (s) at the beginning of the turn.
    time_end : str
        Time (s) at the end of the turn.

    Returns
    -------
    dictionnary of str
    """

    sentences = [["", "", time_beg, time_end]]
    n = 0  # Number of segments in the turn
    has_speech = False

    # For each node in the Turn
    for node in turn.childNodes:

        # Check transcription
        if node.nodeType == node.TEXT_NODE and node.data.replace("\n", "") != "":
            sentence = normalize_sentence(node.data)
            sentences[n][0] += sentence + " "
            sentences[n][1] += (" ".join(list(sentence.replace(" ", "_"))) + " _ ")
            sentences[n][3] = time_end
            has_speech = True

        if node.nodeName == "Sync":
            # If the segment has no speech yet
            if not (has_speech):
                # Change time_beg for the last segment
                sentences[n][2] = node.getAttribute("time")
            # If the segment has speech
            else:
                # Change time_end for the last segment
                sentences[n][3] = node.getAttribute("time")
                sentences.append(["", "", sentences[n][3], time_end])
                has_speech = False
                n += 1

    for n in range(len(sentences)):
        if sentences[n][0] != "":
            sentences[n][0] = sentences[n][0][:-1]  # Remove last ' '
            sentences[n][1] = sentences[n][1][:-3]  # Remove last ' _ '
        else:
            del sentences[n]  # Usefull for last appended segment

    return sentences

def parse_sentences_slu_bis(
    turn, 
    time_beg, 
    time_end, 
    method, 
    concepts_full, 
    concepts_relax
):
    """
    Get the sentences spoken by the speaker (not the "Compère" aka Woz).

    Arguments:
    -------
    nodes : list of Document
        All the xml following nodes present in the turn.
    time_beg : str
        Time (s) at the beginning of the turn.
    time_end : str
        Time (s) at the end of the turn.
    method : str
        Either 'full' or 'relax'.
    concepts_full : list of str
        Concepts in method full.
    concepts_relax : list of str
        Concepts equivalent in method relax.

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

    # For each node in the Turn
    for node in turn.childNodes:

        # Check concept
        if node.nodeName == "SemDebut":
            concept = node.getAttribute("concept")
            if method == "relax":
                concept = get_concept_relax(concept, concepts_full, concepts_relax)

        # Check transcription
        if node.nodeType == node.TEXT_NODE and node.data.replace("\n", "") != "":
            # Add a new concept, when speech following
            # (useful for 'SemDeb + Sync + Speech' & 'SemDeb + Speech + Sync + Speech')
            if concept != "null" and not concept_open:
                sentences[n][0] += "<" + concept + "> "
                sentences[n][1] += "<" + concept + "> _ "
                concept_open = True
            sentence = normalize_sentence(node.data)
            sentences[n][0] += sentence + " "
            sentences[n][1] += (" ".join(list(sentence.replace(" ", "_"))) + " _ ")
            sentences[n][3] = time_end
            has_speech = True
            sync_waiting = False

        # Save audio segment
        if node.nodeName == "SemFin":
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

        if node.nodeName == "Sync":
            # If the segment has no speech yet
            if not (has_speech):
                # Change time_beg for the last segment
                sentences[n][2] = node.getAttribute("time")
            # If the segment has speech and sync doesn't cut a concept
            elif not concept_open:
                # Change time_end for the last segment
                sentences[n][3] = node.getAttribute("time")
                sentences.append(["", "", sentences[n][3], time_end])
                has_speech = False
                n += 1
            else:
                sync_waiting = True
                time = node.getAttribute("time")

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
    sentence = re.sub(r"[^\w\s'-><_]", "", sentence)  # Remove punctuation except '-><_
    # Case
    #sentence = sentence.lower()  # Lowercase letters
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


def write_first_row(folder):
    for corpus in ["train", "dev", "test", "test_bis"]:
        SB_file = open(folder + "/" + corpus + ".csv", "w")
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
    return None


def append_data(path, data):
    """
    Make the csv corpora using data retrieved previously for one Media file.

    Arguments:
    -------
    path : str
        Path of the folder to store csv.
    data : list of str
        Data retrieved from the original xml file.

    Returns
    -------
    None
    """
    SB_file = open(path, "a")
    writer = csv.writer(SB_file, delimiter=",")
    writer.writerows(data)
    SB_file.close()
    return None


def split_audio_channels(path, filename, channel, folder):
    """
    Split the stereo wav Media files from the dowloaded dataset.
    Keep only the speaker channel.

    Arguments:
    -------
    path : str
        Path of the original Media file without the extension ".wav" nor ".trs".
    filename : str
        Name of the Media recording.
    channel : str
        "R" or "L" following the channel of the speaker in the stereo wav file.
    folder : str
        Path where the wavs will be stored.

    Returns
    -------
    None
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
        + folder 
        + "/" 
        + channel 
        + filename 
        + "_8khz.wav remix " 
        + channel_int
    )
    os.system(
        "sox -G " 
        + folder 
        + "/" 
        + channel 
        + filename 
        + "_8khz.wav -r 16000 " 
        + folder 
        + "/" 
        + channel 
        + filename 
        + ".wav 2>/dev/null"
    )
    os.system(
        "rm " 
        + folder 
        + "/" 
        + channel 
        + filename 
        + "_8khz.wav"
    )
    return None


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


def get_speaker_bis(root):
    for speaker in root.getElementsByTagName("Speaker"):
        if speaker.getAttribute("name")[0] == "s":
            speaker_id = speaker.getAttribute("id")
            speaker_name = speaker.getAttribute("name")
            speaker_name = normalize_speaker(speaker_name)
            return speaker_id, speaker_name
            6


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


def get_concepts_full_relax(path):
    concepts_full = []
    concepts_relax = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            concepts_full.append(row[0])
            concepts_relax.append(row[1])
    return concepts_full, concepts_relax


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
        shell=True
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


def get_channel(filename, channels, filenames):
    channel = channels[filenames.index(filename)]
    return channel


def get_concept_relax(concept, concepts_full, concepts_relax):
    for c in concepts_full:
        if (c[-1] == "*" and concept[:len(c)-1] == c[:-1]) or concept == c:
            return concepts_relax[concepts_full.index(c)]
    return concept


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

"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "data_folder", 
        type=str, 
        help="Path where folders S0272 and E0024 are stored."
    )
    parser.add_argument(
        "wav_folder", 
        type=str, 
        help="Path where the wavs will be stored."
    )
    parser.add_argument(
        "csv_folder", 
        type=str, 
        help="Path where the csv will be stored."
    )
    parser.add_argument(
        "-w", 
        "--skip_wav", 
        action="store_true", 
        required=False, 
        help="Skip the wav files storing if already done before."
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-r", 
        "--relax", 
        action="store_true", 
        required=False, 
        help="Remove specifiers from concepts."
    )
    group.add_argument(
        "-f", 
        "--full", 
        action="store_false", 
        required=False, 
        help="Keep specifiers in concepts. Method used by default."
    )

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument(
        "-s", 
        "--slu", 
        action="store_true", 
        required=False, 
        help="Parse SLU data."
    )
    group2.add_argument(
        "-a", 
        "--asr", 
        action="store_false", 
        required=False, 
        help="Parse ASR data."
    )
    args = parser.parse_args()

    data_folder = args.data_folder
    wav_folder = args.wav_folder
    csv_folder = args.csv_folder
    skip_wav = args.skip_wav

    if args.relax:
        method = "relax"
    else:
        method = "full"

    if args.slu:
        task = "slu"
        print(
            "Processing SLU Media Dataset with "
            + method
            + " method for the concepts."
        )
    else:
        task = "asr"
        print("Processing ASR Media Dataset.")

"""

def prepare_media(
    data_folder,
    wav_folder,
    csv_folder,
    skip_wav,
    method,
    task,
    skip_prep
):

    if skip_prep:
        return

    if task == "slu":
        print(
            "Processing SLU Media Dataset with " 
            + method 
            + " method for the concepts."
        )
    else:
        print("Processing ASR Media Dataset.")

    wav_paths = glob.glob(data_folder + "/S0272/**/*.wav", recursive=True)
    channels, filenames = get_channels("./channels.csv")
    concepts_full, concepts_relax = get_concepts_full_relax("./concepts_full_relax.csv")
    unused_dialogs = get_unused_dialogs(data_folder)
    write_first_row(csv_folder)

    xmls = {
        "media_lot1.xml": "train",
        "media_lot2.xml": "train",
        "media_lot3.xml": "train",
        "media_lot4.xml": "train",
        "media_testHC.xml": "test",
        "media_testHC_a_blanc.xml": "dev"
    }

    if not (skip_wav):
        print("Processing wavs.")
        for wav_path in tqdm(wav_paths):
            filename = wav_path.split("/")[-1][:-4]
            channel = get_channel(filename, channels, filenames)
            split_audio_channels(wav_path, filename, channel, wav_folder)

    for xml in xmls:
        print(
            "Processing file " 
            + str(list(xmls.keys()).index(xml)+1) 
            + "/" 
            + str(len(xmls)) + "."
        )
        root = get_root(
            data_folder 
            + "/E0024/MEDIA1FR_00/MEDIA1FR/DATA/" 
            + xml,
            0
        )
        parse(
            root, 
            channels, 
            filenames, 
            wav_folder, 
            csv_folder, 
            method, 
            task, 
            xmls[xml]
        )

    print("Processing files for test_bis.")
    for filename in tqdm(unused_dialogs):
        root = get_root(
            data_folder 
            + "/E0024/MEDIA1FR_00/MEDIA1FR/DATA/semantizer_files/" 
            + filename 
            + "_HC.xml",
            1
        )
        parse_bis(
            root, 
            channels, 
            filenames, 
            wav_folder, 
            csv_folder, 
            method, 
            task, 
            filename, 
            concepts_full, 
            concepts_relax
        )
