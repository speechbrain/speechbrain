"""
Data preparation script for the GigaSpeech dataset.

Download instructions: https://github.com/SpeechColab/GigaSpeech
Reference: https://arxiv.org/abs/2106.06909

Author
-------
 * Adel Moumen, 2024
"""

import logging
import os
import json
import csv
from dataclasses import dataclass
import functools
from speechbrain.utils.parallel import parallel_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SAMPLERATE = 16000
GRABAGE_UTTERANCE_TAGS = ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"]
PUNCTUATION_TAGS = {
    "<COMMA>": ",", 
    "<EXCLAMATIONPOINT>": "!", 
    "<PERIOD>": ".", 
    "<QUESTIONMARK>": "?"
}
SPLITS = ["DEV", "TEST"]
TRAIN_SUBSET = ["XS", "S", "M", "L", "XL"]


@dataclass
class GigaSpeechRow:
    utt_id: str # segment[sid]
    audio_id: str # audio[aid]
    audio_path: str # by default this is opus files
    speaker: str # audio["speaker"]
    begin_time: float
    end_time: float
    duration: float
    text: str

def prepare_gigaspeech(
    data_folder, 
    save_folder,
    splits: list,
    json_file="GigaSpeech.json",
    skip_prep: bool = False,
):
    """TODO. 
    """
    # check that `splits` input is valid
    for split in splits:
        assert split in SPLITS + TRAIN_SUBSET, f"Split {split} not recognized. Valid splits are {SPLITS + TRAIN_SUBSET}."
    
    # check that we are not using multiple train subsets
    if len(set(splits).intersection(TRAIN_SUBSET)) > 1:
        raise ValueError("You cannot use multiple train subsets. Please select only one train subset.")

    if skip_prep:
        logger.info("Skipping data preparation as `skip_prep` is set to `True`")
        return
    
    os.makedirs(save_folder, exist_ok=True)

    if skip(): # TODO: Implement skip function
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Starting data preparation...")

    check_gigaspeech_folders(data_folder, json_file)

    json_metadata = os.path.join(data_folder, json_file)
    logger.info("Creating train, dev, and test subsets.")

    logger.info(f"Starting reading {json_file}.")
    with open(json_metadata, "r") as f:
        info = json.load(f)
    logger.info(f"Reading {json_file} done.")
    
    for split in splits:
        if split in TRAIN_SUBSET:
            logger.info(f"Starting creating train.csv using {split} subset.")
            output_csv_file = os.path.join(save_folder, f"train.csv")
            create_csv(output_csv_file, info, split)
        else:
            logger.info(f"Starting creating {split.lower()}.csv using {split} subset.")
            output_csv_file = os.path.join(save_folder, f"{split.lower()}.csv")
            create_csv(output_csv_file, info, split)

def process_line(audio, split):
    if ("{" + split + "}") in audio["subsets"]:

        audio_path = os.path.join(data_folder, audio["path"])
        assert os.path.isfile(audio_path), f"File not found: {audio_path}"

        # 2. iterate over the utterances
        utterances = []
        for segment in audio["segments"]:
            text = preprocess_text(segment["text_tn"])
            if text:
                begin_time = float(segment["begin_time"])
                end_time = float(segment["end_time"])
                duration = end_time - begin_time
                utterance = GigaSpeechRow(
                    utt_id=segment["sid"],
                    audio_id=audio["aid"],
                    audio_path=str(audio_path),
                    speaker=audio["speaker"],
                    begin_time=begin_time,
                    end_time=end_time,
                    duration=duration,
                    text=text,
                )
                utterances.append(utterance)
        return utterances

def create_csv(csv_file, info, split):
    """TODO. 
    """    
    total_duration = 0.0
    nb_samples = 0
    
    line_processor = functools.partial(
        process_line,
        split=split,
    )
    
    csv_file_tmp = csv_file + ".tmp"
    with open(csv_file_tmp, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header = [
            "utt_id",
            "audio_id",
            "audio_path",
            "speaker",
            "begin_time",
            "end_time",
            "duration",
            "text",
        ]
        csv_writer.writerow(header)
        for row in parallel_map(line_processor, info["audios"]):
            if row is None:
                continue
                
            for item in row:
                csv_writer.writerow([
                    item.utt_id, 
                    item.audio_id, 
                    item.audio_path, 
                    item.speaker, 
                    str(item.begin_time), 
                    str(item.end_time), 
                    str(item.duration), 
                    item.text
                ])
                
                total_duration += item.duration
                nb_samples += 1
        
    os.replace(csv_file_tmp, csv_file)

    logger.info(f"{csv_file} succesfully created!")
    logger.info(f"Number of samples in {split} split: {nb_samples}")
    logger.info(f"Total duration of {split} split: {round(total_duration / 3600, 2)} Hours")
            
def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by removing garbage tags and replacing punctuation tags.

    Parameters
    ----------
    text : str
        The input text to be preprocessed.

    Returns
    -------
    str
        The preprocessed text with removed garbage tags and replaced punctuation tags.

    Raises
    ------
    AssertionError
        If '<' or '>' tags are found in the text after preprocessing.

    Notes
    -----
    The function iterates over predefined garbage utterance tags (GRABAGE_UTTERANCE_TAGS)
    and removes them from the input text. It then iterates over predefined punctuation tags
    (PUNCTUATION_TAGS) and replaces them with the corresponding punctuation.

    Examples
    --------
    >>> text = " DOUGLAS MCGRAY IS GOING TO BE OUR GUIDE YOU WALK THROUGH THE DOOR <COMMA> YOU SEE THE RED CARPETING <COMMA> YOU SEE SOMEONE IN A SUIT <PERIOD> THEY MAY BE GREETING YOU <PERIOD>"
    >>> preprocess_text(text)
    "douglas mcgray is going to be our guide you walk through the door, you see the red carpeting, you see someone in a suit. they may be greeting you."
    """
    # Remove garbage tags
    for tag in GRABAGE_UTTERANCE_TAGS:
        if tag in text:
            return ""
    
    # Remove punctuation tags
    for tag, punctuation in PUNCTUATION_TAGS.items():
        text = text.replace(' ' + tag, punctuation)
    
    assert "<" not in text and ">" not in text, f"Found tags in the text: {text}"
    return text.lower()


def skip():
    """TODO. 
    """
    return False

def check_gigaspeech_folders(data_folder, json_file="GigaSpeech.json", audio_folder="audio"):
    """Check if the data folder actually contains the GigaSpeech dataset.

    If it does not, an error is raised.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If GigaSpeech is not found at the specified path.
    """
    # Checking if "GigaSpeech.json" exist
    json_gigaspeech = os.path.join(data_folder, json_file)
    check_file(json_gigaspeech)
    
    # Check if audio folders exist
    for folder_subset in ["audiobook", "podcast", "youtube"]:
        audio_subset = os.path.join(data_folder, audio_folder, folder_subset)
        if not os.path.exists(audio_subset):
            err_msg = (
                "the file %s does not exist (it is expected in the "
                "Gigaspeech dataset)" % audio_subset
            )
            raise OSError(err_msg)

def check_file(path):
    # Check if file exist
    if not os.path.exists(path):
        err_msg = (
            "the opus file %s does not exist (it is expected in the "
            "Gigaspeech dataset)" % path
        )
        raise OSError(err_msg)
    

if __name__ == "__main__":
    data_folder = "/local_disk/idyie/amoumen/GigaSpeech_data/"
    save_folder = "."
    splits = ["XS", "DEV", "TEST"]
    print("HERE")
    prepare_gigaspeech(data_folder, save_folder, splits=splits)
    print("Done")