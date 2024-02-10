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
from dataclasses import dataclass
import functools
from speechbrain.utils.parallel import parallel_map

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
    wav_id: str # audio[aid]
    wav_path: str
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

    print("Starting reading JSON file.")
    with open(json_metadata, "r") as f:
        info = json.load(f)
    
    print("Reading JSON file done.")
    for split in splits:
        print(f"Creating CSV for {split} split.")
        output_csv_file = os.path.join(save_folder, f"{split}.csv")
        create_csv(output_csv_file, info, split, data_folder)
        exit()

def process_line(audio, split):
    if ("{" + split + "}") in audio["subsets"]:
        print("VALID AUDIO FILE")

        wav_path = os.path.join(data_folder, audio["path"])
        assert os.path.isfile(wav_path), f"File not found: {wav_path}"

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
                    wav_id=audio["aid"],
                    wav_path=str(wav_path),
                    speaker=audio["speaker"],
                    begin_time=begin_time,
                    end_time=end_time,
                    duration=duration,
                    text=text,
                )
                utterances.append(utterance)
        return utterances

def create_csv(csv_file, info, split, data_folder):
    """TODO. 
    """
    ret = []
    print("inside create_csv")
    logger.info(f"Creating CSV for {split} split.")
    csv_file_tmp = csv_file + ".tmp"
    total_duration = 0.0
    
    line_processor = functools.partial(
        process_line,
        split=split,
    )

    print("audios = ", len(info["audios"]))
    for audio in info["audios"]:
        
        # 1. Check if the audio is part of the "subsets". One audio can be part of multiple subsets
        # such as "{XL}" and "{L}".
        if ("{" + split + "}") in audio["subsets"]:
            print("VALID AUDIO FILE")

            wav_path = os.path.join(data_folder, audio["path"])
            assert os.path.isfile(wav_path), f"File not found: {wav_path}"

            # 2. iterate over the utterances
            utterances = []
            print("segments = ", len(audio["segments"]))
            for segment in audio["segments"]:
                text = preprocess_text(segment["text_tn"])
                if text:
                    begin_time = float(segment["begin_time"])
                    end_time = float(segment["end_time"])
                    duration = end_time - begin_time
                    utterance = GigaSpeechRow(
                        utt_id=segment["sid"],
                        wav_id=audio["aid"],
                        wav_path=str(wav_path),
                        speaker=audio["speaker"],
                        begin_time=begin_time,
                        end_time=end_time,
                        duration=duration,
                        text=text,
                    )
            total_duration += duration
            ret.append(utterances)
            exit()
    
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