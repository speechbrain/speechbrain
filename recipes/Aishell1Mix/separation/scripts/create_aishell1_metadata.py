import argparse
import glob
import os

import pandas as pd
import soundfile as sf
from tqdm import tqdm

# Global parameter
# We will filter out files shorter than that
NUMBER_OF_SECONDS = 3
# In aishell1 all the sources are at 16K Hz
RATE = 16000

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--aishell1_dir",
    type=str,
    default="/youraishell1path//aishell1/data_aishell",
    help="Path to aishell1 root directory",
)


def main(args):
    aishell1_dir = args.aishell1_dir
    # Create aishell1 metadata directory
    aishell1_md_dir = os.path.join(aishell1_dir, "metadata")
    os.makedirs(aishell1_md_dir, exist_ok=True)
    create_aishell1_metadata(aishell1_dir, aishell1_md_dir)


def create_aishell1_metadata(aishell1_dir, md_dir):
    """Generate metadata corresponding to downloaded data in aishell1"""
    # Get speakers metadata
    speakers_metadata = create_speakers_dataframe(aishell1_dir)
    filename2transcript = {}
    with open(
        os.path.join(aishell1_dir, "transcript/aishell_transcript_v0.8.txt"),
        "r",
        encoding="utf-8",
    ) as f:
        lines = f.readlines()
        for line in lines:
            key = line.split()[0]
            value = " ".join(line.split()[1:])
            filename2transcript[key] = value

    # Check for already generated files and generate files accordingly
    aishell1_dir = os.path.join(aishell1_dir, "wav")
    dir_to_process = check_already_generated(md_dir, aishell1_dir)

    # Go through each directory and create associated metadata
    for ldir in dir_to_process:
        # Generate the dataframe relative to the directory
        dir_metadata = create_aishell1_dataframe(
            aishell1_dir, ldir, speakers_metadata, filename2transcript
        )
        # Filter out files that are shorter than 3s
        num_samples = NUMBER_OF_SECONDS * RATE
        dir_metadata = dir_metadata[dir_metadata["length"] >= num_samples]
        # Sort the dataframe according to ascending Length
        dir_metadata = dir_metadata.sort_values("length")
        # Write the dataframe in a .csv in the metadata directory
        save_path = os.path.join(md_dir, ldir + ".csv")
        dir_metadata.to_csv(save_path, index=False)


def create_speakers_dataframe(aishell1_dir):
    """Read metadata from the aishell1 dataset and collect infos
    about the speakers"""
    print("Reading speakers metadata")
    # Read SPEAKERS.TXT and create a dataframe
    speakers_metadata_path = os.path.join(
        aishell1_dir, "../resource_aishell/speaker.info"
    )
    speakers_metadata = pd.read_csv(
        speakers_metadata_path,
        sep=" ",
        names=["speaker_ID", "sex"],
        dtype="str",
    )

    speakers_metadata["speaker_ID"] = "S" + speakers_metadata["speaker_ID"]

    return speakers_metadata


def check_already_generated(md_dir, aishell1_dir):
    # If md_dir already exists then check the already generated files
    already_generated_csv = os.listdir(md_dir)
    # Save the already generated files names
    already_generated_csv = [f.split(".")[0] for f in already_generated_csv]
    # Possible directories in the original aishell1
    original_aishell1_dirs = ["dev", "test", "train"]
    # Actual directories extracted in your aishell1 version
    actual_aishell1_dirs = set(next(os.walk(aishell1_dir))[1]) & set(
        original_aishell1_dirs
    )
    # Actual directories that haven't already been processed
    not_already_processed_directories = list(
        set(actual_aishell1_dirs) - set(already_generated_csv)
    )
    return not_already_processed_directories


def create_aishell1_dataframe(
    aishell1_dir, subdir, speakers_md, filename2transcript
):
    """Generate a dataframe that gather infos about the sound files in a
    aishell1 subdirectory"""

    print(
        f"Creating {subdir} metadata file in "
        f"{os.path.join(aishell1_dir, 'metadata')}"
    )
    # Get the current directory path
    dir_path = os.path.join(aishell1_dir, subdir)
    # Recursively look for .flac files in current directory
    sound_paths = glob.glob(os.path.join(dir_path, "**/*.wav"), recursive=True)
    # Create the dataframe corresponding to this directory
    dir_md = pd.DataFrame(
        columns=[
            "speaker_ID",
            "sex",
            "subset",
            "length",
            "origin_path",
            "transcript",
        ]
    )

    # Go through the sound file list
    for sound_path in tqdm(sound_paths, total=len(sound_paths)):
        # Get the ID of the speaker
        spk_id = sound_path.split("/")[-2]
        # Find Sex according to speaker ID in aishell1 metadata
        sex = speakers_md[speakers_md["speaker_ID"] == spk_id].iat[0, 1]
        # Find subset according to speaker ID in aishell1 metadata
        subset = subdir
        # Get its length
        length = len(sf.SoundFile(sound_path))
        # Get the sound file relative path
        rel_path = os.path.relpath(sound_path, aishell1_dir)
        # Get the transcript
        filename = os.path.basename(sound_path).split(".")[0]
        if filename not in filename2transcript:
            continue
        transcript = filename2transcript[filename]
        # Add information to the dataframe
        dir_md.loc[len(dir_md)] = [
            spk_id,
            sex,
            subset,
            length,
            rel_path,
            transcript,
        ]
    return dir_md


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
