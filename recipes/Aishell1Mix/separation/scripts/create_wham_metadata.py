import os
import argparse
import soundfile as sf
import pandas as pd
import glob
from tqdm import tqdm

# Global parameter
# We will filter out files shorter than that
NUMBER_OF_SECONDS = 3
# In WHAM! all the sources are at 16K Hz
RATE = 16000

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--wham_dir",
    type=str,
    default="/yourwhampath/wham_noise",
    help="Path to wham_noise root directory",
)


def main(args):
    wham_noise_dir = args.wham_dir
    # Create wham_noise metadata directory
    wham_noise_md_dir = os.path.join(wham_noise_dir, "metadata")
    os.makedirs(wham_noise_md_dir, exist_ok=True)
    create_wham_noise_metadata(wham_noise_dir, wham_noise_md_dir)


def create_wham_noise_metadata(wham_noise_dir, md_dir):
    """ Generate metadata corresponding to downloaded data in wham_noise """

    # Check already generated files
    not_already_processed_dir = check_already_generated(md_dir)
    # Go through each directory and create associated metadata
    for ldir in not_already_processed_dir:
        # Generate the dataframe relative to the directory
        dir_metadata = create_wham_noise_dataframe(wham_noise_dir, ldir)
        # Sort the dataframe according to ascending Length
        dir_metadata = dir_metadata.sort_values("length")
        # Write the dataframe in a .csv in the metadata directory
        if ldir == "tt":
            name = "test"
        elif ldir == "cv":
            name = "dev"
        else:
            name = "train"
        # Filter out files that are shorter than 3s
        num_samples = NUMBER_OF_SECONDS * RATE
        dir_metadata = dir_metadata[dir_metadata["length"] >= num_samples]
        # Create save path
        save_path = os.path.join(md_dir, name + ".csv")
        print(f"Medatada file created in {save_path}")
        dir_metadata.to_csv(save_path, index=False)


def check_already_generated(md_dir):
    """ Check if files have already been generated """
    # Get the already generated files
    already_generated_csv = os.listdir(md_dir)
    # Data directories in wham_noise
    wham_noise_dirs = ["cv", "tr", "tt"]
    # Save the already data directories names
    already_processed_dir = [
        f.replace("test", "tt")
        .replace("train", "tr")
        .replace("dev", "cv")
        .replace(".csv", "")
        for f in already_generated_csv
    ]
    # Actual directories that haven't already been processed
    not_already_processed_dir = list(
        set(wham_noise_dirs) - set(already_processed_dir)
    )
    return not_already_processed_dir


def create_wham_noise_dataframe(wham_noise_dir, subdir):
    """ Generate a dataframe that gather infos about the sound files in a
    wham_noise subdirectory """

    print(f"Processing files from {subdir} dir")
    # Get the current directory path
    dir_path = os.path.join(wham_noise_dir, subdir)
    # Recursively look for .wav files in current directory
    sound_paths = glob.glob(os.path.join(dir_path, "**/*.wav"), recursive=True)
    # Create the dataframe corresponding to this directory
    dir_md = pd.DataFrame(
        columns=["noise_ID", "subset", "length", "augmented", "origin_path"]
    )

    # Go through the sound file list
    for sound_path in tqdm(sound_paths, total=len(sound_paths)):
        # Get the ID of the noise file
        noise_id = os.path.split(sound_path)[1]
        # Get its length
        length = len(sf.SoundFile(sound_path))
        augment = False
        if "sp08" in sound_path or "sp12" in sound_path:
            augment = True
        # Get the sound file relative path
        rel_path = os.path.relpath(sound_path, wham_noise_dir)
        # Add information to the dataframe
        dir_md.loc[len(dir_md)] = [noise_id, subdir, length, augment, rel_path]
    return dir_md


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
