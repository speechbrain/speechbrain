import argparse
import multiprocessing as mp
import wave
from pathlib import Path

from tqdm import tqdm


def convert_to_wav(filepath):
    """
    This function converts pcm files to wav files

    Arguments
    ---------
    filepath : str
        path to the pcm file
    """

    with open(filepath, "rb") as r:
        data = r.read()
        with wave.open(str(filepath.with_suffix(".wav")), "wb") as w:
            w.setparams((1, 2, 16000, 0, "NONE", "NONE"))
            w.writeframes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath", type=str)
    parser.add_argument("--nj", type=int, default=32)
    args = parser.parse_args()

    file_list = list(Path(args.dirpath).glob("**/*.pcm"))

    pool = mp.Pool(processes=args.nj)
    with tqdm(total=len(file_list)) as pbar:
        for _ in tqdm(pool.imap_unordered(convert_to_wav, file_list)):
            pbar.update()

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
