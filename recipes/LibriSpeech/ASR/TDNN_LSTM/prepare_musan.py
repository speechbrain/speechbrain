import os
import logging
from pathlib import Path

import torchaudio
import speechbrain as sb

from speechbrain.utils.data_utils import get_all_files
from tqdm import tqdm

logger = logging.getLogger(__name__)


def prepare_musan(
    folder, 
    csv_file_paths, 
    max_noise_len=None,
    overwrite: bool=False
):
    """Prepare the musan dataset (music, noise, speech).

    Arguments
    ---------
    folder : str
        The location of the folder containing the dataset.
    csv_file_paths : str
        Filenames for storing the prepared noisy csvs (one for each musan subset).
    max_noise_len : float
        The maximum noise length in seconds. Noises longer
        than this will be cut into pieces.
    overwrite: bool
        If True then files will be overwrite even if they already exist.
    """

    logger.info("Musan Data Preparation...")
    for csv_file in csv_file_paths:
        sub_folder = Path(csv_file).stem
        wav_lst = get_all_files(
            os.path.join(folder, sub_folder), match_and=[".wav"]
        )
        if overwrite or (not os.path.isfile(csv_file)):
            logger.info(csv_file + " creation...")
            _prepare_csv(folder, wav_lst, csv_file, max_noise_len)


def _prepare_csv(folder, filelist, csv_file, max_length=None, min_length=None):
    """Iterate a set of wavs and write the corresponding csv file.
    Arguments
    ---------
    folder : str
        The folder relative to which the files in the list are listed.
    filelist : str
        The location of a file listing the files to be used.
    csvfile : str
        The location to use for writing the csv file.
    max_length : float
        The maximum length in seconds. Waveforms longer
        than this will be cut into pieces.
    min_length: float
        The minimum length in seconds. Waveforms shorter
        than this will be discarded.
    """
    try:
        if sb.utils.distributed.if_main_process():
            seen_ids = set()
            with open(csv_file, "w") as w:
                w.write("ID,duration,wav,wav_format,wav_opts\n\n")
                for line in tqdm(filelist):

                    # Read file for duration/channel info
                    filename = os.path.join(folder, line.split()[-1])
                    signal, rate = torchaudio.load(filename)

                    # Ensure only one channel
                    if signal.shape[0] > 1:
                        signal = signal[0].unsqueeze(0)
                        torchaudio.save(filename, signal, rate)

                    ID, ext = os.path.basename(filename).split(".")
                    duration = signal.shape[1] / rate
                    if ID in seen_ids:
                        continue
                    seen_ids.add(ID)

                    # Handle long waveforms
                    if max_length is not None and duration > max_length:    
                        new_dir = os.path.join(
                            os.path.dirname(filename),
                            "..",
                            os.path.basename(os.path.dirname(filename)) + "_chunked"
                        )
                        if not os.path.isdir(new_dir):
                            os.mkdir(new_dir)
                        for i in range(int(duration / max_length)):
                            new_id = f"{ID}_{i}"
                            if new_id in seen_ids:
                                continue
                            seen_ids.add(new_id)
                            start = int(max_length * i * rate)
                            stop = int(
                                min(max_length * (i + 1), duration) * rate
                            )
                            new_filename = os.path.join(
                                new_dir,
                                os.path.basename(
                                    filename[: -len(f".{ext}")] + f"_{i}.{ext}"
                                )
                            )
                            torchaudio.save(
                                new_filename, signal[:, start:stop], rate
                            )
                            csv_row = (
                                new_id,
                                str((stop - start) / rate),
                                new_filename,
                                ext,
                                "\n",
                            )
                            w.write(",".join(csv_row))
                    else:
                        w.write(
                            ",".join((ID, str(duration), filename, ext, "\n"))
                        )
    finally:
        sb.utils.distributed.ddp_barrier()