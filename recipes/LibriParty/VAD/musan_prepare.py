import os

import torchaudio

import speechbrain as sb
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_musan(folder, music_csv, noise_csv, speech_csv, max_noise_len=None):
    """Prepare the musan dataset (music, noise, speech).

    Arguments
    ---------
    folder : str
        The location of the folder containing the dataset.
    music_csv : str
        Filename for storing the prepared music csv.
    noise_csv : str
        Filename for storing the prepared noise csv.
    speech_csv : str
        Filename for storing the prepared speech csv.
    max_noise_len : float
        The maximum noise length in seconds. Noises longer
        than this will be cut into pieces.
    """

    sub_folders = ["music", "noise", "speech"]
    csv_files = [music_csv, noise_csv, speech_csv]
    logger.info("Musan Data Preparation...")
    for sub_folder, csv_file in zip(sub_folders, csv_files):
        wav_lst = get_all_files(
            os.path.join(folder, sub_folder), match_and=[".wav"]
        )
        if not os.path.isfile(csv_file):
            logger.info(csv_file + " creation...")
            _prepare_csv(folder, wav_lst, csv_file, max_noise_len)


def _prepare_csv(folder, filelist, csv_file, max_length=None):
    """Iterate a set of wavs and write the corresponding csv file.

    Arguments
    ---------
    folder : str
        The folder relative to which the files in the list are listed.
    filelist : str
        The location of a file listing the files to be used.
    csv_file : str
        The location to use for writing the csv file.
    max_length : float
        The maximum length in seconds. Waveforms longer
        than this will be cut into pieces.
    """
    try:
        if sb.utils.distributed.if_main_process():
            with open(csv_file, "w", encoding="utf-8") as w:
                w.write("ID,duration,wav,wav_format,wav_opts\n\n")
                for line in filelist:
                    # Read file for duration/channel info
                    filename = os.path.join(folder, line.split()[-1])
                    signal, rate = torchaudio.load(filename)

                    # Ensure only one channel
                    if signal.shape[0] > 1:
                        signal = signal[0].unsqueeze(0)
                        torchaudio.save(filename, signal, rate)

                    ID, ext = os.path.basename(filename).split(".")
                    duration = signal.shape[1] / rate

                    # Handle long waveforms
                    if max_length is not None and duration > max_length:
                        # Delete old file
                        os.remove(filename)
                        for i in range(int(duration / max_length)):
                            start = int(max_length * i * rate)
                            stop = int(
                                min(max_length * (i + 1), duration) * rate
                            )
                            new_filename = (
                                filename[: -len(f".{ext}")] + f"_{i}.{ext}"
                            )
                            torchaudio.save(
                                new_filename, signal[:, start:stop], rate
                            )
                            csv_row = (
                                f"{ID}_{i}",
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
