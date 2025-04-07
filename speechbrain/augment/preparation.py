"""Library for Downloading and Preparing Datasets for Data Augmentation,
This library provides functions for downloading datasets from the web and
preparing the necessary CSV data manifest files for use by data augmenters.

Authors:
* Mirco Ravanelli 2023

"""

import os

import torchaudio

from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.utils.distributed import main_process_only
from speechbrain.utils.logger import get_logger

# Logger init
logger = get_logger(__name__)


@main_process_only
def prepare_dataset_from_URL(URL, dest_folder, ext, csv_file, max_length=None):
    """Downloads a dataset containing recordings (e.g., noise sequences)
    from the provided URL and prepares the necessary CSV files for use by the noise augmenter.

    Arguments
    ---------
    URL : str
        The URL of the dataset to download.
    dest_folder : str
        The local folder where the noisy dataset will be downloaded.
    ext : str
        File extensions to search for within the downloaded dataset.
    csv_file : str
        The path to store the prepared noise CSV file.
    max_length : float
        The maximum length in seconds.
        Recordings longer than this will be automatically cut into pieces.
    """

    # Download and unpack if necessary
    data_file = os.path.join(dest_folder, "data.zip")

    if not os.path.isdir(dest_folder):
        download_file(URL, data_file, unpack=True)
    else:
        download_file(URL, data_file)

    # Prepare noise csv if necessary
    if not os.path.isfile(csv_file):
        filelist = get_all_files(dest_folder, match_and=["." + ext])
        prepare_csv(filelist, csv_file, max_length)


@main_process_only
def prepare_csv(filelist, csv_file, max_length=None):
    """Iterate a set of wavs and write the corresponding csv file.

    Arguments
    ---------
    filelist : str
        A list containing the paths of files of interest.
    csv_file : str
        The path to store the prepared noise CSV file.
    max_length : float
        The maximum length in seconds.
        Recordings longer than this will be automatically cut into pieces.
    """
    try:
        write_csv(filelist, csv_file, max_length)
    except Exception as e:
        # Handle the exception or log the error message
        logger.error("Exception:", exc_info=(e))

        # Delete the file if something fails
        if os.path.exists(csv_file):
            os.remove(csv_file)


@main_process_only
def write_csv(filelist, csv_file, max_length=None):
    """
    Iterate through a list of audio files and write the corresponding CSV file.

    Arguments
    ---------
    filelist : list of str
        A list containing the paths of audio files of interest.
    csv_file : str
        The path where to store the prepared noise CSV file.
    max_length : float (optional)
        The maximum recording length in seconds.
        Recordings longer than this will be automatically cut into pieces.
    """
    with open(csv_file, "w", encoding="utf-8") as w:
        w.write("ID,duration,wav,wav_format,wav_opts\n")
        for i, filename in enumerate(filelist):
            _write_csv_row(w, filename, i, max_length)


def _write_csv_row(w, filename, index, max_length):
    """
    Write a single row to the CSV file based on the audio file information.

    Arguments
    ---------
    w : file
        The open CSV file for writing.
    filename : str
        The path to the audio file.
    index : int
        The index of the audio file in the list.
    max_length : float (optional)
        The maximum recording length in seconds.
    """
    signal, rate = torchaudio.load(filename)
    signal = _ensure_single_channel(signal, filename, rate)

    ID, ext = os.path.basename(filename).split(".")
    duration = signal.shape[1] / rate

    if max_length is not None and duration > max_length:
        _handle_long_waveform(
            w, filename, ID, ext, signal, rate, duration, max_length, index
        )
    else:
        _write_short_waveform_csv(w, ID, ext, duration, filename, index)


def _ensure_single_channel(signal, filename, rate):
    """
    Ensure that the audio signal has only one channel.

    Arguments
    ---------
    signal : torch.Tensor
        The audio signal.
    filename : str
        The path to the audio file.
    rate : int
        The sampling frequency of the signal.

    Returns
    -------
    signal : Torch.Tensor
        The audio signal with a single channel.
    """
    if signal.shape[0] > 1:
        signal = signal[0].unsqueeze(0)
        torchaudio.save(filename, signal, rate)
    return signal


def _handle_long_waveform(
    w, filename, ID, ext, signal, rate, duration, max_length, index
):
    """
    Handle long audio waveforms by cutting them into pieces and writing to the CSV.

    Arguments
    ---------
    w : file
        The open CSV file for writing.
    filename : str
        The path to the audio file.
    ID : str
        The unique identifier for the audio.
    ext :  str
        The audio file extension.
    signal : torch.Tensor
        The audio signal.
    rate : int
        The audio sample rate.
    duration :  float
        The duration of the audio in seconds.
    max_length :  float
        The maximum recording length in seconds.
    index : int
        The index of the audio file in the list.
    """
    os.remove(filename)
    for j in range(int(duration / max_length)):
        start = int(max_length * j * rate)
        stop = int(min(max_length * (j + 1), duration) * rate)
        ext = filename.split(".")[1]
        new_filename = filename.replace("." + ext, "_" + str(j) + "." + ext)

        torchaudio.save(new_filename, signal[:, start:stop], rate)
        csv_row = (
            f"{ID}_{index}_{j}",
            str((stop - start) / rate),
            new_filename,
            ext,
            "\n",
        )
        w.write(",".join(csv_row))


def _write_short_waveform_csv(w, ID, ext, duration, filename, index):
    """
    Write a CSV row for a short audio waveform.

    Arguments
    ---------
    w : file
        The open CSV file for writing.
    ID : str
        The unique identifier for the audio.
    ext : str
        The audio file extension.
    duration : float
        The duration of the audio in seconds.
    filename : str
        The path to the audio file.
    index : int
        The index of the audio file in the list.
    """
    w.write(",".join((f"{ID}_{index}", str(duration), filename, ext, "\n")))
