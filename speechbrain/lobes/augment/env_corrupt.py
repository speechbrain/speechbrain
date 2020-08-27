"""
Environmental corruptions for speech signals.

Authors
 * Peter Plantinga 2020
"""
import os
import shutil
import torch
import torchaudio
from speechbrain.utils.data_utils import download_file
from speechbrain.processing.speech_augmentation import (
    AddBabble,
    AddNoise,
    AddReverb,
)

OPENRIR_URL = "http://www.openslr.org/resources/28/rirs_noises.zip"


class EnvCorrupt(torch.nn.Module):
    """Environmental Corruptions for speech signals: noise, reverb, babble.

    Arguments
    ---------
    reverb_prob : float from 0 to 1
        The probability that each batch will have reverberation applied.
    babble_prob : float from 0 to 1
        The probability that each batch will have babble added.
    noise_prob : float from 0 to 1
        The probability that each batch will have noise added.
    openrir_source : str
        If provided, prepare openrir from this source path.
    openrir_folder : str
        If provided, download and prepare openrir to this location. The
        reverberation csv and noise csv will come from here unless overridden
        by the ``reverb_csv`` or ``noise_csv`` arguments.
    openrir_max_noise_len : float
        The maximum length in seconds for a noise segment from openrir. Only
        takes effect if ``openrir_folder`` is used for noises. Cuts longer
        noises into segments equal to or less than this length.
    reverb_csv : str
        A prepared csv file for loading room impulse responses.
    noise_csv : str
        A prepared csv file for loading noise data.
    noise_cache : bool
        Whether to cache noises.
    noise_num_workers : int
        Number of workers to use for loading noises.
    babble_speaker_count : int
        Number of speakers to use for babble. Must be less than batch size.
    babble_snr_low : int
        Lowest generated SNR of reverbed signal to babble.
    babble_snr_high : int
        Highest generated SNR of reverbed signal to babble.
    noise_snr_low : int
        Lowest generated SNR of babbled signal to noise.
    noise_snr_high : int
        Highest generated SNR of babbled signal to noise.
    rir_scale_factor: float
        It compresses or dilates the given impuse response.
        If ``0 < rir_scale_factor < 1``, the impulse response is compressed
        (less reverb), while if ``rir_scale_factor > 1`` it is dilated
        (more reverb).

    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> corrupter = EnvCorrupt(babble_speaker_count=9)
    >>> feats = corrupter(inputs, torch.ones(10), init_params=True)
    """

    def __init__(
        self,
        reverb_prob=1.0,
        babble_prob=1.0,
        noise_prob=1.0,
        openrir_source=None,
        openrir_folder=None,
        openrir_max_noise_len=None,
        reverb_csv=None,
        noise_csv=None,
        noise_cache=False,
        noise_num_workers=0,
        babble_speaker_count=0,
        babble_snr_low=0,
        babble_snr_high=0,
        noise_snr_low=0,
        noise_snr_high=0,
        rir_scale_factor=1.0,
    ):
        super().__init__()

        # Download and prepare openrir
        if openrir_folder and (not reverb_csv or not noise_csv):
            open_reverb_csv = os.path.join(openrir_folder, "reverb.csv")
            open_noise_csv = os.path.join(openrir_folder, "noise.csv")
            _prepare_openrir(
                openrir_source,
                openrir_folder,
                open_reverb_csv,
                open_noise_csv,
                openrir_max_noise_len,
            )

            # Override if they aren't specified
            reverb_csv = reverb_csv or open_reverb_csv
            noise_csv = noise_csv or open_noise_csv

        # Initialize corrupters
        if reverb_csv is not None:
            self.add_reverb = AddReverb(
                reverb_prob=reverb_prob,
                csv_file=reverb_csv,
                rir_scale_factor=rir_scale_factor,
            )

        if babble_speaker_count > 0:
            self.add_babble = AddBabble(
                mix_prob=babble_prob,
                speaker_count=babble_speaker_count,
                snr_low=babble_snr_low,
                snr_high=babble_snr_high,
            )

        if noise_csv is not None:
            self.add_noise = AddNoise(
                mix_prob=noise_prob,
                csv_file=noise_csv,
                do_cache=noise_cache,
                num_workers=noise_num_workers,
                snr_low=noise_snr_low,
                snr_high=noise_snr_high,
            )

    def forward(self, waveforms, lengths, init_params=False):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort
        """
        # Augmentation
        if self.training:
            if hasattr(self, "add_reverb"):
                waveforms = self.add_reverb(waveforms, lengths)
            if hasattr(self, "add_babble"):
                waveforms = self.add_babble(waveforms, lengths)
            if hasattr(self, "add_noise"):
                waveforms = self.add_noise(waveforms, lengths)

        return waveforms


def _prepare_openrir(source, folder, reverb_csv, noise_csv, max_noise_len):
    """Prepare the openrir dataset for adding reverb and noises.

    Arguments
    ---------
    source : str
        The location of the folder containing the dataset source.
    folder : str
        The location of the folder containing the dataset.
    reverb_csv : str
        Filename for storing the prepared reverb csv.
    noise_csv : str
        Filename for storing the prepared noise csv.
    max_noise_len : float
        The maximum noise length in seconds. Noises longer
        than this will be cut into pieces.
    """

    # Download and unpack if necessary
    filepath = os.path.join(folder, "rirs_noises.zip")

    if source is not None:
        if not os.path.isdir(os.path.join(folder, "RIRS_NOISES")):
            print(f"Extracting {source} to {folder}")
            shutil.unpack_archive(source, folder)
    elif not os.path.isdir(os.path.join(folder, "RIRS_NOISES")):
        download_file(OPENRIR_URL, filepath, unpack=True)
    else:
        download_file(OPENRIR_URL, filepath)

    # Prepare reverb csv if necessary
    if not os.path.isfile(reverb_csv):
        rir_filelist = os.path.join(
            folder, "RIRS_NOISES", "real_rirs_isotropic_noises", "rir_list"
        )
        _prepare_csv(folder, rir_filelist, reverb_csv)

    # Prepare noise csv if necessary
    if not os.path.isfile(noise_csv):
        noise_filelist = os.path.join(
            folder, "RIRS_NOISES", "pointsource_noises", "noise_list"
        )
        _prepare_csv(folder, noise_filelist, noise_csv, max_noise_len)


def _prepare_csv(folder, filelist, csv_file, max_length=None):
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
    """
    with open(csv_file, "w") as w:
        w.write("ID,duration,wav,wav_format,wav_opts\n\n")
        for line in open(filelist):

            # Read file for duration/channel info
            filename = os.path.join(folder, line.split()[-1])
            signal, rate = torchaudio.load(filename)

            # Ensure only one channel
            if signal.shape[0] > 1:
                signal = signal[0]
                torchaudio.save(filename, signal, rate)

            ID, ext = os.path.basename(filename).split(".")
            duration = signal.shape[1] / rate

            # Handle long waveforms
            if max_length is not None and duration > max_length:
                # Delete old file
                os.remove(filename)
                for i in range(int(duration / max_length)):
                    start = int(max_length * i * rate)
                    stop = int(min(max_length * (i + 1), duration) * rate)
                    new_filename = filename[: -len(f".{ext}")] + f"_{i}.{ext}"
                    torchaudio.save(new_filename, signal[:, start:stop], rate)
                    csv_row = (
                        f"{ID}_{i}",
                        str((stop - start) / rate),
                        new_filename,
                        ext,
                        "\n",
                    )
                    w.write(",".join(csv_row))
            else:
                w.write(",".join((ID, str(duration), filename, ext, "\n")))
