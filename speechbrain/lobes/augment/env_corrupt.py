"""
Environmental corruptions for speech signals.

Authors
 * Peter Plantinga 2020
"""
import os
import torch
import shutil
import logging
import torchaudio
import urllib.request
from speechbrain.processing.speech_augmentation import (
    AddBabble,
    AddNoise,
    AddReverb,
)

logger = logging.getLogger(__name__)
OPENRIR_URL = ("http://www.openslr.org/resources/28/rirs_noises.zip",)


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
    openrir_folder : str
        If provided, download and prepare openrir to this location. The
        reverberation csv and noise csv will come from here unless overridden
        by the ``reverb_csv`` or ``noise_csv`` arguments.
    reverb_csv : str
        A prepared csv file for loading room impulse responses.
    noise_csv : str
        A prepared csv file for loading noise data.
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
        openrir_folder=None,
        reverb_csv=None,
        noise_csv=None,
        babble_speaker_count=0,
        babble_snr_low=0,
        babble_snr_high=0,
        noise_snr_low=0,
        noise_snr_high=0,
    ):
        super().__init__()

        # Download and prepare openrir
        if openrir_folder and (not reverb_csv or not noise_csv):
            open_reverb_csv = os.path.join(openrir_folder, "reverb.csv")
            open_noise_csv = os.path.join(openrir_folder, "noise.csv")
            _prepare_openrir(openrir_folder, open_reverb_csv, open_noise_csv)

            # Override if they aren't specified
            reverb_csv = reverb_csv or open_reverb_csv
            noise_csv = noise_csv or open_noise_csv

        # Initialize corrupters
        if reverb_csv is not None:
            self.add_reverb = AddReverb(csv_file=reverb_csv)

        if babble_speaker_count > 0:
            self.add_babble = AddBabble(
                speaker_count=babble_speaker_count,
                snr_low=babble_snr_low,
                snr_high=babble_snr_high,
            )

        if noise_csv is not None:
            self.add_noise = AddNoise(
                csv_file=noise_csv,
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


def _prepare_openrir(folder, reverb_csv, noise_csv):
    """Prepare the openrir dataset for adding reverb and noises.

    Arguments
    ---------
    folder : str
        The location of the folder containing the dataset.
    reverb_csv : str
        Filename for storing the prepared reverb csv.
    noise_csv : str
        Filename for storing the prepared noise csv.
    """

    # Download if necessary
    filepath = os.path.join(folder, "rirs_noises.zip")
    if not os.path.isfile(filepath):
        logger.info(f"Downloading {OPENRIR_URL} to {filepath}")
        with urllib.request.urlopen(OPENRIR_URL) as response:
            with open(filepath, "wb") as w:
                shutil.copyfileobj(response, w)

    # Unpack if necessary
    if not os.path.isdir(os.path.join(folder, "RIRS_NOISES")):
        logger.info(f"Extracting {filepath}")
        shutil.unpack_archive(filepath, folder)

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
        _prepare_csv(folder, noise_filelist, noise_csv)


def _prepare_csv(folder, filelist, csv_file):
    """Iterate a set of wavs and write the corresponding csv file.

    Arguments
    ---------
    folder : str
        The folder relative to which the files in the list are listed.
    filelist : str
        The location of a file listing the files to be used.
    csvfile : str
        The location to use for writing the csv file.
    """
    with open(csv_file, "w") as w:
        w.write("ID,duration,wav,wav_format,wav_opts\n\n")
        for line in open(filelist):

            # Read file for duration/channel info
            filename = os.path.join(folder, line.split()[-1])
            signal, rate = torchaudio.load(filename)

            # Ensure only one channel
            if signal.shape[0] > 1:
                torchaudio.save(filename, signal[0], rate)

            # Write CSV line
            ID = os.path.splitext(os.path.basename(filename))[0]
            duration = str(len(signal) / rate)
            w.write(",".join((ID, duration, filename, "wav", "\n")))
