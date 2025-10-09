"""Specifies the inference interfaces for Text-To-Speech (TTS) modules.

Authors:
 * Aku Rouhe 2021
 * Peter Plantinga 2021
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Titouan Parcollet 2021
 * Abdel Heba 2021
 * Andreas Nautsch 2022, 2023
 * Pooneh Mousavi 2023
 * Sylvain de Langen 2023
 * Adel Moumen 2023
 * Pradnya Kandarkar 2023
"""

import torch

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.inference.interfaces import Pretrained
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class HIFIGAN(Pretrained):
    """
    A ready-to-use wrapper for HiFiGAN (mel_spec -> waveform).

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> tmpdir_vocoder = getfixture("tmpdir") / "vocoder"
    >>> hifi_gan = HIFIGAN.from_hparams(
    ...     source="speechbrain/tts-hifigan-ljspeech", savedir=tmpdir_vocoder
    ... )
    >>> mel_specs = torch.rand(2, 80, 298)
    >>> waveforms = hifi_gan.decode_batch(mel_specs)
    >>> # You can use the vocoder coupled with a TTS system
    >>>	# Initialize TTS (tacotron2)
    >>> tmpdir_tts = getfixture("tmpdir") / "tts"
    >>> from speechbrain.inference.TTS import Tacotron2
    >>>	tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir=tmpdir_tts)
    >>>	# Running the TTS
    >>>	mel_output, mel_length, alignment = tacotron2.encode_text("Mary had a little lamb")
    >>>	# Running Vocoder (spectrogram-to-waveform)
    >>>	waveforms = hifi_gan.decode_batch(mel_output)
    """

    HPARAMS_NEEDED = ["generator"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer = self.hparams.generator.inference
        self.first_call = True

    def decode_batch(self, spectrogram, mel_lens=None, hop_len=None):
        """Computes waveforms from a batch of mel-spectrograms

        Arguments
        ---------
        spectrogram: torch.Tensor
            Batch of mel-spectrograms [batch, mels, time]
        mel_lens: torch.tensor
            A list of lengths of mel-spectrograms for the batch
            Can be obtained from the output of Tacotron/FastSpeech
        hop_len: int
            hop length used for mel-spectrogram extraction
            should be the same value as in the .yaml file

        Returns
        -------
        waveforms: torch.Tensor
            Batch of mel-waveforms [batch, 1, time]
        """
        # Prepare for inference by removing the weight norm
        if self.first_call:
            self.hparams.generator.remove_weight_norm()
            self.first_call = False
        with torch.no_grad():
            waveform = self.infer(spectrogram.to(self.device))

        # Mask the noise caused by padding during batch inference
        if mel_lens is not None and hop_len is not None:
            waveform = self.mask_noise(waveform, mel_lens, hop_len)

        return waveform

    def mask_noise(self, waveform, mel_lens, hop_len):
        """Mask the noise caused by padding during batch inference

        Arguments
        ---------
        waveform: torch.tensor
            Batch of generated waveforms [batch, 1, time]
        mel_lens: torch.tensor
            A list of lengths of mel-spectrograms for the batch
            Can be obtained from the output of Tacotron/FastSpeech
        hop_len: int
            hop length used for mel-spectrogram extraction
            same value as in the .yaml file

        Returns
        -------
        waveform: torch.tensor
            Batch of waveforms without padded noise [batch, 1, time]
        """
        waveform = waveform.squeeze(1)
        # the correct audio length should be hop_len * mel_len
        mask = length_to_mask(
            mel_lens * hop_len, waveform.shape[1], device=waveform.device
        ).bool()
        waveform.masked_fill_(~mask, 0.0)
        return waveform.unsqueeze(1)

    def decode_spectrogram(self, spectrogram):
        """Computes waveforms from a single mel-spectrogram

        Arguments
        ---------
        spectrogram: torch.Tensor
            mel-spectrogram [mels, time]

        Returns
        -------
        waveform: torch.Tensor
            waveform [1, time]
        audio can be saved by:
        >>> import torchaudio
        >>> waveform = torch.rand(1, 666666)
        >>> sample_rate = 22050
        >>> torchaudio.save(
        ...     str(getfixture("tmpdir") / "test.wav"), waveform, sample_rate
        ... )
        """
        if self.first_call:
            self.hparams.generator.remove_weight_norm()
            self.first_call = False
        with torch.no_grad():
            waveform = self.infer(spectrogram.unsqueeze(0).to(self.device))
        return waveform.squeeze(0)

    def forward(self, spectrogram):
        "Decodes the input spectrograms"
        return self.decode_batch(spectrogram)


class DiffWaveVocoder(Pretrained):
    """
    A ready-to-use inference wrapper for DiffWave as vocoder.
    The wrapper allows to perform generative tasks:
        locally-conditional generation: mel_spec -> waveform

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.
    """

    HPARAMS_NEEDED = ["diffusion"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.hparams, "diffwave"):
            self.infer = self.hparams.diffusion.inference
        else:
            raise NotImplementedError

    def decode_batch(
        self,
        mel,
        hop_len,
        mel_lens=None,
        fast_sampling=False,
        fast_sampling_noise_schedule=None,
    ):
        """Generate waveforms from spectrograms

        Arguments
        ---------
        mel: torch.tensor
            spectrogram [batch, mels, time]
        hop_len: int
            Hop length during mel-spectrogram extraction
            Should be the same value as in the .yaml file
            Used to determine the output wave length
            Also used to mask the noise for vocoding task
        mel_lens: torch.tensor
            Used to mask the noise caused by padding
            A list of lengths of mel-spectrograms for the batch
            Can be obtained from the output of Tacotron/FastSpeech
        fast_sampling: bool
            whether to do fast sampling
        fast_sampling_noise_schedule: list
            the noise schedules used for fast sampling
        Returns
        -------
        waveforms: torch.tensor
            Batch of mel-waveforms [batch, 1, time]

        """
        with torch.no_grad():
            waveform = self.infer(
                unconditional=False,
                scale=hop_len,
                condition=mel.to(self.device),
                fast_sampling=fast_sampling,
                fast_sampling_noise_schedule=fast_sampling_noise_schedule,
            )

        # Mask the noise caused by padding during batch inference
        if mel_lens is not None and hop_len is not None:
            waveform = self.mask_noise(waveform, mel_lens, hop_len)
        return waveform

    def mask_noise(self, waveform, mel_lens, hop_len):
        """Mask the noise caused by padding during batch inference

        Arguments
        ---------
        waveform: torch.tensor
            Batch of generated waveforms [batch, 1, time]
        mel_lens: torch.tensor
            A list of lengths of mel-spectrograms for the batch
            Can be obtained from the output of Tacotron/FastSpeech
        hop_len: int
            hop length used for mel-spectrogram extraction
            same value as in the .yaml file

        Returns
        -------
        waveform: torch.tensor
            Batch of waveforms without padded noise [batch, 1, time]
        """
        waveform = waveform.squeeze(1)
        # the correct audio length should be hop_len * mel_len
        mask = length_to_mask(
            mel_lens * hop_len, waveform.shape[1], device=waveform.device
        ).bool()
        waveform.masked_fill_(~mask, 0.0)
        return waveform.unsqueeze(1)

    def decode_spectrogram(
        self,
        spectrogram,
        hop_len,
        fast_sampling=False,
        fast_sampling_noise_schedule=None,
    ):
        """Computes waveforms from a single mel-spectrogram

        Arguments
        ---------
        spectrogram: torch.tensor
            mel-spectrogram [mels, time]
        hop_len: int
            hop length used for mel-spectrogram extraction
            same value as in the .yaml file
        fast_sampling: bool
            whether to do fast sampling
        fast_sampling_noise_schedule: list
            the noise schedules used for fast sampling

        Returns
        -------
        waveform: torch.tensor
            waveform [1, time]

        audio can be saved by:
        >>> import torchaudio
        >>> waveform = torch.rand(1, 666666)
        >>> sample_rate = 22050
        >>> torchaudio.save(
        ...     str(getfixture("tmpdir") / "test.wav"), waveform, sample_rate
        ... )
        """
        with torch.no_grad():
            waveform = self.infer(
                unconditional=False,
                scale=hop_len,
                condition=spectrogram.unsqueeze(0).to(self.device),
                fast_sampling=fast_sampling,
                fast_sampling_noise_schedule=fast_sampling_noise_schedule,
            )
        return waveform.squeeze(0)

    def forward(self, spectrogram):
        """Decodes the input spectrograms"""
        return self.decode_batch(spectrogram)


class UnitHIFIGAN(Pretrained):
    """
    A ready-to-use wrapper for Unit HiFiGAN (discrete units -> waveform).

    Arguments
    ---------
    *args : tuple
        See `Pretrained`
    **kwargs : dict
        See `Pretrained`

    Example
    -------
    >>> tmpdir_vocoder = getfixture("tmpdir") / "vocoder"
    >>> hifi_gan = UnitHIFIGAN.from_hparams(
    ...     source="speechbrain/hifigan-hubert-l1-3-7-12-18-23-k1000-LibriTTS",
    ...     savedir=tmpdir_vocoder,
    ... )
    >>> codes = torch.randint(0, 99, (100, 1))
    >>> waveform = hifi_gan.decode_unit(codes)
    """

    HPARAMS_NEEDED = ["generator"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer = self.hparams.generator.inference
        self.first_call = True
        # Temporary fix for mapping indices from the range [0, k] to [1, k+1]
        self.tokenize = True

    def decode_batch(self, units, spk=None):
        """Computes waveforms from a batch of discrete units

        Arguments
        ---------
        units: torch.tensor
            Batch of discrete units [batch, codes]
        spk: torch.tensor
            Batch of speaker embeddings [batch, spk_dim]

        Returns
        -------
        waveforms: torch.tensor
            Batch of mel-waveforms [batch, 1, time]
        """
        # Remove weight norm for inference if it's the first call
        if self.first_call:
            self.hparams.generator.remove_weight_norm()
            self.first_call = False

        # Ensure that the units sequence has a length of at least 3
        if units.size(1) < 3:
            raise ValueError(
                "The 'units' argument should have a length of at least 3 because of padding size."
            )

        # Increment units if tokenization is enabled
        if self.tokenize:
            units += 1
        if spk is not None:
            spk = spk.to(self.device)
        with torch.no_grad():
            waveform = self.infer(units.to(self.device), spk=spk)
        return waveform

    def decode_unit(self, units, spk=None):
        """Computes waveforms from a single sequence of discrete units
        Arguments
        ---------
        units: torch.tensor
            codes: [time]
        spk: torch.tensor
            spk: [spk_dim]
        Returns
        -------
        waveform: torch.tensor
            waveform [1, time]
        """
        # Remove weight norm for inference if it's the first call
        if self.first_call:
            self.hparams.generator.remove_weight_norm()
            self.first_call = False

        # Ensure that the units sequence has a length of at least 4
        if units.size(0) < 4:
            raise ValueError(
                "The 'units' argument should have a length of at least 4 because of padding size."
            )

        # Increment units if tokenization is enabled
        if self.tokenize:
            units = units + 1
        if spk is not None:
            spk = spk.unsqueeze(0).to(self.device)
        with torch.no_grad():
            waveform = self.infer(units.unsqueeze(0).to(self.device), spk=spk)
        return waveform.squeeze(0)

    def forward(self, units, spk=None):
        "Decodes the input units"
        return self.decode_batch(units, spk=spk)
