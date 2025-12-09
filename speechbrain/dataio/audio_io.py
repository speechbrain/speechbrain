"""
Lightweight soundfile-based audio I/O compatibility layer.

This module provides a minimal compatibility wrapper for audio I/O operations
using soundfile (pysoundfile) library, replacing torchaudio's load, save, and
info functions.

Example
-------
>>> from speechbrain.dataio import audio_io
>>> import torch
>>> # Save audio file
>>> waveform = torch.randn(1, 16000)
>>> tmpdir = getfixture("tmpdir")
>>> audio_io.save(tmpdir / "example.wav", waveform, 16000)
>>> # Load audio file
>>> audio, sr = audio_io.load(tmpdir / "example.wav")
>>> # Get audio metadata
>>> info = audio_io.info(tmpdir / "example.wav")
>>> info.duration
1.0

Authors
 * Peter Plantinga 2025
"""

import dataclasses

import numpy as np
import soundfile as sf
import torch


@dataclasses.dataclass
class AudioInfo:
    """Container for audio file metadata, compatible with torchaudio.info output.

    Attributes
    ----------
    sample_rate : int
        Sample rate of the audio file.
    frames : int
        Total number of frames in the audio file.
    channels : int
        Number of audio channels.
    subtype : str
        Audio subtype/encoding (e.g., 'PCM_16', 'PCM_24').
    format : str
        Container format (e.g., 'WAV', 'FLAC').
    """

    sample_rate: int
    frames: int
    channels: int
    subtype: str
    format: str

    @property
    def num_frames(self):
        """Alias for frames for compatibility."""
        return self.frames

    @property
    def num_channels(self):
        """Alias for channels for compatibility."""
        return self.channels

    @property
    def duration(self):
        """Calculate duration in seconds."""
        return self.frames / self.sample_rate if self.sample_rate > 0 else 0.0


def load(
    path,
    *,
    channels_first=True,
    dtype=None,
    always_2d=True,
    frame_offset=0,
    num_frames=-1,
):
    """Load audio file using soundfile.

    Arguments
    ---------
    path : str
        Path to the audio file.
    channels_first : bool
        If True, returns tensor with shape (channels, frames).
        If False, returns tensor with shape (frames, channels).
        Ignored if `always_2d` is False and input is mono.
        Default: True.
    dtype : torch.dtype, optional
        Data type for the output tensor. Respects default torch type.
        If the dtype is not one of the available dtypes in soundfile, loads
        with float32 first and then converts to the requested dtype.
    always_2d : bool
        If True, always return a 2D tensor even for mono audio.
        If False, mono audio returns a 1D tensor (frames,).
        Default: True.
    frame_offset : int
        Number of frames to skip at the start of the file. Default: 0.
    num_frames : int
        Number of frames to read. If -1, reads to the end of the file. Default: -1.

    Returns
    -------
    tensor : torch.Tensor
        Audio waveform as a tensor.
    sample_rate : int
        Sample rate of the audio file.
    """
    try:
        # Compute type for loading
        dtype = dtype or torch.get_default_dtype()
        _, dtype_string = str(dtype).split(".")

        # If the selected dtype is not a valid soundfile type, just use float32
        if dtype_string not in sf._ffi_types:
            dtype_string = "float32"

        # Read audio file - soundfile returns (frames, channels) or (frames,) for mono
        audio_np, sample_rate = sf.read(
            path,
            start=frame_offset,
            frames=num_frames,
            dtype=dtype_string,
            always_2d=always_2d,
        )

        # Convert to torch tensor
        audio = torch.from_numpy(audio_np).to(dtype)

        # Convert from (frames, channels) to (channels, frames)
        if audio.ndim == 2 and channels_first:
            audio = audio.transpose(0, 1)

        return audio, int(sample_rate)

    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {path}: {e}") from e


def save(path, src, sample_rate, channels_first=True, subtype=None):
    """Save audio to file using soundfile.

    Arguments
    ---------
    path : str
        Path where to save the audio file.
    src : torch.Tensor or numpy.ndarray
        Audio waveform. Can be:
        - 1D tensor/array: (frames,) - mono
        - 2D tensor/array:
            - (channels, frames) if channels_first=True
            - (frames, channels) if channels_first=False
    sample_rate : int
        Sample rate for the audio file.
    channels_first : bool
        If True, input is assumed to be (channels, frames)
        If False, input is assumed to be (frames, channels).
        Ignored if input is 1D tensor/array.
        Default: True.
    subtype : str, optional
        Audio encoding subtype (e.g., 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT').
        If None, soundfile will choose an appropriate subtype based on the file format.
        Default: None.
    """
    try:
        # Convert to numpy if needed
        if isinstance(src, torch.Tensor):
            audio_np = src.detach().cpu().numpy()
        else:
            audio_np = np.asarray(src)

        # Convert to (frames, channels) if channels_first is True
        if audio_np.ndim == 2 and channels_first:
            audio_np = audio_np.T

        if audio_np.ndim not in [1, 2]:
            raise ValueError(
                f"Unsupported audio shape: {audio_np.shape}. "
                "Expected 1D frames or 2D channels and frames."
            )

        sf.write(path, audio_np, sample_rate, subtype=subtype)

    except Exception as e:
        raise RuntimeError(f"Failed to save audio to {path}: {e}") from e


def info(path):
    """Get audio file metadata using soundfile.

    Arguments
    ---------
    path : str
        Path to the audio file.

    Returns
    -------
    AudioInfo
        Object containing audio metadata (sample_rate, frames, channels,
        subtype, format, duration).
    """
    try:
        file_info = sf.info(path)
        return AudioInfo(
            sample_rate=file_info.samplerate,
            frames=file_info.frames,
            channels=file_info.channels,
            subtype=file_info.subtype,
            format=file_info.format,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to get info for {path}: {e}") from e


def list_audio_backends():
    """List available audio backends.

    Returns
    -------
    list of str
        List of available backend names. Currently only ['soundfile'].
    """
    return ["soundfile"]
