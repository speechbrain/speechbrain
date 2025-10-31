"""
Lightweight soundfile-based audio I/O compatibility layer.

This module provides a minimal compatibility wrapper for audio I/O operations
using soundfile (pysoundfile) library, replacing torchaudio's load, save, and
info functions.

Example
-------
>>> from speechbrain.dataio import audio_io
>>> import torch
>>> # Load audio file
>>> audio, sr = audio_io.load("example.wav")  # doctest: +SKIP
>>> # Get audio metadata
>>> info = audio_io.info("example.wav")  # doctest: +SKIP
>>> print(info.sample_rate, info.duration)  # doctest: +SKIP
>>> # Save audio file
>>> waveform = torch.randn(1, 16000)
>>> tmpdir = getfixture("tmpdir")  # doctest: +SKIP
>>> audio_io.save(tmpdir / "output.wav", waveform, 16000)  # doctest: +SKIP

Authors
 * SpeechBrain Contributors 2025
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


def load(path, *, channels_first=True, dtype=torch.float32, always_2d=True, 
         frame_offset=0, num_frames=-1):
    """Load audio file using soundfile.
    
    Arguments
    ---------
    path : str
        Path to the audio file.
    channels_first : bool, optional
        If True, returns tensor with shape (channels, frames).
        If False, returns tensor with shape (frames, channels).
        Default: True.
    dtype : torch.dtype, optional
        Data type for the output tensor. Default: torch.float32.
    always_2d : bool, optional
        If True, always return 2D tensor even for mono audio.
        If False, mono audio returns 1D tensor (frames,) when channels_first=False.
        Default: True.
    frame_offset : int, optional
        Number of frames to skip at the start of the file. Default: 0.
    num_frames : int, optional
        Number of frames to read. If -1, reads to the end of the file. Default: -1.
    
    Returns
    -------
    tensor : torch.Tensor
        Audio waveform as a tensor.
    sample_rate : int
        Sample rate of the audio file.
    
    Example
    -------
    >>> # This is an example, actual file needed to run
    >>> # audio, sr = load("example.wav")
    """
    try:
        # Read audio file - soundfile returns (frames, channels) or (frames,) for mono
        audio_np, sample_rate = sf.read(
            path, 
            start=frame_offset, 
            frames=num_frames, 
            dtype='float32', 
            always_2d=always_2d
        )
        
        # Convert to torch tensor
        audio = torch.from_numpy(audio_np)
        
        # Convert to requested dtype
        if dtype != torch.float32:
            audio = audio.to(dtype)
        
        # Handle shape conversion
        if audio.ndim == 1:
            # Mono audio as 1D - only add channel dimension if always_2d is True
            if always_2d:
                # Need to add channel dimension: (frames,) -> (1, frames) or (frames, 1)
                if channels_first:
                    audio = audio.unsqueeze(0)
                else:
                    audio = audio.unsqueeze(1)
        elif audio.ndim == 2:
            # Multi-channel or explicitly 2D mono
            if channels_first:
                # Convert from (frames, channels) to (channels, frames)
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
        - 2D tensor/array: (channels, frames) if channels_first=True, or (frames, channels) if channels_first=False
        - 3D tensor/array: (batch, channels, frames) if channels_first=True - uses first batch element
    sample_rate : int
        Sample rate for the audio file.
    channels_first : bool, optional
        If True, input is assumed to be (channels, frames) for 2D or (batch, channels, frames) for 3D.
        If False, input is assumed to be (frames, channels).
        Default: True.
    subtype : str, optional
        Audio encoding subtype (e.g., 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT').
        If None, soundfile will choose an appropriate subtype based on the file format.
        Default: None.
    
    Returns
    -------
    None
    
    Example
    -------
    >>> # This is an example, actual tensor needed to run
    >>> # waveform = torch.randn(1, 16000)
    >>> # save("output.wav", waveform, 16000)
    """
    try:
        # Convert to numpy if needed
        if isinstance(src, torch.Tensor):
            audio_np = src.detach().cpu().numpy()
        else:
            audio_np = np.asarray(src)
        
        # Handle different input shapes
        if audio_np.ndim == 3:
            # Batched input: (batch, channels, frames) if channels_first - take first batch
            audio_np = audio_np[0]
        
        if audio_np.ndim == 2:
            # Convert to (frames, channels) if channels_first is True
            if channels_first:
                # Convert (channels, frames) to (frames, channels)
                audio_np = audio_np.T
            # Otherwise, assume already in (frames, channels) format
        elif audio_np.ndim == 1:
            # Mono: (frames,) - soundfile handles this
            pass
        else:
            raise ValueError(
                f"Unsupported audio shape: {audio_np.shape}. "
                "Expected 1D (frames,), 2D (channels, frames) or (frames, channels), "
                "or 3D (batch, channels, frames)."
            )
        
        # Write audio file
        if subtype is not None:
            sf.write(path, audio_np, sample_rate, subtype=subtype)
        else:
            sf.write(path, audio_np, sample_rate)
        
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
    
    Example
    -------
    >>> # This is an example, actual file needed to run
    >>> # info_obj = info("example.wav")
    >>> # print(info_obj.sample_rate, info_obj.duration)
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
