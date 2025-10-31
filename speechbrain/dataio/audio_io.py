"""
Lightweight soundfile-based audio I/O compatibility layer.

This module provides a minimal compatibility wrapper for audio I/O operations
using soundfile (pysoundfile) library, replacing torchaudio's load, save, and
info functions.

Authors
 * SpeechBrain Contributors 2025
"""

import numpy as np
import soundfile as sf
import torch


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
    duration : float
        Duration in seconds.
    """
    
    def __init__(self, sample_rate, frames, channels, subtype, format):
        self.sample_rate = sample_rate
        self.frames = frames
        self.num_frames = frames  # Alias for compatibility
        self.channels = channels
        self.num_channels = channels  # Alias for compatibility
        self.subtype = subtype
        self.format = format
    
    @property
    def duration(self):
        """Calculate duration in seconds."""
        return self.frames / self.sample_rate if self.sample_rate > 0 else 0.0
    
    def __repr__(self):
        return (
            f"AudioInfo(sample_rate={self.sample_rate}, frames={self.frames}, "
            f"channels={self.channels}, subtype='{self.subtype}', "
            f"format='{self.format}', duration={self.duration:.2f}s)"
        )


def load(path, *, channels_first=True, dtype=torch.float32, always_2d=False):
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
        Default: False.
    
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
        audio_np, sample_rate = sf.read(path, dtype='float32', always_2d=always_2d)
        
        # Convert to torch tensor
        audio = torch.from_numpy(audio_np)
        
        # Convert to requested dtype
        if dtype != torch.float32:
            audio = audio.to(dtype)
        
        # Handle shape conversion
        if audio.ndim == 1:
            # Mono audio as 1D
            if channels_first:
                # Need to add channel dimension: (frames,) -> (1, frames)
                audio = audio.unsqueeze(0)
        elif audio.ndim == 2:
            # Multi-channel or explicitly 2D mono
            if channels_first:
                # Convert from (frames, channels) to (channels, frames)
                audio = audio.transpose(0, 1)
        
        return audio, int(sample_rate)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {path}: {e}") from e


def save(path, src, sample_rate, subtype="PCM_16"):
    """Save audio to file using soundfile.
    
    Arguments
    ---------
    path : str
        Path where to save the audio file.
    src : torch.Tensor or numpy.ndarray
        Audio waveform. Can be:
        - 1D tensor/array: (frames,) - mono
        - 2D tensor/array: (channels, frames) or (frames, channels)
        - 3D tensor/array: (batch, channels, frames) - uses first batch element
    sample_rate : int
        Sample rate for the audio file.
    subtype : str, optional
        Audio encoding subtype (e.g., 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT').
        Default: 'PCM_16'.
    
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
            # Batched input: (batch, channels, frames) - take first batch
            audio_np = audio_np[0]
        
        if audio_np.ndim == 2:
            # Determine if it's (channels, frames) or (frames, channels)
            # Heuristic: if first dimension is smaller and <= 16, likely channels-first
            if audio_np.shape[0] <= 16 and audio_np.shape[0] < audio_np.shape[1]:
                # Convert (channels, frames) to (frames, channels)
                audio_np = audio_np.T
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
