"""Audio I/O compatibility layer using soundfile.

This module provides audio I/O functions to replace torchaudio's load, save, 
and info functions with soundfile-based implementations.

Authors
 * SpeechBrain Team 2024
"""

import soundfile
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# Maximum expected number of channels for automatic shape detection
MAX_EXPECTED_CHANNELS = 8


@dataclass
class AudioMetadata:
    """Metadata about an audio file.
    
    Attributes
    ----------
    sample_rate : int
        The sample rate of the audio file.
    frames : int
        The number of frames in the audio file.
    channels : int
        The number of channels in the audio file.
    subtype : str
        The audio subtype (e.g., 'PCM_16', 'PCM_24').
    format : str
        The audio format (e.g., 'WAV', 'FLAC').
    """
    sample_rate: int
    frames: int
    channels: int
    subtype: str
    format: str
    
    @property
    def duration(self) -> float:
        """Return the duration of the audio in seconds."""
        return self.frames / self.sample_rate if self.sample_rate > 0 else 0.0
    
    @property
    def num_frames(self) -> int:
        """Return the number of frames (alias for frames, for torchaudio compatibility)."""
        return self.frames


def load(path, *, channels_first=True, dtype=torch.float32, always_2d=False):
    """Load an audio file using soundfile.
    
    Parameters
    ----------
    path : str or Path
        Path to the audio file to load.
    channels_first : bool, optional
        If True, return tensor with shape (channels, frames).
        If False, return tensor with shape (frames, channels).
        Default: True.
    dtype : torch.dtype, optional
        The desired data type for the output tensor.
        Default: torch.float32.
    always_2d : bool, optional
        If True, always return a 2D tensor even for mono audio.
        For mono with channels_first=True, returns (1, frames).
        For mono with channels_first=False, returns (frames, 1).
        Default: False.
    
    Returns
    -------
    waveform : torch.Tensor
        Audio tensor with shape depending on channels_first parameter.
    sample_rate : int
        The sample rate of the audio file.
    
    Examples
    --------
    >>> import tempfile
    >>> import numpy as np
    >>> # Create a test file
    >>> with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    ...     test_file = f.name
    >>> test_data = np.random.randn(16000).astype(np.float32)
    >>> soundfile.write(test_file, test_data, 16000)
    >>> # Load the file
    >>> waveform, sr = load(test_file)
    >>> print(waveform.shape, sr)
    torch.Size([16000]) 16000
    >>> import os
    >>> os.unlink(test_file)
    """
    # Read audio file with soundfile (always returns float32, always_2d=True)
    data, sample_rate = soundfile.read(str(path), dtype='float32', always_2d=True)
    
    # Convert to torch tensor
    waveform = torch.from_numpy(data)
    
    # Convert to requested dtype if needed
    if dtype != torch.float32:
        waveform = waveform.to(dtype)
    
    # Handle channel ordering
    # soundfile returns (frames, channels), we need to transpose if channels_first
    if channels_first:
        waveform = waveform.t()  # (channels, frames)
    
    # Handle always_2d parameter
    if not always_2d:
        # If mono and user doesn't want always_2d, squeeze the channel dimension
        if waveform.shape[0] == 1 and channels_first:
            waveform = waveform.squeeze(0)
        elif waveform.shape[1] == 1 and not channels_first:
            waveform = waveform.squeeze(1)
    
    return waveform, sample_rate


def save(path, src, sample_rate, subtype="PCM_16"):
    """Save an audio file using soundfile.
    
    Parameters
    ----------
    path : str or Path
        Path where the audio file should be saved.
    src : torch.Tensor or numpy.ndarray
        Audio data to save. Can have shapes:
        - (frames,) - mono
        - (channels, frames) - multi-channel, channels first
        - (frames, channels) - multi-channel, channels last
        - (1, channels, frames) - batched with batch size 1
    sample_rate : int
        The sample rate for the audio file.
    subtype : str, optional
        The audio subtype (e.g., 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT').
        Default: 'PCM_16'.
    
    Examples
    --------
    >>> import tempfile
    >>> import torch
    >>> # Create test data
    >>> test_data = torch.randn(16000)
    >>> with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    ...     test_file = f.name
    >>> # Save the file
    >>> save(test_file, test_data, 16000)
    >>> # Verify it was saved
    >>> import os
    >>> os.path.exists(test_file)
    True
    >>> os.unlink(test_file)
    """
    # Convert to numpy if needed
    if isinstance(src, torch.Tensor):
        data = src.detach().cpu().numpy()
    else:
        data = np.array(src)
    
    # Handle different input shapes
    # Remove batch dimension if present
    if data.ndim == 3 and data.shape[0] == 1:
        data = data.squeeze(0)
    
    # Determine if we need to transpose
    # soundfile expects (frames, channels) or (frames,)
    if data.ndim == 2:
        # Check if it looks like (channels, frames) - typically channels < frames
        # Common heuristic: if first dim is small (<=MAX_EXPECTED_CHANNELS) and second is large, it's (channels, frames)
        if data.shape[0] <= MAX_EXPECTED_CHANNELS and data.shape[1] > data.shape[0]:
            # Likely (channels, frames), transpose to (frames, channels)
            data = data.T
    
    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect format from file extension and adjust subtype if needed
    path_str = str(path)
    ext = Path(path).suffix.lower()
    
    # For certain formats, we need to use appropriate subtypes or let soundfile choose
    if ext in ['.ogg', '.oga']:
        # OGG Vorbis doesn't use PCM
        subtype = None  # Let soundfile choose the default
    elif ext == '.mp3':
        # MP3 doesn't use PCM either
        subtype = None
    elif ext in ['.flac', '.wav']:
        # These support PCM subtypes
        pass
    else:
        # For unknown formats, let soundfile handle it
        subtype = None
    
    # Write the file
    soundfile.write(path_str, data, sample_rate, subtype=subtype)


def info(path):
    """Get metadata about an audio file using soundfile.
    
    Parameters
    ----------
    path : str or Path
        Path to the audio file.
    
    Returns
    -------
    metadata : AudioMetadata
        Metadata object containing information about the audio file.
        Has attributes: sample_rate, frames, channels, subtype, format
        and a duration property.
    
    Examples
    --------
    >>> import tempfile
    >>> import numpy as np
    >>> # Create a test file
    >>> with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    ...     test_file = f.name
    >>> test_data = np.random.randn(16000, 2).astype(np.float32)  # stereo
    >>> soundfile.write(test_file, test_data, 16000)
    >>> # Get file info
    >>> file_info = info(test_file)
    >>> print(file_info.sample_rate, file_info.channels)
    16000 2
    >>> import os
    >>> os.unlink(test_file)
    """
    # Get info from soundfile
    sf_info = soundfile.info(str(path))
    
    # Create and return AudioMetadata
    return AudioMetadata(
        sample_rate=sf_info.samplerate,
        frames=sf_info.frames,
        channels=sf_info.channels,
        subtype=sf_info.subtype,
        format=sf_info.format,
    )


def list_audio_backends():
    """List available audio backends.
    
    Returns
    -------
    list
        List of available audio backends. Currently only returns ['soundfile'].
    
    Examples
    --------
    >>> backends = list_audio_backends()
    >>> print(backends)
    ['soundfile']
    """
    return ["soundfile"]
