"""Tests for audio I/O module.

Authors
 * SpeechBrain Team 2024
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

from speechbrain.dataio.audio_io import load, save, info, list_audio_backends, AudioMetadata


def _create_test_audio(sample_rate=16000, duration=1.0):
    """Helper to create clipped test audio to avoid PCM_16 overflow."""
    samples = torch.randn(int(sample_rate * duration)) * 0.5
    return torch.clamp(samples, -1.0, 1.0)


def test_list_audio_backends():
    """Test that list_audio_backends returns expected backends."""
    backends = list_audio_backends()
    assert isinstance(backends, list)
    assert "soundfile" in backends


def test_save_and_load_mono_wav():
    """Test saving and loading a mono WAV file."""
    # Create a simple sine wave
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    
    try:
        # Save the audio
        save(temp_file, samples, sample_rate)
        
        # Load it back
        loaded, loaded_sr = load(temp_file)
        
        # Check sample rate
        assert loaded_sr == sample_rate
        
        # Check shape (mono should be 1D by default)
        assert loaded.dim() == 1
        assert loaded.shape[0] == len(samples)
        
        # Check values (with some tolerance for encoding/decoding)
        np.testing.assert_allclose(
            loaded.numpy(), samples, rtol=1e-3, atol=1e-3
        )
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_save_and_load_stereo_wav():
    """Test saving and loading a stereo WAV file."""
    sample_rate = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create stereo signal (left: 440Hz, right: 880Hz)
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    samples = np.stack([left, right], axis=0)  # (2, frames)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    
    try:
        # Save as channels-first
        save(temp_file, samples, sample_rate)
        
        # Load back (channels-first by default)
        loaded, loaded_sr = load(temp_file, channels_first=True)
        
        assert loaded_sr == sample_rate
        assert loaded.shape == samples.shape  # (2, frames)
        
        # Check both channels
        np.testing.assert_allclose(
            loaded.numpy(), samples, rtol=1e-3, atol=1e-3
        )
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_load_channels_first_vs_last():
    """Test loading audio with different channel ordering."""
    sample_rate = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration))
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    samples = np.stack([left, right], axis=0)  # (2, frames)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    
    try:
        save(temp_file, samples, sample_rate)
        
        # Load channels-first
        cf_loaded, _ = load(temp_file, channels_first=True)
        assert cf_loaded.shape[0] == 2  # channels dimension first
        
        # Load channels-last
        cl_loaded, _ = load(temp_file, channels_first=False)
        assert cl_loaded.shape[1] == 2  # channels dimension last
        
        # They should be transposes of each other
        assert torch.allclose(cf_loaded, cl_loaded.t(), rtol=1e-5)
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_load_always_2d():
    """Test the always_2d parameter for mono files."""
    sample_rate = 16000
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate)).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    
    try:
        save(temp_file, samples, sample_rate)
        
        # Load without always_2d (default)
        loaded_1d, _ = load(temp_file, always_2d=False)
        assert loaded_1d.dim() == 1
        
        # Load with always_2d
        loaded_2d, _ = load(temp_file, always_2d=True)
        assert loaded_2d.dim() == 2
        assert loaded_2d.shape[0] == 1  # (1, frames) with channels_first=True
        
        # Values should match
        assert torch.allclose(loaded_1d, loaded_2d.squeeze(0))
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_save_with_torch_tensor():
    """Test saving with PyTorch tensors."""
    sample_rate = 16000
    samples = _create_test_audio(sample_rate, 1.0)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    
    try:
        save(temp_file, samples, sample_rate)
        
        loaded, loaded_sr = load(temp_file)
        assert loaded_sr == sample_rate
        assert loaded.shape == samples.shape
        
        # Should be close (some precision loss from encoding)
        assert torch.allclose(loaded, samples, rtol=1e-2, atol=1e-2)
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_info_wav():
    """Test getting info from a WAV file."""
    sample_rate = 16000
    duration = 1.0
    samples = np.random.randn(int(sample_rate * duration), 2).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    
    try:
        save(temp_file, samples, sample_rate)
        
        file_info = info(temp_file)
        
        assert file_info.sample_rate == sample_rate
        assert file_info.channels == 2
        assert file_info.frames == samples.shape[0]
        assert file_info.format == 'WAV'
        
        # Check duration property
        expected_duration = samples.shape[0] / sample_rate
        assert abs(file_info.duration - expected_duration) < 0.01
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_save_and_load_flac():
    """Test saving and loading a FLAC file."""
    sample_rate = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as f:
        temp_file = f.name
    
    try:
        # Save the audio as FLAC
        save(temp_file, samples, sample_rate, subtype='PCM_16')
        
        # Load it back
        loaded, loaded_sr = load(temp_file)
        
        assert loaded_sr == sample_rate
        assert loaded.shape[0] == len(samples)
        
        # FLAC is lossless, should be very close
        np.testing.assert_allclose(
            loaded.numpy(), samples, rtol=1e-3, atol=1e-3
        )
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_info_flac():
    """Test getting info from a FLAC file."""
    sample_rate = 22050
    duration = 0.5
    samples = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as f:
        temp_file = f.name
    
    try:
        save(temp_file, samples, sample_rate)
        
        file_info = info(temp_file)
        
        assert file_info.sample_rate == sample_rate
        assert file_info.channels == 1
        assert file_info.frames == len(samples)
        assert file_info.format == 'FLAC'
        
        # Check duration
        expected_duration = len(samples) / sample_rate
        assert abs(file_info.duration - expected_duration) < 0.01
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_save_creates_parent_directory():
    """Test that save creates parent directories if they don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a nested path that doesn't exist
        temp_file = Path(tmpdir) / "subdir" / "test.wav"
        
        samples = np.random.randn(16000).astype(np.float32)
        save(temp_file, samples, 16000)
        
        assert temp_file.exists()


def test_save_batch_dimension():
    """Test saving with batch dimension (batch_size=1)."""
    sample_rate = 16000
    samples = torch.randn(1, 2, sample_rate)  # (batch=1, channels=2, frames)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    
    try:
        # Should handle batch dimension gracefully
        save(temp_file, samples, sample_rate)
        
        loaded, _ = load(temp_file, channels_first=True)
        
        # Should match the data without the batch dimension
        assert loaded.shape == samples.squeeze(0).shape
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
