"""Tests for audio_io module.

Authors
 * SpeechBrain Contributors 2025
"""

import os

import numpy as np
import pytest
import torch


def test_audio_io_roundtrip_wav(tmpdir):
    """Test save and load roundtrip for WAV format."""
    from speechbrain.dataio import audio_io

    # Create a simple sine wave test signal
    sample_rate = 16000
    duration = 1.0  # seconds
    frequency = 440.0  # Hz (A4 note)

    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * frequency * t)
    waveform = waveform.unsqueeze(0)  # Add channel dimension: (1, frames)

    # Save audio
    audio_path = os.path.join(tmpdir, "test.wav")
    audio_io.save(audio_path, waveform, sample_rate)

    # Load audio back
    loaded_waveform, loaded_sr = audio_io.load(audio_path, channels_first=True)

    # Check sample rate
    assert loaded_sr == sample_rate, (
        f"Expected sample rate {sample_rate}, got {loaded_sr}"
    )

    # Check shape
    assert loaded_waveform.shape[0] == 1, (
        f"Expected 1 channel, got {loaded_waveform.shape[0]}"
    )

    # Check values are close (allow for encoding/decoding differences)
    assert torch.allclose(loaded_waveform, waveform, atol=1e-3), (
        "Waveforms don't match"
    )


def test_audio_io_roundtrip_flac(tmpdir):
    """Test save and load roundtrip for FLAC format."""
    from speechbrain.dataio import audio_io

    # Create a test signal
    sample_rate = 22050
    waveform = torch.rand(1, 22050)  # 1 second of random noise

    # Save as FLAC
    audio_path = os.path.join(tmpdir, "test.flac")
    audio_io.save(audio_path, waveform, sample_rate, subtype="PCM_16")

    # Load back
    loaded_waveform, loaded_sr = audio_io.load(audio_path)

    # Check sample rate
    assert loaded_sr == sample_rate

    # Check shape
    assert loaded_waveform.shape == waveform.shape

    # Check values are reasonably close
    assert torch.allclose(loaded_waveform, waveform, atol=5e-3)


def test_audio_io_info(tmpdir):
    """Test info function returns expected metadata."""
    from speechbrain.dataio import audio_io

    # Create test audio
    sample_rate = 16000
    duration = 2.5  # seconds
    num_frames = int(sample_rate * duration)
    waveform = torch.rand(1, num_frames)

    # Save audio
    audio_path = os.path.join(tmpdir, "test_info.wav")
    audio_io.save(audio_path, waveform, sample_rate)

    # Get info
    info = audio_io.info(audio_path)

    # Check metadata
    assert info.sample_rate == sample_rate, (
        f"Expected sample rate {sample_rate}, got {info.sample_rate}"
    )
    assert info.frames == num_frames, (
        f"Expected {num_frames} frames, got {info.frames}"
    )
    assert info.num_frames == num_frames, "num_frames alias doesn't match"
    assert info.channels == 1, f"Expected 1 channel, got {info.channels}"
    assert info.num_channels == 1, "num_channels alias doesn't match"
    assert abs(info.duration - duration) < 0.01, (
        f"Expected duration ~{duration}s, got {info.duration}s"
    )
    assert info.format == "WAV", f"Expected format WAV, got {info.format}"


def test_audio_io_load_channels_first(tmpdir):
    """Test load with channels_first=True."""
    from speechbrain.dataio import audio_io

    # Create stereo audio
    sample_rate = 16000
    waveform = torch.rand(2, 8000)  # (channels, frames)

    audio_path = os.path.join(tmpdir, "stereo.wav")
    audio_io.save(audio_path, waveform, sample_rate)

    # Load with channels_first=True (default)
    loaded, sr = audio_io.load(audio_path, channels_first=True)
    assert loaded.shape == (2, 8000), f"Expected (2, 8000), got {loaded.shape}"

    # Load with channels_first=False
    loaded_cf, sr = audio_io.load(audio_path, channels_first=False)
    assert loaded_cf.shape == (8000, 2), (
        f"Expected (8000, 2), got {loaded_cf.shape}"
    )


def test_audio_io_load_always_2d(tmpdir):
    """Test load with always_2d parameter."""
    from speechbrain.dataio import audio_io

    # Create mono audio
    sample_rate = 16000
    waveform = torch.rand(16000)  # 1D mono

    audio_path = os.path.join(tmpdir, "mono.wav")
    audio_io.save(audio_path, waveform, sample_rate)

    # Load with always_2d=True, channels_first=True
    loaded, sr = audio_io.load(audio_path, channels_first=True, always_2d=True)
    assert loaded.shape == (1, 16000), (
        f"Expected (1, 16000), got {loaded.shape}"
    )

    # Load with always_2d=True, channels_first=False
    loaded_cf, sr = audio_io.load(
        audio_path, channels_first=False, always_2d=True
    )
    assert loaded_cf.shape == (16000, 1), (
        f"Expected (16000, 1), got {loaded_cf.shape}"
    )


def test_audio_io_save_shapes(tmpdir):
    """Test save handles various input shapes correctly."""
    from speechbrain.dataio import audio_io

    sample_rate = 16000

    # Test 1D input (mono)
    waveform_1d = torch.rand(8000)
    path_1d = os.path.join(tmpdir, "mono_1d.wav")
    audio_io.save(path_1d, waveform_1d, sample_rate)
    loaded_1d, _ = audio_io.load(path_1d, channels_first=True, always_2d=True)
    assert loaded_1d.shape == (1, 8000)

    # Test 2D input channels-first (channels, frames)
    waveform_2d = torch.rand(1, 8000)
    path_2d = os.path.join(tmpdir, "mono_2d.wav")
    audio_io.save(path_2d, waveform_2d, sample_rate)
    loaded_2d, _ = audio_io.load(path_2d, channels_first=True, always_2d=True)
    assert loaded_2d.shape == (1, 8000)


def test_audio_io_save_stereo(tmpdir):
    """Test save and load stereo audio."""
    from speechbrain.dataio import audio_io

    sample_rate = 16000
    waveform = torch.rand(2, 8000)  # Stereo (2 channels)

    audio_path = os.path.join(tmpdir, "stereo.wav")
    audio_io.save(audio_path, waveform, sample_rate)

    loaded, sr = audio_io.load(audio_path, channels_first=True)
    assert loaded.shape == (2, 8000), (
        f"Expected stereo (2, 8000), got {loaded.shape}"
    )
    assert torch.allclose(loaded, waveform, atol=1e-3)


def test_audio_io_dtype(tmpdir):
    """Test load with different dtype."""
    from speechbrain.dataio import audio_io

    sample_rate = 16000
    waveform = torch.rand(1, 8000)

    audio_path = os.path.join(tmpdir, "test_dtype.wav")
    audio_io.save(audio_path, waveform, sample_rate)

    # Load with float64
    loaded_f64, _ = audio_io.load(audio_path, dtype=torch.float64)
    assert loaded_f64.dtype == torch.float64

    # Load with float32 (default)
    loaded_f32, _ = audio_io.load(audio_path, dtype=torch.float32)
    assert loaded_f32.dtype == torch.float32


def test_audio_io_numpy_input(tmpdir):
    """Test save with numpy array input."""
    from speechbrain.dataio import audio_io

    sample_rate = 16000
    waveform_np = np.random.rand(1, 8000).astype(np.float32)

    audio_path = os.path.join(tmpdir, "numpy_input.wav")
    audio_io.save(audio_path, waveform_np, sample_rate)

    loaded, sr = audio_io.load(audio_path, channels_first=True)
    assert loaded.shape == (1, 8000)
    assert torch.allclose(loaded, torch.from_numpy(waveform_np), atol=1e-3)


def test_audio_io_list_backends():
    """Test list_audio_backends function."""
    from speechbrain.dataio import audio_io

    backends = audio_io.list_audio_backends()
    assert isinstance(backends, list)
    assert "soundfile" in backends


def test_audio_io_error_handling(tmpdir):
    """Test error handling for invalid inputs."""
    from speechbrain.dataio import audio_io

    # Test loading non-existent file
    with pytest.raises(RuntimeError, match="Failed to load"):
        audio_io.load(os.path.join(tmpdir, "nonexistent.wav"))

    # Test info on non-existent file
    with pytest.raises(RuntimeError, match="Failed to get info"):
        audio_io.info(os.path.join(tmpdir, "nonexistent.wav"))


def test_audio_io_different_subtypes(tmpdir):
    """Test saving with different audio subtypes."""
    from speechbrain.dataio import audio_io

    sample_rate = 16000
    waveform = torch.rand(1, 8000)

    # Test PCM_16 (default)
    path_16 = os.path.join(tmpdir, "pcm16.wav")
    audio_io.save(path_16, waveform, sample_rate, subtype="PCM_16")
    info_16 = audio_io.info(path_16)
    assert info_16.subtype == "PCM_16"

    # Test PCM_24
    path_24 = os.path.join(tmpdir, "pcm24.wav")
    audio_io.save(path_24, waveform, sample_rate, subtype="PCM_24")
    info_24 = audio_io.info(path_24)
    assert info_24.subtype == "PCM_24"
