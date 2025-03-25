"""Test voice analysis functions for dysarthric speech.

Author:
 * Peter Plantinga
"""

import pytest
import torch

SAMPLE_RATE = 16000
TONE_HZ = 220


@pytest.fixture
def pure_tone():

    # Create 3-second pure tone wave with known f0
    duration = 3
    values = [
        TONE_HZ * 2 * torch.pi * x / SAMPLE_RATE
        for x in range(SAMPLE_RATE * duration)
    ]
    tone = torch.sin(torch.tensor(values)).unsqueeze(0)

    # Add light white noise
    tone += torch.randn(tone.shape) * 0.005

    return tone


def test_voice_analysis(pure_tone):
    from speechbrain.processing.vocal_features import (
        compute_autocorr_features,
        compute_periodic_features,
    )

    frames = pure_tone.unfold(-1, 800, 200)
    harmonicity, best_lags = compute_autocorr_features(frames, 60, 100)
    jitter, shimmer = compute_periodic_features(frames, best_lags)

    # Basic feature computation
    f0 = SAMPLE_RATE / best_lags

    # Frequency f0
    assert TONE_HZ - 1 < f0.mean() < TONE_HZ + 1
    # HNR
    assert 0.99 < harmonicity.mean() < 1.0
    # Jitter
    assert 0.0 < jitter.mean() < 0.02
    # Shimmer
    assert 0.0 < shimmer.mean() < 0.01


def test_spectral_features(pure_tone):
    from speechbrain.processing.features import STFT, spectral_magnitude
    from speechbrain.processing.vocal_features import compute_spectral_features

    spectrum = spectral_magnitude(STFT(SAMPLE_RATE)(pure_tone), power=0.5)
    features = compute_spectral_features(spectrum)

    # Centroid
    assert 0.05 < features[:, :, 0].mean() < 0.07
    # Spread
    assert 0.13 < features[:, :, 1].mean() < 0.15
    # Skew
    assert 4.5 < features[:, :, 2].mean() < 4.7
    # Kurtosis
    assert 24 < features[:, :, 3].mean() < 26
    # Entropy -- a pure tone has low entropy
    assert -4.1 < features[:, :, 4].mean() < -4.0
    # Flatness
    assert 0.09 < features[:, :, 5].mean() < 0.11
    # Crest
    assert 0.37 < features[:, :, 6].mean() < 0.39
    # Flux
    assert 0.18 < features[:, :, 7].mean() < 0.21


def test_gne(pure_tone):
    from speechbrain.processing.vocal_features import compute_gne

    gne = compute_gne(pure_tone, sample_rate=SAMPLE_RATE)

    assert 0.93 < gne.mean() < 0.95
