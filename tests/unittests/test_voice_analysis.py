"""Test voice analysis functions for dysarthric speech.

Author:
 * Peter Plantinga
"""

import torch


def test_voice_analysis():

    from speechbrain.processing.voice_analysis import (
        compute_gne,
        vocal_characteristics,
    )

    # Create 3-second pure tone wave with known f0
    sample_rate = 16000
    frequency = 220
    duration = 3
    values = [
        frequency * 2 * torch.pi * x / sample_rate
        for x in range(sample_rate * duration)
    ]
    tone = torch.sin(torch.tensor(values)).unsqueeze(0)

    # Add noise for better HNR calculation
    tone += torch.randn(tone.shape) * 0.005

    # Add batch dimension
    features = vocal_characteristics(tone, sample_rate=sample_rate)

    # Frequency f0
    assert frequency - 1 < features[:, :, 0].mean() < frequency + 1
    # HNR
    assert 28 < features[:, :, 1].mean() < 31
    # Jitter
    assert 20 < features[:, :, 2].mean() < 23
    # Shimmer
    assert 27 < features[:, :, 3].mean() < 29
    # Centroid
    assert 0.05 < features[:, :, 4].mean() < 0.07
    # Spread
    assert 0.13 < features[:, :, 5].mean() < 0.15
    # Skew
    assert 4.5 < features[:, :, 6].mean() < 4.7
    # Kurtosis
    assert 24 < features[:, :, 7].mean() < 26
    # Entropy -- a pure tone has low entropy
    assert -4.9 < features[:, :, 8].mean() < -4.6
    # Flatness
    assert 0.06 < features[:, :, 9].mean() < 0.08
    # Crest
    assert 0.45 < features[:, :, 10].mean() < 0.47
    # Flux
    assert 0.03 < features[:, :, 11].mean() < 0.04

    gne = compute_gne(tone, sample_rate=sample_rate)

    assert 11 < gne.mean() < 13
