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
    frequency = 150
    duration = 3
    values = [
        frequency * 2 * torch.pi * x / sample_rate
        for x in range(sample_rate * duration)
    ]
    tone = torch.sin(torch.tensor(values)).unsqueeze(0)

    # Add noise for better HNR calculation
    tone += torch.randn(tone.shape) * 0.005

    # Add batch dimension
    estimated_f0, voiced_frames, jitter, shimmer, hnr = vocal_characteristics(
        tone, sample_rate=sample_rate
    )

    # Remove last few values, they are affected by padding
    assert all(abs(estimated_f0[0, :-3] - frequency) < 2)
    assert jitter.mean() < 0.01
    assert shimmer.mean() < 0.01
    assert hnr.mean() > 20

    gne = compute_gne(tone, sample_rate=sample_rate)

    assert gne.mean() > 10
