"""Test voice analysis functions for dysarthric speech.

Author:
 * Peter Plantinga
"""

import torch


def test_voice_analysis():

    from speechbrain.processing.voice_analysis import vocal_characteristics

    # Create 3-second pure tone wave with known f0
    sample_rate = 44100
    frequency = 220
    duration = 3
    values = [
        frequency * 2 * torch.pi * x / sample_rate
        for x in range(sample_rate * duration)
    ]
    tone = torch.sin(torch.tensor(values))

    # Add batch dimension
    estimated_f0, voiced_frames, jitter, shimmer = vocal_characteristics(
        tone, sample_rate=sample_rate
    )

    assert all(abs(estimated_f0[5:-5] - frequency) < 5)

    assert all(jitter[5:-5] < 0.02)
    assert all(shimmer[5:-5] < 0.001)
