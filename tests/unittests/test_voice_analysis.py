"""Test voice analysis functions for dysarthric speech.

Author:
 * Peter Plantinga
"""

import torch


def test_voice_analysis():

    from speechbrain.processing.voice_analysis import estimate_f0

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
    estimated_f0, voiced_frames = estimate_f0(
        tone.unsqueeze(0), sample_rate=sample_rate
    )

    assert torch.all(abs(estimated_f0[0, 10:-10] - frequency) < 5)
