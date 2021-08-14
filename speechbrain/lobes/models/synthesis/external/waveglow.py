"""Modules for the pretrained Waveglow model

Authors:
 * Artem Ploujnikov 2021
"""


def adapter(model, inputs, sigma=0.667):
    """An adapter that performs inference on the pretrained
    WaveGlow model"""

    return model.infer(inputs, sigma)
