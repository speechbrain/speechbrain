import torch


def test_normalize():

    from speechbrain.processing.signal_processing import compute_amplitude
    from speechbrain.processing.signal_processing import rescale
    import random
    import numpy as np

    for scale in ["dB", "linear"]:
        for amp_type in ["peak", "avg"]:
            for test_vec in [
                torch.zeros((100)),
                torch.rand((10, 100)),
                torch.rand((10, 100, 5)),
            ]:

                lengths = (
                    test_vec.size(1)
                    if len(test_vec.shape) > 1
                    else test_vec.size(0)
                )
                amp = compute_amplitude(test_vec, lengths, amp_type, scale)
                scaled_back = rescale(
                    random.random() * test_vec, lengths, amp, amp_type, scale,
                )
                np.testing.assert_array_almost_equal(
                    scaled_back.numpy(), test_vec.numpy()
                )
