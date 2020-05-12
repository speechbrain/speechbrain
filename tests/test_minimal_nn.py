import os
import sys


def test_autoencoder():
    path = "recipes/minimal_examples/neural_networks/autoencoder/"
    sys.path.append(os.path.abspath(path))
    import experiment  # noqa F401


def test_speaker_id():
    path = "recipes/minimal_examples/neural_networks/speaker_identification/"
    sys.path.append(os.path.abspath(path))
    import experiment  # noqa F401


def test_ASR_CTC():
    path = "recipes/minimal_examples/neural_networks/ASR_CTC/"
    sys.path.append(os.path.abspath(path))
    import experiment  # noqa F401


def test_ASR_DNN_HMM():
    path = "recipes/minimal_examples/neural_networks/ASR_DNN_HMM/"
    sys.path.append(os.path.abspath(path))
    import experiment  # noqa F401
