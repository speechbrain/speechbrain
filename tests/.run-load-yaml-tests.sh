#!/bin/bash
pip install pesq
pip install pystoi
<<<<<<< HEAD
python -c 'from speechbrain.utils.recipe_tests import load_yaml_test; print("TEST FAILED!") if not(load_yaml_test()) else print("TEST PASSED")'
=======
pip install librosa
pip install tensorboard
pip install transformers
# avoid list: these yamls cause segfaults (on a GPU node)
# pip install git+https://github.com/jfsantos/SRMRpy
python -c 'from tests.utils.recipe_tests import load_yaml_test; print("TEST FAILED!") if not(load_yaml_test(avoid_list=["recipes/Voicebank/dereverb/MetricGAN-U/hparams/train_dereverb.yaml", "recipes/Voicebank/dereverb/spectral_mask/hparams/train.yaml", "recipes/Voicebank/enhance/MetricGAN-U/hparams/train_dnsmos.yaml", "recipes/Voicebank/enhance/MetricGAN/hparams/train.yaml", "recipes/Voicebank/enhance/spectral_mask/hparams/train.yaml", "recipes/Voicebank/enhance/waveform_map/hparams/train.yaml"])) else print("TEST PASSED")'
>>>>>>> 891318f5950c337bb951912bf64bd5973af7c908
