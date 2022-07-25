#!/bin/bash
pip install pesq
pip install pystoi
pip install librosa
pip install tensorboard
pip install transformers
python -c 'from speechbrain.utils.recipe_tests import load_yaml_test; print("TEST FAILED!") if not(load_yaml_test()) else print("TEST PASSED")'
