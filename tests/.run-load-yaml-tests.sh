#!/bin/bash
pip install pesq
pip install pystoi
python -c 'from speechbrain.utils.recipe_tests import load_yaml_test; print("TEST FAILED!") if not(load_yaml_test()) else print("TEST PASSED")'
