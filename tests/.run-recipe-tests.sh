#!/bin/bash
python -c 'from tests.utils.recipe_tests import run_recipe_tests; print("TEST FAILED!") if not(run_recipe_tests(run_opts="--device=cuda", do_checks=True)) else print("TEST PASSED")'
# In the future, we need to run the recipe tests on multiple-gpus as well.
