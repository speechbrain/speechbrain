#!/bin/bash
python -c 'from tests.utils.recipe_tests import run_recipe_tests; print("TEST FAILED!") if not(run_recipe_tests(run_opts="--device=cuda")) else print("TEST PASSED")'
