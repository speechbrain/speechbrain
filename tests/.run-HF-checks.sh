#!/bin/bash
python -c 'from speechbrain.utils.check_HF_repo import run_HF_check; print("TEST FAILED!") if not(run_HF_check()) else print("TEST PASSED!")'
