#!/bin/bash
pip install requests
python -c 'from speechbrain.utils.check_url import test_links; print("TEST FAILED!") if not(test_links()) else print("TEST PASSED!")'
