#!/bin/bash
pip install requests
python -c 'from tests.utils.check_url import check_links; print("TEST FAILED!") if not(check_links()) else print("TEST PASSED!")'
