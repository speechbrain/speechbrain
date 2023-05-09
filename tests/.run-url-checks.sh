#!/bin/bash
pip install requests
<<<<<<< HEAD
python -c 'from speechbrain.utils.check_url import test_links; print("TEST FAILED!") if not(test_links()) else print("TEST PASSED!")'
=======
python -c 'from tests.utils.check_url import check_links; print("TEST FAILED!") if not(check_links()) else print("TEST PASSED!")'
>>>>>>> 891318f5950c337bb951912bf64bd5973af7c908
