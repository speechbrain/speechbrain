#!/bin/bash
set -e -u -o pipefail

avoid="test_ctc_segmentation.py"
git ls-files tests/unittests | grep -e "\.py$" | grep -v $avoid | xargs pytest
