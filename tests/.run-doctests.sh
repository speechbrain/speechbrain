#!/bin/bash
set -e -u -o pipefail

# We filter out tests that require not mandatory dependencies.
avoid="transducer_loss.py\|fairseq_wav2vec.py\|huggingface_wav2vec.py"
git ls-files speechbrain | grep -e "\.py$" | grep -v $avoid | xargs pytest --doctest-modules
