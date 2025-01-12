#!/bin/bash
set -e -u -o pipefail

# All external dependencies need to be installed before running tests on our third-party integrations.
# Here is the old code from integration testing for reference:
#   uv pip install --system ctc-segmentation  # ctc-segmentation is funky with uv due to their oldest-supported-numpy dependency
#   uv pip install --system -r requirements.txt torch==2.2.1+cpu torchaudio==2.2.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu k2==1.24.4.dev20240223+cpu.torch2.2.1 --find-links https://k2-fsa.github.io/k2/cpu.html kaldilm==1.15.1 spacy==3.7.4 flair==0.13.1 gensim==4.3.2

# To run third party unittests locally, the easiest approach is to do:
# > pytest speechbrain/integration/tests/
# However, we take this more complex approach to avoid files not tracked
git ls-files speechbrain/integrations/tests | grep -e "\.py$" | xargs pytest

# To run third party doctests locally, the easiest approach is to do:
# > pytest --doctest-modules speechbrain/integration/
# However, we take this more complex approach to avoid files not tracked
git ls-files speechbrain/integrations | grep -e "\.py$" | xargs pytest --doctest-modules

