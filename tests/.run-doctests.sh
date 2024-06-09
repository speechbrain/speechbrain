#!/bin/bash
set -e -u -o pipefail

# To run doctests locally, the easiest approach is to do:
# > pytest --doctest-modules speechbrain/
# However, we take this more complex approach to avoid testing files not
# tracked by git. We filter out tests that require optional dependencies.
avoid="transducer_loss.py\|fairseq_wav2vec.py\|huggingface_wav2vec.py\|bleu.py\|ctc_segmentation.py\|check_url.py\|huggingface_whisper.py\|language_model.py\|vocos.py\|snac.py\|discrete_ssl.py\|lobes/flair|lobes/spacy"
git ls-files speechbrain | grep -e "\.py$" | grep -v $avoid | xargs pytest --doctest-modules
