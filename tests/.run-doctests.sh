#!/bin/bash
set -e -u -o pipefail

# It is ugly but currently we have to specifically filter out transducer loss here.
# Otherwise pytest will stubbornly fail on import if you don't have numba.
# (In contrast, implicitly discovered files which produce ImportErrors are ignored.)
git ls-files speechbrain | grep -e "\.py$" | grep -v "transducer_loss.py" | xargs pytest --doctest-modules
