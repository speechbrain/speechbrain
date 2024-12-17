#!/bin/bash
set -e -u -o pipefail

# To run third party doctests locally, the easiest approach is to do:
# > pytest --doctest-modules speechbrain/integration/
# However, we take this more complex approach to avoid files not tracked
git ls-files speechbrain/integration | grep -e "\.py$" | xargs pytest --doctest-modules
