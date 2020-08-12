#!/bin/bash
set -e -u -o pipefail

echo "===Black==="
git ls-files | grep -E "\.py$" | grep -v "^tools/kaldi_decoder/" | xargs black --check --diff
echo "===Flake8==="
git ls-files | grep -E "\.py$" | grep -v "^tools/kaldi_decoder/" | xargs flake8 --count --statistics
echo "===Yamllint==="
git ls-files | grep -E "\.yaml$|\.yml$" | xargs yamllint --no-warnings
