#!/bin/bash
set -e -u -o pipefail

echo "===Ruff Format==="
git ls-files | grep -E "\.py$" | xargs ruff format --check
echo "===Ruff==="
git ls-files | grep -E "\.py$" | xargs ruff check --statistics
echo "===Yamllint==="
git ls-files | grep -E "\.yaml$|\.yml$" | xargs yamllint --no-warnings
