#!/bin/bash
set -e -u -o pipefail

echo "===Black==="
git ls-files | grep -E "\.py$" | xargs black --check --diff
echo "===Ruff==="
git ls-files | grep -E "\.py$" | xargs ruff check --statistics
echo "===Yamllint==="
git ls-files | grep -E "\.yaml$|\.yml$" | xargs yamllint --no-warnings
