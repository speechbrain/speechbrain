#!/bin/bash
set -e -u -o pipefail

git ls-files tests/unittests | grep -e "\.py$" | xargs pytest
