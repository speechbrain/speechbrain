#!/bin/bash
set -e -u -o pipefail

pytest --doctest-modules speechbrain/
