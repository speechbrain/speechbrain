#!/bin/bash

git ls-files tests/unittests | grep -e "\.py$" | xargs pytest
