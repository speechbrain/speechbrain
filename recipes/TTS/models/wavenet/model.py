"""
WaveNet - Building Blocks

Elements of the architecture are inspired by
https://github.com/r9y9/wavenet_vocoder
"""

from __future__ import with_statement, print_function, absolute_import

# verify input type
# 1. raw [-1, 1]
# 2. mulaw [-1, 1]
# 3. mulaw-quantize [0, mu]
# If input_type is raw or mulaw, network assumes scalar input and discretized mixture 
# of logistic distributions output, otherwise one-hot input and softmax output are assumed.
def _assert_valid_input_type(s):
    assert s == "mulaw-quantize" or s == "mulaw" or s == "raw"
def is_mulaw_quantize(s):
    _assert_valid_input_type(s)
    return s == "mulaw-quantize"
def is_mulaw(s):
    _assert_valid_input_type(s)
    return s == "mulaw"
def is_raw(s):
    _assert_valid_input_type(s)
    return s == "raw"
def is_scalar_input(s):
    return is_raw(s) or is_mulaw(s)