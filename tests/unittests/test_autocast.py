"""Tests for autocast utilities.

Authors
 * Adel Moumen 2025
"""

import pytest
import torch

from speechbrain.utils.autocast import _infer_device_type, fwd_default_precision


def test_infer_device_type_cpu():
    """Test device type inference with CPU tensors."""
    x = torch.randn(10, 10)
    device_type = _infer_device_type(x)
    assert device_type == "cpu"


def test_infer_device_type_kwargs():
    """Test device type inference with tensors in kwargs."""
    x = torch.randn(10, 10)
    device_type = _infer_device_type(input=x)
    assert device_type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_infer_device_type_cuda():
    """Test device type inference with CUDA tensors."""
    x = torch.randn(10, 10).cuda()
    device_type = _infer_device_type(x)
    assert device_type == "cuda"


def test_fwd_default_precision_cpu():
    """Test fwd_default_precision decorator with CPU tensors."""

    class TestModule(torch.nn.Module):
        @fwd_default_precision(cast_inputs=torch.float32)
        def forward(self, x):
            return x * 2

    module = TestModule()
    x = torch.randn(10, 10, dtype=torch.float16)

    # Without autocast, should work normally
    y = module(x)
    assert y.dtype == torch.float16  # No casting outside autocast

    # With autocast on CPU with bfloat16
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = module(x)
        # Should be cast to float32 due to the decorator
        assert y.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fwd_default_precision_cuda():
    """Test fwd_default_precision decorator with CUDA tensors."""

    class TestModule(torch.nn.Module):
        @fwd_default_precision(cast_inputs=torch.float32)
        def forward(self, x):
            return x * 2

    module = TestModule().cuda()
    x = torch.randn(10, 10, dtype=torch.float16).cuda()

    # Without autocast, should work normally
    y = module(x)
    assert y.dtype == torch.float16  # No casting outside autocast

    # With autocast on CUDA with float16
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = module(x)
        # Should be cast to float32 due to the decorator
        assert y.dtype == torch.float32


def test_fwd_default_precision_force_allow_autocast():
    """Test force_allow_autocast parameter."""

    class TestModule(torch.nn.Module):
        @fwd_default_precision(cast_inputs=torch.float32)
        def forward(self, x):
            return x * 2

    module = TestModule()
    x = torch.randn(10, 10, dtype=torch.float16)

    # With autocast and force_allow_autocast=True, autocast behavior is preserved
    # and the decorator's cast_inputs is ignored
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = module(x, force_allow_autocast=True)
        # Just verify the function executes without error
        assert y is not None


def test_fwd_default_precision_partial():
    """Test fwd_default_precision as a partial decorator."""

    # Create a custom decorator with different cast_inputs
    custom_decorator = fwd_default_precision(cast_inputs=torch.float64)

    class TestModule(torch.nn.Module):
        @custom_decorator
        def forward(self, x):
            return x * 2

    module = TestModule()
    x = torch.randn(10, 10, dtype=torch.float16)

    # With autocast, should cast to float64
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = module(x)
        assert y.dtype == torch.float64
