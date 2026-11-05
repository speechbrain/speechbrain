import pytest
import torch
import torch.nn as nn


def test_lora_linear_identity_at_init(device):
    from speechbrain.nnet.adapters import LoRA

    base = nn.Linear(16, 24).to(device)
    adapter = LoRA(base, rank=4)
    x = torch.rand(8, 10, 16, device=device)

    # LoRA property: the up projection starts at zero, so the adapted
    # layer must behave exactly like the pretrained layer at init.
    assert torch.allclose(adapter(x), base(x))


def test_lora_conv1d_identity_at_init(device):
    from speechbrain.nnet.adapters import LoRA

    base = nn.Conv1d(16, 32, kernel_size=3, stride=2).to(device)
    adapter = LoRA(base, rank=4)
    x = torch.rand(8, 16, 100, device=device)
    out = adapter(x)

    assert out.shape == base(x).shape
    assert torch.allclose(out, base(x))


def test_lora_conv_geometries(device):
    from speechbrain.nnet.adapters import LoRA

    for conv_kwargs in (
        {"kernel_size": 10, "stride": 5},
        {"kernel_size": 3, "padding": "same"},
        {"kernel_size": 5, "padding": 2, "dilation": 2},
    ):
        base = nn.Conv1d(4, 8, **conv_kwargs).to(device)
        adapter = LoRA(base, rank=2)
        x = torch.rand(2, 4, 50, device=device)
        assert adapter(x).shape == base(x).shape


def test_lora_conv2d_and_conv3d(device):
    from speechbrain.nnet.adapters import LoRA

    base_2d = nn.Conv2d(3, 8, kernel_size=3, stride=2).to(device)
    adapter_2d = LoRA(base_2d, rank=2)
    x_2d = torch.rand(2, 3, 32, 32, device=device)
    assert torch.allclose(adapter_2d(x_2d), base_2d(x_2d))

    base_3d = nn.Conv3d(2, 4, kernel_size=3).to(device)
    adapter_3d = LoRA(base_3d, rank=2)
    x_3d = torch.rand(2, 2, 8, 8, 8, device=device)
    assert torch.allclose(adapter_3d(x_3d), base_3d(x_3d))


def test_lora_conv_matches_manual_formula(device):
    from speechbrain.nnet.adapters import LoRA

    rank, alpha = 2, 8.0
    base = nn.Conv1d(3, 6, kernel_size=3).to(device)
    adapter = LoRA(base, rank=rank, alpha=alpha)
    nn.init.normal_(adapter.adapter_up_proj.weight)

    x = torch.rand(2, 3, 20, device=device)
    expected = base(x) + adapter.adapter_up_proj(
        adapter.adapter_down_proj(x)
    ) * (alpha / rank)

    assert torch.allclose(adapter(x), expected)


def test_lora_conv_freezes_pretrained_and_trains_adapters(device):
    from speechbrain.nnet.adapters import LoRA

    base = nn.Conv1d(4, 8, kernel_size=3).to(device)
    adapter = LoRA(base, rank=2)
    nn.init.normal_(adapter.adapter_up_proj.weight)

    x = torch.rand(2, 4, 30, device=device)
    adapter(x).sum().backward()

    assert adapter.pretrained_module.weight.grad is None
    assert adapter.adapter_down_proj.weight.grad is not None
    assert adapter.adapter_up_proj.weight.grad is not None


def test_lora_conv_learns(device):
    from speechbrain.nnet.adapters import LoRA

    base = nn.Conv1d(4, 8, kernel_size=3).to(device)
    adapter = LoRA(base, rank=2)
    x = torch.rand(2, 4, 30, device=device)
    out_before = adapter(x).detach().clone()

    optimizer = torch.optim.SGD(
        [p for p in adapter.parameters() if p.requires_grad], lr=1.0
    )
    adapter(x).sum().backward()
    optimizer.step()

    assert not torch.allclose(adapter(x), out_before)
    # The pretrained weights must not have moved
    assert torch.allclose(adapter.pretrained_module(x), base(x))


def test_lora_conv_groups_not_supported(device):
    from speechbrain.nnet.adapters import LoRA

    base = nn.Conv1d(4, 8, kernel_size=3, groups=2).to(device)
    with pytest.raises(ValueError, match="groups"):
        LoRA(base, rank=2)


def test_adapted_model_all_conv(device):
    # Reproduces the exact scenario reported in issue #3056
    from speechbrain.nnet.adapters import AdaptedModel, LoRA

    class SimpleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=10, stride=5)
            self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=2)

        def forward(self, x):
            return self.conv2(self.conv1(x))

    model = SimpleConv()
    reference = SimpleConv()
    reference.load_state_dict(model.state_dict())
    reference = reference.to(device)

    adapted = AdaptedModel(
        model_to_adapt=model,
        adapter_class=LoRA,
        all_conv=True,
        adapter_kwargs={"rank": 4},
    ).to(device)

    x = torch.rand(2, 1, 1600, device=device)
    out = adapted(x)

    assert out.shape == reference(x).shape
    assert torch.allclose(out, reference(x))
    out.sum().backward()

    trainable = [
        name for name, p in adapted.named_parameters() if p.requires_grad
    ]
    assert trainable, "The adapters should be trainable"
    assert all("adapter" in name for name in trainable)
