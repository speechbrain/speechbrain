import torch
import torch.nn


def test_QLinear(device):
    from speechbrain.nnet.quaternion_networks.q_linear import QLinear

    inputs = torch.rand((1, 2, 4), device=device)

    # Test bias is correctly zeroed
    qlin_no_bias = QLinear(
        n_neurons=1, input_shape=inputs.shape, bias=False
    ).to(device)
    assert (qlin_no_bias.bias == 0).all().item()

    qlin_with_bias = QLinear(
        n_neurons=4, input_shape=inputs.shape, bias=True
    ).to(device)
    assert (qlin_with_bias.bias == 0).all().item()

    # Test output shape is correct
    outputs = qlin_with_bias(inputs)
    assert outputs.shape[-1] == 16

    with torch.no_grad():
        # Initialize weights equivalent to identity matrix:
        # | r -i -j  -k |
        # | i  r -k   j |
        # | j  k  r  -i |
        # | k -j  i   r |

        qlin_no_bias.r_weight.fill_(1)
        qlin_no_bias.i_weight.fill_(0)
        qlin_no_bias.j_weight.fill_(0)
        qlin_no_bias.k_weight.fill_(0)
    outputs = qlin_no_bias(inputs)
    assert torch.all(torch.eq(inputs, outputs))

    assert torch.jit.trace(qlin_with_bias, inputs)


def test_QPooling2d(device):
    from speechbrain.nnet.quaternion_networks.q_pooling import QPooling2d

    input = (
        torch.tensor(
            [
                [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]],
                [[0, 0, 0, 4], [3, 3, 0, 0], [0, 5, 0, 0]],
            ],
            device=device,
        )
        .float()
        .unsqueeze(0)
    )
    pool = QPooling2d("max", (2, 3)).to(device)
    output = pool(input)

    # Max pool by magnitude
    assert torch.all(torch.eq(output, input[:, 1, 2]))  # [0, 5, 0, 0]
    assert output.shape == (1, 1, 1, 4)

    pool = QPooling2d("max", (1, 3)).to(device)
    output = pool(input)
    assert torch.all(
        torch.eq(output, input[:, :, [2]])
    )  # [0,0,3,0] and [0,5,0,0]
    assert output.shape == (1, 2, 1, 4)

    pool = QPooling2d("avg", (2, 3)).to(device)
    output = pool(input)

    # Component-wise average:
    assert torch.all(torch.eq(output, input.mean((1, 2))))

    pool = QPooling2d("avg", (1, 3)).to(device)
    output = pool(input)

    assert torch.all(torch.eq(output, input.mean(2).unsqueeze(2)))

    assert torch.jit.trace(pool, input)


def test_QBatchNorm(device):
    from speechbrain.nnet.quaternion_networks.q_normalization import QBatchNorm

    input = torch.randn(100, 4 * 4, device=device) + 2.0
    norm = QBatchNorm(input_size=input.shape[-1]).to(device)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0)
    assert torch.all(torch.abs(current_mean) < 1e-06)