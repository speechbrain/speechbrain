import torch
import torch.nn


def test_QLinear(device):
    from speechbrain.nnet.quaternion_networks.q_linear import QLinear

    inputs = torch.rand((1, 2, 4), device=device)

    # Test bias is correctly zeroed
    qlin_no_bias = QLinear(n_neurons=1, input_shape=inputs.shape, bias=False)
    assert (qlin_no_bias.bias == 0).all().item()

    qlin_with_bias = QLinear(n_neurons=4, input_shape=inputs.shape, bias=True)
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
