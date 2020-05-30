import torch
import torch.nn


def test_RNN():

    from speechbrain.nnet.RNN import RNN, GRU, LSTM, LiGRU

    # Check GRU
    inputs = torch.randn(4, 2, 7)
    net = RNN(
        hidden_size=5, num_layers=2, return_hidden=True, bidirectional=False,
    )
    output, hn = net(inputs, init_params=True)
    output_l = []
    hn_t = None
    for t in range(inputs.shape[1]):
        out_t, hn_t = net(inputs[:, t, :].unsqueeze(1), hn_t)
        output_l.append(out_t.squeeze(1))

    out_steps = torch.stack(output_l, dim=1)
    assert torch.all(
        torch.lt(torch.add(out_steps, -output), 1e-3)
    ), "GRU output mismatch"
    assert torch.all(
        torch.lt(torch.add(hn_t, -hn), 1e-3)
    ), "GRU hidden states mismatch"

    # Check GRU
    inputs = torch.randn(4, 2, 7)
    net = GRU(
        hidden_size=5, num_layers=2, return_hidden=True, bidirectional=False,
    )
    output, hn = net(inputs, init_params=True)
    output_l = []
    hn_t = None
    for t in range(inputs.shape[1]):
        out_t, hn_t = net(inputs[:, t, :].unsqueeze(1), hn_t)
        output_l.append(out_t.squeeze(1))

    out_steps = torch.stack(output_l, dim=1)
    assert torch.all(
        torch.lt(torch.add(out_steps, -output), 1e-3)
    ), "GRU output mismatch"
    assert torch.all(
        torch.lt(torch.add(hn_t, -hn), 1e-3)
    ), "GRU hidden states mismatch"

    # Check LSTM
    inputs = torch.randn(4, 2, 7)
    net = LSTM(
        hidden_size=5, num_layers=2, return_hidden=True, bidirectional=False,
    )
    output, hn = net(inputs, init_params=True)
    output_l = []
    hn_t = None
    for t in range(inputs.shape[1]):
        out_t, hn_t = net(inputs[:, t, :].unsqueeze(1), hn_t)
        output_l.append(out_t.squeeze(1))

    out_steps = torch.stack(output_l, dim=1)
    assert torch.all(
        torch.lt(torch.add(out_steps, -output), 1e-3)
    ), "LSTM output mismatch"
    assert torch.all(torch.lt(torch.add(hn_t[0], -hn[0]), 1e-3)) and torch.all(
        torch.lt(torch.add(hn_t[1], -hn[1]), 1e-3)
    ), "LSTM hidden states mismatch"

    # Check LiGRU
    inputs = torch.randn(1, 2, 2)
    net = LiGRU(
        hidden_size=5,
        num_layers=2,
        return_hidden=True,
        bidirectional=False,
        normalization="layernorm",
    )

    output, hn = net(inputs, init_params=True)
    output_l = []
    hn_t = None
    for t in range(inputs.shape[1]):
        out_t, hn_t = net(inputs[:, t, :].unsqueeze(1), hn_t)
        output_l.append(out_t.squeeze(1))

    out_steps = torch.stack(output_l, dim=1)

    assert torch.all(
        torch.lt(torch.add(out_steps, -output), 1e-3)
    ), "LiGRU output mismatch"
    assert torch.all(torch.lt(torch.add(hn_t[0], -hn[0]), 1e-3)) and torch.all(
        torch.lt(torch.add(hn_t[1], -hn[1]), 1e-3)
    ), "LiGRU hidden states mismatch"


test_RNN()
