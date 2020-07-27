import torch
import torch.nn
from collections import OrderedDict


def test_RNN():

    from speechbrain.nnet.RNN import (
        RNN,
        GRU,
        LSTM,
        LiGRU,
        QuasiRNN,
        RNNCell,
        GRUCell,
        LSTMCell,
    )

    # Check RNN
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

    # Check QuasiRNN
    inputs = torch.randn(1, 2, 2)
    net = QuasiRNN(
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
    ), "QuasiRNN output mismatch"
    assert torch.all(
        torch.lt(torch.add(hn_t[0], -hn[0][1]), 1e-3)
    ) and torch.all(
        torch.lt(torch.add(hn_t[1], -hn[1][1]), 1e-3)
    ), "QuasiRNN hidden states mismatch"

    # Check RNNCell
    inputs = torch.randn(4, 2, 7)
    net = RNNCell(hidden_size=5, num_layers=2, dropout=0.0,)
    hn_t = None
    output_lst = []
    for t in range(inputs.shape[1]):
        init_params = True if t == 0 else False
        output, hn_t = net(inputs[:, t], hn_t, init_params=init_params)
        output_lst.append(output)

    out_steps = torch.stack(output_lst, dim=1)
    rnn = torch.nn.RNN(
        input_size=7, hidden_size=5, num_layers=2, batch_first=True
    )

    # rename the state_dict
    state = net.state_dict()
    new_state = []
    for name, tensor in state.items():
        index, weight_id = name[len("rnn_cells.")], name[len("rnn_cells.0.") :]
        new_state.append((f"{weight_id}_l{index}", tensor))
    new_state = OrderedDict(new_state)
    rnn.load_state_dict(new_state)
    output, hn = rnn(inputs)

    assert torch.all(
        torch.lt(torch.add(out_steps, -output), 1e-3)
    ), "RNNCell output mismatch"
    assert torch.all(torch.lt(torch.add(hn_t[0], -hn[0]), 1e-3)) and torch.all(
        torch.lt(torch.add(hn_t[1], -hn[1]), 1e-3)
    ), "RNNCell hidden states mismatch"

    # Check GRUCell
    inputs = torch.randn(4, 2, 7)
    net = GRUCell(hidden_size=5, num_layers=2, dropout=0.0,)
    hn_t = None
    output_lst = []
    for t in range(inputs.shape[1]):
        init_params = True if t == 0 else False
        output, hn_t = net(inputs[:, t], hn_t, init_params=init_params)
        output_lst.append(output)

    out_steps = torch.stack(output_lst, dim=1)
    rnn = torch.nn.GRU(
        input_size=7, hidden_size=5, num_layers=2, batch_first=True
    )

    # rename the state_dict
    state = net.state_dict()
    new_state = []
    for name, tensor in state.items():
        index, weight_id = name[len("rnn_cells.")], name[len("rnn_cells.0.") :]
        new_state.append((f"{weight_id}_l{index}", tensor))
    new_state = OrderedDict(new_state)
    rnn.load_state_dict(new_state)
    output, hn = rnn(inputs)

    assert torch.all(
        torch.lt(torch.add(out_steps, -output), 1e-3)
    ), "GRUCell output mismatch"
    assert torch.all(torch.lt(torch.add(hn_t[0], -hn[0]), 1e-3)) and torch.all(
        torch.lt(torch.add(hn_t[1], -hn[1]), 1e-3)
    ), "GRUCell hidden states mismatch"

    # Check GRUCell
    inputs = torch.randn(4, 2, 7)
    net = LSTMCell(hidden_size=5, num_layers=2, dropout=0.0,)
    hn_t = None
    output_lst = []
    for t in range(inputs.shape[1]):
        init_params = True if t == 0 else False
        output, hn_t = net(inputs[:, t], hn_t, init_params=init_params)
        output_lst.append(output)

    out_steps = torch.stack(output_lst, dim=1)
    rnn = torch.nn.LSTM(
        input_size=7, hidden_size=5, num_layers=2, batch_first=True
    )

    # rename the state_dict
    state = net.state_dict()
    new_state = []
    for name, tensor in state.items():
        index, weight_id = name[len("rnn_cells.")], name[len("rnn_cells.0.") :]
        new_state.append((f"{weight_id}_l{index}", tensor))
    new_state = OrderedDict(new_state)
    rnn.load_state_dict(new_state)
    output, hn = rnn(inputs)

    assert torch.all(
        torch.lt(torch.add(out_steps, -output), 1e-3)
    ), "LSTMCell output mismatch"
    assert (
        torch.all(torch.lt(torch.add(hn_t[0][0], -hn[0][0]), 1e-3))
        and torch.all(torch.lt(torch.add(hn_t[0][1], -hn[0][1]), 1e-3))
        and torch.all(torch.lt(torch.add(hn_t[1][0], -hn[1][0]), 1e-3))
        and torch.all(torch.lt(torch.add(hn_t[1][1], -hn[1][1]), 1e-3))
    ), "LSTMCell hidden states mismatch"


test_RNN()
