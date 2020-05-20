import torch
import torch.nn


def test_RNN():

    from speechbrain.nnet.RNN import RNN

    # Check GRU
    inputs = torch.randn(4, 2, 7)
    net = RNN(
        rnn_type="gru",
        n_neurons=5,
        num_layers=2,
        return_hidden=True,
        bidirectional=False,
    )
    output, hn = net(inputs, init_params=True)
    output_l = []
    hn_t = None
    for t in range(inputs.shape[1]):
        out_t, hn_t = net(inputs[:, t, :].unsqueeze(1), hn_t)
        output_l.append(out_t.squeeze(1))

    out_steps = torch.stack(output_l, dim=1)
    assert torch.all(torch.eq(out_steps, output)), "GRU output mismatch"
    assert torch.all(torch.eq(hn_t, hn)), "GRU hidden states mismatch"

    # Check LSTM
    inputs = torch.randn(4, 2, 7)
    net = RNN(
        rnn_type="lstm",
        n_neurons=5,
        num_layers=2,
        return_hidden=True,
        bidirectional=False,
    )
    output, hn = net(inputs, init_params=True)
    output_l = []
    hn_t = None
    for t in range(inputs.shape[1]):
        out_t, hn_t = net(inputs[:, t, :].unsqueeze(1), hn_t)
        output_l.append(out_t.squeeze(1))

    out_steps = torch.stack(output_l, dim=1)
    assert torch.all(torch.eq(out_steps, output)), "LSTM output mismatch"
    assert torch.all(torch.eq(hn_t[0], hn[0])) and torch.all(
        torch.eq(hn_t[1], hn[1])
    ), "LSTM hidden states mismatch"

    # Check LiGRU
    inputs = torch.randn(4, 2, 7)
    net = RNN(
        rnn_type="ligru",
        n_neurons=5,
        num_layers=2,
        return_hidden=True,
        bidirectional=False,
    )
    output, hn = net(inputs, init_params=True)
    output_l = []
    hn_t = None
    for t in range(inputs.shape[1]):
        out_t, hn_t = net(inputs[:, t, :].unsqueeze(1), hn_t)
        output_l.append(out_t.squeeze(1))

    out_steps = torch.stack(output_l, dim=1)
    # output missmatch
    # TODO: layer normalization
    # assert torch.all(torch.eq(out_steps,output)), "LiGRU output mismatch"
    # assert torch.all(torch.eq(hn_t,hn)), "LiGRU hidden states mismatch"
