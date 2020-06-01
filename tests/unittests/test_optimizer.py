def test_optimizers():
    import torch
    from speechbrain.nnet.linear import Linear
    from speechbrain.nnet.losses import nll_loss

    inp_tensor = torch.rand([1, 660, 3])
    model = Linear(n_neurons=4)

    from speechbrain.nnet.optimizers import RMSprop_Optimizer

    optim = RMSprop_Optimizer(learning_rate=0.01)
    output = model(inp_tensor, init_params=True)
    optim.init_params([model])
    prediction = torch.nn.functional.log_softmax(output, dim=2)
    label = torch.randint(3, (1, 660))
    lengths = torch.Tensor([1.0])
    out_cost = nll_loss(prediction, label, lengths)
    out_cost.backward()
    optim.step()
    optim.zero_grad()

    from speechbrain.nnet.optimizers import Adam_Optimizer

    optim = Adam_Optimizer(learning_rate=0.01)
    output = model(inp_tensor, init_params=True)
    optim.init_params([model])
    prediction = torch.nn.functional.log_softmax(output, dim=2)
    label = torch.randint(3, (1, 660))
    lengths = torch.Tensor([1.0])
    out_cost = nll_loss(prediction, label, lengths)
    out_cost.backward()
    optim.step()
    optim.zero_grad()

    from speechbrain.nnet.optimizers import Adadelta_Optimizer

    optim = Adadelta_Optimizer(learning_rate=0.01)
    output = model(inp_tensor, init_params=True)
    optim.init_params([model])
    prediction = torch.nn.functional.log_softmax(output, dim=2)
    label = torch.randint(3, (1, 660))
    lengths = torch.Tensor([1.0])
    out_cost = nll_loss(prediction, label, lengths)
    out_cost.backward()
    optim.step()
    optim.zero_grad()

    from speechbrain.nnet.optimizers import SGD_Optimizer

    optim = SGD_Optimizer(learning_rate=0.01)
    output = model(inp_tensor, init_params=True)
    optim.init_params([model])
    prediction = torch.nn.functional.log_softmax(output, dim=2)
    label = torch.randint(3, (1, 660))
    lengths = torch.Tensor([1.0])
    out_cost = nll_loss(prediction, label, lengths)
    out_cost.backward()
    optim.step()
    optim.zero_grad()

    from speechbrain.nnet.optimizers import Rprop_Optimizer

    optim = Rprop_Optimizer(learning_rate=0.01)
    output = model(inp_tensor, init_params=True)
    optim.init_params([model])
    prediction = torch.nn.functional.log_softmax(output, dim=2)
    label = torch.randint(3, (1, 660))
    lengths = torch.Tensor([1.0])
    out_cost = nll_loss(prediction, label, lengths)
    out_cost.backward()
    optim.step()
    optim.zero_grad()
