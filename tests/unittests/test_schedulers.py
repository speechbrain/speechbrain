def test_NewBobScheduler():

    from speechbrain.nnet.schedulers import NewBobScheduler

    scheduler = NewBobScheduler(initial_value=0.8)

    prev_lr, next_lr = scheduler(1.0)
    assert prev_lr == 0.8
    assert next_lr == 0.8

    prev_lr, next_lr = scheduler(1.1)
    assert next_lr == 0.4

    prev_lr, next_lr = scheduler(0.5)
    assert next_lr == 0.4

    scheduler = NewBobScheduler(initial_value=0.8, patient=3)
    prev_lr, next_lr = scheduler(1.0)
    assert next_lr == 0.8

    prev_lr, next_lr = scheduler(1.1)
    prev_lr, next_lr = scheduler(1.1)
    prev_lr, next_lr = scheduler(1.1)
    assert next_lr == 0.8

    prev_lr, next_lr = scheduler(1.1)
    assert next_lr == 0.4
    assert scheduler.current_patient == 3


def test_IntervalScheduler():
    from speechbrain.nnet.schedulers import IntervalScheduler
    from torch.optim import Adam
    from torch import nn

    scheduler = IntervalScheduler(
        intervals=[
            {'steps': 5, 'lr': 0.1},
            {'steps': 10, 'lr': 0.01},
            {'steps': 20, 'lr': 0.001}
        ]
    )
    model = nn.Linear(10, 2)
    opt = Adam(model.parameters(), lr=0.5)
    for _ in range(4):
        prev_lr, next_lr = scheduler(opt)
        assert prev_lr == next_lr == 0.5

    prev_lr, next_lr = scheduler(opt)
    assert prev_lr == 0.5
    assert next_lr == 0.1
    assert opt.param_groups[0]['lr'] == 0.1

    for _ in range(4):
        prev_lr, next_lr = scheduler(opt)
        assert prev_lr == next_lr == 0.1

    prev_lr, next_lr = scheduler(opt)
    assert prev_lr == 0.1
    assert next_lr == 0.01
    assert opt.param_groups[0]['lr'] == 0.01

    for _ in range(9):
        prev_lr, next_lr = scheduler(opt)
        assert prev_lr == next_lr == 0.01

    prev_lr, next_lr = scheduler(opt)
    assert prev_lr == 0.01
    assert next_lr == 0.001
    assert opt.param_groups[0]['lr'] == 0.001

    for _ in range(50):
        prev_lr, next_lr = scheduler(opt)
        assert prev_lr == next_lr == 0.001

