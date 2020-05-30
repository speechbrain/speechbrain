def test_NewBobLRScheduler():

    from speechbrain.nnet.lr_schedulers import NewBobLRScheduler

    scheduler = NewBobLRScheduler()

    next_lr = scheduler._new_bob_scheduler(1.0, 1.0, 0.8)
    assert next_lr == 0.4

    next_lr = scheduler._new_bob_scheduler(1.0, 1.1, 0.8)
    assert next_lr == 0.4

    next_lr = scheduler._new_bob_scheduler(1.0, 0.5, 0.8)
    assert next_lr == 0.8

    scheduler = NewBobLRScheduler(patient=3)
    next_lr = scheduler._new_bob_scheduler(1.0, 1.1, 0.8)
    assert next_lr == 0.8

    next_lr = scheduler._new_bob_scheduler(1.0, 1.1, 0.8)
    assert next_lr == 0.8

    next_lr = scheduler._new_bob_scheduler(1.0, 1.1, 0.8)
    assert next_lr == 0.8

    next_lr = scheduler._new_bob_scheduler(1.0, 1.1, 0.8)
    assert next_lr == 0.4
    assert scheduler.current_patient == 3
