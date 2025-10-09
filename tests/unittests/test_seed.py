from speechbrain.utils import seed_everything


def test_default_seed():
    assert seed_everything() == 0


def test_correct_seed_with_environment_variable():
    assert seed_everything(seed=42) == 42


def test_invalid_seed():
    out_of_bound_seed = 10e9
    seed = seed_everything(seed=out_of_bound_seed)
    assert seed == 0

    out_of_bound_seed = -10e9
    seed = seed_everything(seed=out_of_bound_seed)
    assert seed == 0
