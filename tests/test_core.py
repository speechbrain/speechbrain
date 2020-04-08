import pytest


def test_nest():
    from speechbrain.core import nest

    params = {'a': {'b': 'c'}}
    nest(params, ['a', 'b'], 'd')
    assert params == {'a': {'b': 'd'}}
    nest(params, ['a', 'c'], 'e')
    assert params == {'a': {'b': 'd', 'c': 'e'}}
    nest(params, ['a', 'b'], {'d': 'f'})
    assert params == {'a': {'b': {'d': 'f'}, 'c': 'e'}}


def test_parse_arguments():
    from speechbrain.core import parse_arguments

    args = parse_arguments(['--seed', '3', '--ckpts_to_save', '2'])
    assert args == {'seed': 3, 'ckpts_to_save': 2}


def test_parse_overrides():
    from speechbrain.core import parse_overrides

    overrides = parse_overrides('{model.arg1: 1, model.arg1: 2}')
    assert overrides == {'model': {'arg1': 2}}


def test_experiment():
    from speechbrain.core import Experiment

    yaml = """
    constants:
        output_folder: exp
    """
    sb = Experiment(yaml)
    assert sb.output_folder == 'exp'
    sb = Experiment(yaml, ['--output_folder', 'exp/example'])
    assert sb.output_folder == 'exp/example'

    yaml = """
    constants:
        output_folder: exp
    functions:
        output_folder: y
    """
    with pytest.raises(KeyError):
        sb = Experiment(yaml)
