import pytest


def test_nest():
    from speechbrain.core import nest

    params = {"a": {"b": "c"}}
    nest(params, ["a", "b"], "d")
    assert params == {"a": {"b": "d"}}
    nest(params, ["a", "c"], "e")
    assert params == {"a": {"b": "d", "c": "e"}}
    nest(params, ["a", "b"], {"d": "f"})
    assert params == {"a": {"b": {"d": "f"}, "c": "e"}}


def test_parse_arguments():
    from speechbrain.core import parse_arguments

    args = parse_arguments(["--seed", "3", "--ckpts_to_save", "2"])
    assert args == {"seed": 3, "ckpts_to_save": 2}


def test_parse_overrides():
    from speechbrain.core import parse_overrides

    overrides = parse_overrides("{model.arg1: 1, model.arg1: 2}")
    assert overrides == {"model": {"arg1": 2}}


def test_experiment(tmpdir):
    from speechbrain.core import Experiment

    yaml = f"""
    constants:
        output_folder: {tmpdir}
    """
    sb = Experiment(yaml)
    assert sb.output_folder == tmpdir
    sb = Experiment(
        yaml_stream=yaml,
        overrides={"constants": {"output_folder": f"{tmpdir}/example"}},
    )
    assert sb.output_folder == f"{tmpdir}/example"
    sb = Experiment(
        yaml, commandline_args=["--output_folder", f"{tmpdir}/example"]
    )
    assert sb.output_folder == f"{tmpdir}/example"
    sb = Experiment(
        yaml_stream=yaml,
        overrides={"constants": {"output_folder": f"{tmpdir}/abc"}},
        commandline_args=["--output_folder", f"{tmpdir}/example"],
    )
    assert sb.output_folder == f"{tmpdir}/example"

    yaml = f"""
    constants:
        output_folder: {tmpdir}
    functions:
        output_folder: y
    """
    with pytest.raises(KeyError):
        sb = Experiment(yaml)
