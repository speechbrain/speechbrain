import pytest


def test_load_extended_yaml():
    from speechbrain.utils.data_utils import load_extended_yaml
    # Basic functionality
    # NOTE: using unittest.TestCase as an importable standard library class
    yaml = """
    constants:
        a: 1
    thing: !unittest.TestCase
    """
    things, variables = load_extended_yaml(yaml)
    assert variables.a == 1
    from unittest import TestCase
    assert things.thing.__class__ == TestCase
    # Missing sections
    yaml = """
    thing: !unittest.TestCase
    """
    with pytest.raises(ValueError):
        things, variables = load_extended_yaml(yaml)
    assert variables == {}
    yaml = """
    """
    things, variables = load_extended_yaml(yaml)
    assert things == {}
    assert variables == {}
    # String replacement
    yaml = """
    constants:
        a: 1
        b: !$a
    thing: !unittest.TestCase
        a: !$a
    """
    things, variables = load_extended_yaml(yaml)
    assert things.thing['a'] == variables.a
    assert variables.b == variables.a
    # Partial substitution:
    yaml = """
    constants:
        a: "a"
        b: !$a/b
    """
    things, variables = load_extended_yaml(yaml)
    assert variables['b'] == 'a/b'
    # Partial substitution with bad val raises TypeError:
    yaml = """
    variables:
        a: 1
        b: !$a/
    """
    with pytest.raises(TypeError):
        load_extended_yaml(yaml)
    # Nested structures:
    yaml = """
    variables:
        a: 1
    thing: !unittest.TestCase
        other: !unittest.TestCase
            a: !$a
    """
    things, variables = load_extended_yaml(yaml)
    assert things['thing']['other']['object'] == TestCase
    assert things['thing']['other']['a'] == variables.a
    yaml = """
    variables:
        a: 1
    thing:
        - !$a
        - object: !unittest.TestCase
    """
    things, variables = load_extended_yaml(yaml)
    assert isinstance(things['thing'][1], dict)
    assert things['thing'][1]['object'] == TestCase
    assert things['thing'][0] == variables.a


def test_make_object():
    from speechbrain.core import make_object
    spec = {"object": float}
    obj = make_object(spec)
    assert obj == 0.

    class Mock:
        def __init__(self, arg, kwarg="foo"):
            self.arg = arg
            self.kwarg = kwarg
    spec = {"object": Mock, "arg": "test"}
    obj = make_object(spec)
    assert obj.arg == "test"
    assert obj.kwarg == "foo"
    spec = {"object": Mock, "arg": "test", "kwarg": "bar"}
    obj = make_object(spec)
    assert obj.kwarg == "bar"


def test_make_all_objects():
    from speechbrain.core import make_all_objects
    nested_specs = {"foo": {"object": float}}
    f = make_all_objects(nested_specs)
    assert f.foo == 0.
    nested_specs = {"foo": {"object": float}, "bar": 1.}
    f = make_all_objects(nested_specs)
    assert f.foo == 0.
    assert f.bar == 1.
    nested_specs = {"foo": [{"object": float}, 1.]}
    f = make_all_objects(nested_specs)
    assert f.foo[0] == 0.
    assert f.foo[1] == 1.
    nested_specs = {"foo": [{"object": float}, 1.]}
    f = make_all_objects(nested_specs)
    assert f.foo[0] == 0.
    assert f.foo[1] == 1.

    class Mock:
        def __init__(self, arg, kwarg="foo"):
            self.arg = arg
            self.kwarg = kwarg
    nested_specs = {"foo": {"object": Mock, "arg": "test"},
                    "bar": {"object": Mock, "arg": "test", "kwarg": "bar"}}
    f = make_all_objects(nested_specs)
    assert f.foo.arg == "test"
    assert f.foo.kwarg == "foo"
    assert f.bar.kwarg == "bar"
