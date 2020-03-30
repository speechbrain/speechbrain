import pytest


def test_load_extended_yaml():
    from speechbrain.utils.data_utils import load_extended_yaml

    # Basic functionality
    yaml = """
    constants:
        a: 1
    thing: !collections.Counter
    """
    things = load_extended_yaml(yaml)
    assert things['constants']['a'] == 1
    from collections import Counter
    assert things['thing'].__class__ == Counter

    # Missing sections
    yaml = """
    """
    things = load_extended_yaml(yaml)
    assert things is None

    # String replacement
    yaml = """
    constants:
        a: abc
        b: !$ <constants.a>
    thing: !collections.Counter
        a: !$ <constants.a>
    """
    things = load_extended_yaml(yaml)
    assert things['thing']['a'] == things['constants']['a']
    assert things['constants']['a'] == things['constants']['b']

    # String interpolation
    yaml = """
    constants:
        a: "a"
        b: !$ <constants.a>/b
    """
    things = load_extended_yaml(yaml)
    assert things['constants']['b'] == 'a/b'

    # Substitution with string conversion
    yaml = """
    constants:
        a: 1
        b: !$ <constants.a>/b
    """
    things = load_extended_yaml(yaml)
    assert things['constants']['b'] == '1/b'

    # Nested structures:
    yaml = """
    constants:
        a: 1
    thing: !collections.Counter
        other: !collections.Counter
            a: !$ <constants.a>
    """
    things = load_extended_yaml(yaml)
    assert things['thing']['other'].__class__ == Counter
    assert things['thing']['other']['a'] == things['constants']['a']

    # Positional arguments
    yaml = """
    constants:
        a: hello
    thing: !collections.Counter
        - !$ <constants.a>
    """
    things = load_extended_yaml(yaml)
    assert things['thing']['l'] == 2

    # Invalid class
    yaml = """
    thing: !abcdefg.hij
    """
    with pytest.raises(ValueError):
        things = load_extended_yaml(yaml)

    # Invalid reference
    yaml = """
    constants:
        a: 1
        b: !$ <constants.c>
    """
    with pytest.raises(ValueError):
        things = load_extended_yaml(yaml)
