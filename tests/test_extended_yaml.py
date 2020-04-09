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

    overrides = {'constants': {'a': 2}}
    things = load_extended_yaml(yaml, overrides=overrides)

    # Missing sections
    yaml = """
    """
    things = load_extended_yaml(yaml)
    assert things is None

    # String replacement
    yaml = """
    constants:
        a: abc
        b: !ref <constants.a>
    thing: !collections.Counter
        a: !ref <constants.a>
    """
    things = load_extended_yaml(yaml)
    assert things['thing']['a'] == things['constants']['a']
    assert things['constants']['a'] == things['constants']['b']

    # String interpolation
    yaml = """
    constants:
        a: "a"
        b: !ref <constants.a>/b
    """
    things = load_extended_yaml(yaml)
    assert things['constants']['b'] == 'a/b'

    # Substitution with string conversion
    yaml = """
    constants:
        a: 1
        b: !ref <constants.a>/b
    """
    things = load_extended_yaml(yaml)
    assert things['constants']['b'] == '1/b'

    # Nested structures:
    yaml = """
    constants:
        a: 1
    thing: !collections.Counter
        other: !collections.Counter
            a: !ref <constants.a>
    """
    things = load_extended_yaml(yaml)
    assert things['thing']['other'].__class__ == Counter
    assert things['thing']['other']['a'] == things['constants']['a']

    # Positional arguments
    yaml = """
    constants:
        a: hello
    thing: !collections.Counter
        - !ref <constants.a>
    """
    things = load_extended_yaml(yaml)
    assert things['thing']['l'] == 2

    # Invalid class
    yaml = """
    thing: !abcdefg.hij
    """
    with pytest.raises(ImportError):
        things = load_extended_yaml(yaml)

    # Invalid reference
    yaml = """
    constants:
        a: 1
        b: !ref <constants.c>
    """
    with pytest.raises(ValueError):
        things = load_extended_yaml(yaml)

    # Anchors and aliases
    yaml = """
    thing1: !collections.Counter &thing
        a: 3
        b: 5
    thing2: !collections.Counter
        <<: *thing
        b: 7
    """
    things = load_extended_yaml(yaml)
    assert things['thing1']['a'] == things['thing2']['a']
    assert things['thing1']['b'] != things['thing2']['b']

    # List of arguments that are objects
    yaml = """
    thing: !os.path.join
        - !os.path.join
            - abc
            - def
        - !os.path.join
            - foo
            - bar
    """
    things = load_extended_yaml(yaml)
    assert things['thing'] == 'abc/def/foo/bar'

    yaml = """
    thing1: !collections.Counter
        a: 3
        b: 5
    thing2: !ref <thing1>
    """
    things = load_extended_yaml(yaml)
    assert things['thing2']['b'] == things['thing1']['b']
    things['thing2']['b'] = 7
    assert things['thing2']['b'] == things['thing1']['b']
