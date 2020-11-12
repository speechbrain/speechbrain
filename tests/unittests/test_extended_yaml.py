import pytest


def test_load_extended_yaml():
    from speechbrain.yaml import (
        load_extended_yaml,
        RefTag,
        Placeholder,
        dump_extended_yaml,
    )

    # Basic functionality
    yaml = """
    a: 1
    thing: !new:collections.Counter {}
    """
    things = load_extended_yaml(yaml)
    assert things["a"] == 1
    from collections import Counter

    assert things["thing"].__class__ == Counter

    overrides = {"a": 2}
    things = load_extended_yaml(yaml, overrides=overrides)
    assert things["a"] == 2
    overrides = "{a: 2}"
    things = load_extended_yaml(yaml, overrides=overrides)
    assert things["a"] == 2
    overrides = "{thing: !new:collections.Counter {b: 3}}"
    things = load_extended_yaml(yaml, overrides=overrides)
    assert things["thing"]["b"] == 3

    # String replacement
    yaml = """
    a: abc
    b: !ref <a>
    thing: !new:collections.Counter
        a: !ref <a>
    thing2: !new:zip
        - !ref <a>
        - abc
    """
    things = load_extended_yaml(yaml)
    assert things["thing"]["a"] == things["a"]
    assert things["a"] == things["b"]
    assert next(things["thing2"]) == ("a", "a")

    # String interpolation
    yaml = """
    a: "a"
    b: !ref <a>/b
    """
    things = load_extended_yaml(yaml)
    assert things["b"] == "a/b"

    # Substitution with string conversion
    yaml = """
    a: 1
    b: !ref <a>/b
    """
    things = load_extended_yaml(yaml)
    assert things["b"] == "1/b"

    # Nested structures:
    yaml = """
    constants:
        a: 1
    thing: !new:collections.Counter
        other: !new:collections.Counter
            a: !ref <constants[a]>
    """
    things = load_extended_yaml(yaml)
    assert things["thing"]["other"].__class__ == Counter
    assert things["thing"]["other"]["a"] == things["constants"]["a"]

    # Positional arguments
    yaml = """
    a: hello
    thing: !new:collections.Counter
        - !ref <a>
    """
    things = load_extended_yaml(yaml)
    assert things["thing"]["l"] == 2

    # Invalid class
    yaml = """
    thing: !new:abcdefg.hij
    """
    with pytest.raises(ImportError):
        things = load_extended_yaml(yaml)

    # Invalid reference
    yaml = """
    constants:
        a: 1
        b: !ref <constants[c]>
    """
    with pytest.raises(ValueError):
        things = load_extended_yaml(yaml)

    # Anchors and aliases
    yaml = """
    thing1: !new:collections.Counter &thing
        a: 3
        b: 5
    thing2: !new:collections.Counter
        <<: *thing
        b: 7
    """
    things = load_extended_yaml(yaml)
    assert things["thing1"]["a"] == things["thing2"]["a"]
    assert things["thing1"]["b"] != things["thing2"]["b"]

    # Test references point to same object
    yaml = """
    thing1: !new:collections.Counter
        a: 3
        b: 5
    thing2: !ref <thing1>
    thing3: !new:speechbrain.TestThing
        - !ref <thing1>
        - abc
    """
    things = load_extended_yaml(yaml)
    assert things["thing2"]["b"] == things["thing1"]["b"]
    things["thing2"]["b"] = 7
    assert things["thing2"]["b"] == things["thing1"]["b"]
    assert things["thing3"].args[0] == things["thing1"]

    # Copy tag
    yaml = """
    thing1: !new:collections.Counter
        a: 3
        b: 5
    thing2: !copy <thing1>
    """
    things = load_extended_yaml(yaml)
    assert things["thing2"]["b"] == things["thing1"]["b"]
    things["thing2"]["b"] = 7
    assert things["thing2"]["b"] != things["thing1"]["b"]

    # Name tag
    yaml = """
    Counter: !name:collections.Counter
    """
    things = load_extended_yaml(yaml)
    counter = things["Counter"]()
    assert counter.__class__ == Counter

    # Module tag
    yaml = """
    mod: !module:collections
    """
    things = load_extended_yaml(yaml)
    assert things["mod"].__name__ == "collections"

    # Apply tag
    yaml = """
    a: !apply:sum [[1, 2]]
    """
    things = load_extended_yaml(yaml)
    assert things["a"] == 3

    # Refattr:
    yaml = """
    thing1: "A string"
    thing2: !ref <thing1.lower>
    thing3: !new:speechbrain.TestThing
        - !ref <thing1.lower>
        - abc
    """
    things = load_extended_yaml(yaml)
    assert things["thing2"]() == "a string"
    print(things["thing3"].args)
    assert things["thing3"].args[0]() == "a string"

    # Dumping
    dump_dict = {
        "data_folder": Placeholder(),
        "examples": {"ex1": RefTag("data_folder", suffix="/ex1.wav")},
    }

    from io import StringIO

    stringio = StringIO()
    dump_extended_yaml(dump_dict, stringio)
    assert stringio.getvalue() == (
        "data_folder: !PLACEHOLDER\nexamples:\n"
        "  ex1: !ref <data_folder>/ex1.wav\n"
    )
