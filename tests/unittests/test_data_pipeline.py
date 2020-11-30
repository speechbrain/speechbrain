def test_data_pipeline():
    from speechbrain.utils.data_pipeline import DataPipeline

    pipeline = DataPipeline.from_configuration(
        funcs={
            "foo": {"func": lambda x: x.lower(), "argnames": ["text"]},
            "bar": {"func": lambda x: x[::-1], "argnames": ["foo"]},
        },
        final_names=["bar"],
    )
    result = pipeline({"text": "Test"})
    assert result["bar"] == "tset"
    pipeline = DataPipeline()
    pipeline.add_func(
        "foobar", func=lambda x, y: x + y, argnames=["foo", "bar"]
    )
    pipeline.final_names.append("foobar")
    result = pipeline({"foo": 1, "bar": 2})
    assert result["foobar"] == 3
    pipeline = DataPipeline()
    from unittest.mock import MagicMock, Mock

    watcher = Mock()
    pipeline.add_func("foobar", func=watcher, argnames=["foo", "bar"])
    result = pipeline({"foo": 1, "bar": 2})
    assert not watcher.called
    pipeline = DataPipeline()
    watcher = MagicMock(return_value=3)
    pipeline.add_func("foobar", func=watcher, argnames=["foo", "bar"])
    pipeline.add_func("truebar", func=lambda x: x, argnames=["foobar"])
    pipeline.final_names.append("truebar")
    result = pipeline({"foo": 1, "bar": 2})
    assert watcher.called
    assert result["truebar"] == 3
    pipeline = DataPipeline()
    watcher = MagicMock(return_value=3)
    pipeline.add_func("foobar", func=watcher, argnames=["foo", "bar"])
    pipeline.add_func("truebar", func=lambda x: x, argnames=["foo"])
    pipeline.final_names.append("truebar")
    result = pipeline({"foo": 1, "bar": 2})
    assert not watcher.called
    assert result["truebar"] == 1
