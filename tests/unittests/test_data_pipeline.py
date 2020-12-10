def test_data_pipeline():
    from speechbrain.utils.data_pipeline import DataPipeline

    pipeline = DataPipeline.from_configuration(
        dynamic_items={
            "foo": {"func": lambda x: x.lower(), "argkeys": ["text"]},
            "bar": {"func": lambda x: x[::-1], "argkeys": ["foo"]},
        },
        output_keys=["bar"],
    )
    result = pipeline({"text": "Test"})
    assert result["bar"] == "tset"
    pipeline = DataPipeline()
    pipeline.add_dynamic_item(
        "foobar", func=lambda x, y: x + y, argkeys=["foo", "bar"]
    )
    pipeline.output_keys.append("foobar")
    result = pipeline({"foo": 1, "bar": 2})
    assert result["foobar"] == 3
    pipeline = DataPipeline()
    from unittest.mock import MagicMock, Mock

    watcher = Mock()
    pipeline.add_dynamic_item("foobar", func=watcher, argkeys=["foo", "bar"])
    result = pipeline({"foo": 1, "bar": 2})
    assert not watcher.called
    pipeline = DataPipeline()
    watcher = MagicMock(return_value=3)
    pipeline.add_dynamic_item("foobar", func=watcher, argkeys=["foo", "bar"])
    pipeline.add_dynamic_item("truebar", func=lambda x: x, argkeys=["foobar"])
    pipeline.output_keys.append("truebar")
    result = pipeline({"foo": 1, "bar": 2})
    assert watcher.called
    assert result["truebar"] == 3
    pipeline = DataPipeline()
    watcher = MagicMock(return_value=3)
    pipeline.add_dynamic_item("foobar", func=watcher, argkeys=["foo", "bar"])
    pipeline.add_dynamic_item("truebar", func=lambda x: x, argkeys=["foo"])
    pipeline.output_keys.append("truebar")
    result = pipeline({"foo": 1, "bar": 2})
    assert not watcher.called
    assert result["truebar"] == 1

    pipeline = DataPipeline()
    watcher = MagicMock(return_value=3)
    pipeline.add_dynamic_item("foobar", func=watcher, argkeys=["foo", "bar"])
    pipeline.output_keys.append("foobar")
    pipeline.output_keys.append("foo")
    result = pipeline({"foo": 1, "bar": 2})
    assert watcher.called
    assert "foo" in result
    assert "foobar" in result
    assert "bar" not in result
    # Can change the outputs (continues previous tests)
    watcher.reset_mock()
    pipeline.set_output_keys(["bar"])
    result = pipeline({"foo": 1, "bar": 2})
    assert not watcher.called
    assert "foo" not in result
    assert "foobar" not in result
    assert "bar" in result
    # Finally, can also still request any specific key:
    computed = pipeline.compute_specific(["foobar"], {"foo": 1, "bar": 2})
    assert watcher.called
    assert computed["foobar"] == 3
