def test_dynamic_item_dataset():
    from speechbrain.data_io.dataset import DynamicItemDataset
    import operator

    data = {
        "utt1": {"foo": -1, "bar": 0, "text": "hello world"},
        "utt2": {"foo": 1, "bar": 2, "text": "how are you world"},
        "utt3": {"foo": 3, "bar": 4, "text": "where are you world"},
        "utt4": {"foo": 5, "bar": 6, "text": "hello nation"},
    }
    dynamic_items = {
        "foobar": {"func": operator.add, "argkeys": ["foo", "bar"]}
    }
    output_keys = ["text"]
    dataset = DynamicItemDataset(data, dynamic_items, output_keys)
    assert dataset[0] == {"text": "hello world"}
    dataset.set_output_keys(["id", "foobar"])
    assert dataset[1] == {"id": "utt2", "foobar": 3}
    dataset.add_dynamic_item("barfoo", operator.sub, ["bar", "foo"])
    dataset.set_output_keys(["id", "barfoo"])
    assert dataset[1] == {"id": "utt2", "barfoo": 1}


def test_subset_dynamic_item_dataset():
    from speechbrain.data_io.dataset import DynamicItemDataset
    import operator

    data = {
        "utt1": {"foo": -1, "bar": 0, "text": "hello world"},
        "utt2": {"foo": 1, "bar": 2, "text": "how are you world"},
        "utt3": {"foo": 3, "bar": 4, "text": "where are you world"},
        "utt4": {"foo": 5, "bar": 6, "text": "hello nation"},
    }
    dynamic_items = {
        "foobar": {"func": operator.add, "argkeys": ["foo", "bar"]}
    }
    output_keys = ["text"]
    dataset = DynamicItemDataset(data, dynamic_items, output_keys)
    subset = dataset.filtered_subset(key_min_value={"foo": 3})
    # Note: subset is not a shallow view!
    dataset.set_output_keys(["id", "foo"])
    assert subset[0] == {"text": "where are you world"}
    subset.set_output_keys(["id", "foo"])
    assert subset[0] == {"id": "utt3", "foo": 3}

    # Note: now making a subset from a version which had id and foo as output keys
    subset = dataset.filtered_subset(key_max_value={"bar": 2})
    assert len(subset) == 2
    assert subset[0] == {"id": "utt1", "foo": -1}

    dataset.add_dynamic_item("barfoo", operator.sub, ["bar", "foo"])
    subset = dataset.filtered_subset(key_test={"barfoo": lambda x: x == 1})
    assert len(subset) == 4
    assert subset[3] == {"id": "utt4", "foo": 5}
    subset = dataset.filtered_subset(key_min_value={"foo": 3, "bar": 2})
    assert subset[0]["id"] == "utt3"
    subset = dataset.filtered_subset(
        key_min_value={"foo": 3}, key_max_value={"foobar": 7}
    )
    assert len(subset) == 1
    subset = dataset.filtered_subset(
        key_min_value={"foo": 3}, key_max_value={"foobar": 3}
    )
    assert len(subset) == 0
    subset = dataset.filtered_subset(first_n=1, key_min_value={"foo": 3})
    assert len(subset) == 1
    assert subset[0]["id"] == "utt3"
