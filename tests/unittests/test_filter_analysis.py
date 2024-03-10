from speechbrain.utils.filter_analysis import FilterProperties


def test_simple_filter_stacks():
    assert FilterProperties(window_size=3, stride=2).with_on_top(
        FilterProperties(window_size=3, stride=2)
    ) == FilterProperties(window_size=7, stride=4)

    assert FilterProperties(window_size=3, stride=1).with_on_top(
        FilterProperties(window_size=3, stride=1)
    ) == FilterProperties(window_size=5, stride=1)


def test_causal_filter_properties():
    assert FilterProperties(
        3, 1, causal=True
    ).get_noncausal_equivalent() == FilterProperties(5, 1)
