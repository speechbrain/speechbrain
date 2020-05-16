def test_pad_ends():
    from speechbrain.lm.counting import pad_ends

    assert next(pad_ends(["a", "b", "c"], n=2)) == "<s>"
    assert next(pad_ends(["a", "b", "c"], n=1)) == "a"
    assert list(pad_ends(["a", "b", "c"], n=1))[-1] == "</s>"
    assert list(pad_ends([], n=1))
    assert list(pad_ends([], n=2))


def test_ngrams():
    from speechbrain.lm.counting import ngrams

    assert next(ngrams(["a", "b", "c"], n=3)) == ("a", "b", "c")
    assert next(ngrams(["a", "b", "c"], n=1)) == ("a",)
    assert not list(ngrams(["a", "b", "c"], n=4))
    assert list(ngrams(["a", "b", "c"], n=2)) == [("a", "b"), ("b", "c")]


def test_ngrams_for_evaluation():
    from speechbrain.lm.counting import ngrams_for_evaluation

    assert list(ngrams_for_evaluation(["a", "b", "c"], max_n=3)) == [
        ("a", ()),
        ("b", ("a",)),
        ("c", ("a", "b")),
    ]
