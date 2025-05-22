import os


def test_synonym_dictionary_json(tmpdir):
    from speechbrain.utils.dictionaries import SynonymDictionary

    # the synonym dictionary itself is also tested in test_metrics

    tmp_path = os.path.join(tmpdir, "syndict.json")

    with open(tmp_path, "w", encoding="utf8") as f:
        f.write('\n[["a", "a2", "a3"], ["b", "b2", "b3"], ["a", "c"]]')

    syn_dict = SynonymDictionary.from_json_path(tmp_path)

    assert syn_dict("a", "a2")
    assert syn_dict("a3", "a")

    assert syn_dict("b", "b2")
    assert syn_dict("b", "b3")

    assert syn_dict("a", "c")

    assert not syn_dict("a", "b")

    assert not syn_dict("a2", "c")  # not transitive
