import os


def test_get_all_files_exclude_and(tmpdir):
    from speechbrain.utils.data_utils import get_all_files

    names = ["a.wav", "b.wav", "c.wav"]
    for name in names:
        tmpdir.join(name).write("")

    for excluded in names:
        found = get_all_files(str(tmpdir), exclude_and=[excluded])
        assert sorted(os.path.basename(f) for f in found) == [
            name for name in names if name != excluded
        ]
