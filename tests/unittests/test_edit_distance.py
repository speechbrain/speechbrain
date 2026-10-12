def test_accumulatable_wer_stats():
    from speechbrain.utils.edit_distance import accumulatable_wer_stats

    refs = [[[1, 2, 3], [4, 5, 6]], [[7, 8], [9]]]
    hyps = [[[1, 2, 4], [5, 6]], [[7, 8], [10]]]
    # Test basic functionality:
    stats = accumulatable_wer_stats(refs[0], hyps[0])
    assert stats["WER"] == 100.0 * 2 / 6
    stats = accumulatable_wer_stats(refs[1], hyps[1], stats)
    assert stats["WER"] == 100.0 * 3 / 9
    # Test edge cases:
    import math

    # No batches:
    stats = accumulatable_wer_stats([], [])
    assert stats["num_ref_tokens"] == 0
    assert math.isnan(stats["WER"])
    # Empty hyp sequence:
    stats = accumulatable_wer_stats([[1, 2, 3]], [[]])
    assert stats["num_ref_tokens"] == 3
    assert stats["WER"] == 100.0
    # Empty ref sequence:
    stats = accumulatable_wer_stats([[]], [[1, 2, 3]])
    assert stats["num_ref_tokens"] == 0
    assert stats["insertions"] == 3
    assert math.isnan(stats["WER"])


def test_op_table():
    from speechbrain.utils.edit_distance import EDIT_SYMBOLS, op_table

    assert len(op_table([1, 2, 3], [1, 2, 4])) == 4
    assert len(op_table([1, 2, 3], [1, 2, 4])[0]) == 4
    assert len(op_table([1, 2, 3], [])) == 4
    assert len(op_table([1, 2, 3], [])[0]) == 1
    assert op_table([1, 2, 3], [1, 2, 4])[3][3] == EDIT_SYMBOLS["sub"]
    assert op_table([1, 2, 3], [1, 2, 4])[2][2] == EDIT_SYMBOLS["eq"]
    assert op_table([1, 2, 3], [1, 2, 4])[0][0] == EDIT_SYMBOLS["eq"]


def test_alignment():
    from speechbrain.utils.edit_distance import EDIT_SYMBOLS, alignment

    I = EDIT_SYMBOLS["ins"]  # noqa: E741, here I is a good var name
    D = EDIT_SYMBOLS["del"]
    S = EDIT_SYMBOLS["sub"]
    E = EDIT_SYMBOLS["eq"]
    table = [[I, I, I, I], [D, E, I, I], [D, D, E, I], [D, D, D, S]]
    assert alignment(table) == [(E, 0, 0), (E, 1, 1), (S, 2, 2)]


def test_count_ops():
    from speechbrain.utils.edit_distance import EDIT_SYMBOLS, count_ops

    I = EDIT_SYMBOLS["ins"]  # noqa: E741, here I is a good var name
    D = EDIT_SYMBOLS["del"]
    S = EDIT_SYMBOLS["sub"]
    E = EDIT_SYMBOLS["eq"]
    table = [[I, I, I, I], [D, E, I, I], [D, D, E, I], [D, D, D, S]]
    assert count_ops(table)["insertions"] == 0
    assert count_ops(table)["deletions"] == 0
    assert count_ops(table)["substitutions"] == 1


def test_wer_summary_all_empty_references():
    from speechbrain.utils.edit_distance import (
        wer_details_by_utterance,
        wer_summary,
    )

    # Every reference is empty, but hypotheses have insertions: the summary
    # must not report a perfect WER of 0.0 (see issue #3090).
    refs = {"utt1": [], "utt2": []}
    hyps = {"utt1": ["a", "b"], "utt2": ["c"]}
    details = wer_details_by_utterance(refs, hyps, None, "all")
    summary = wer_summary(details)
    assert summary["num_scored_tokens"] == 0
    assert summary["num_edits"] == 3
    assert summary["WER"] == 100.0 * 3 / 1

    # Empty references AND no edits (empty hypotheses): stays 0.0.
    refs = {"utt1": []}
    hyps = {"utt1": []}
    details = wer_details_by_utterance(refs, hyps, None, "all")
    summary = wer_summary(details)
    assert summary["WER"] == 0.0

    # Regular mixed case unchanged.
    refs = {"utt1": ["a", "b"], "utt2": ["c"]}
    hyps = {"utt1": ["a", "x"], "utt2": ["c"]}
    details = wer_details_by_utterance(refs, hyps, None, "all")
    summary = wer_summary(details)
    assert summary["WER"] == 100.0 * 1 / 3
