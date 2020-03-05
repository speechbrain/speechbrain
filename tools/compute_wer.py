#!/usr/bin/env python3
import sys
import collections
import speechbrain.utils.edit_distance as edit_distance

"""
Description:
    This script computes Word Error Rate and other related information.
    Just given a reference and a hypothesis, the script closely matches
    Kaldi's compute_wer binary.
    Additionally, the script can produce human readable edit distance
    alignments, and find the top WER utterances and speakers.

    The functionality of the script can also be used as an imported module.
    See the if __name__ == "__main__": block at the bottom for usage examples.
    Also see speechbrain.utils.edit_distance, particularly
    accumulatable_wer_stats, which may be nicer to integrate into a training
    routine.

Author:
    Aku Rouhe 2020
"""


# These internal utilities read Kaldi-style text/utt2spk files:
def _plain_text_reader(path):
    # yields key, token_list
    with open(path, "r") as fi:
        for line in fi:
            key, *tokens = line.strip().split()
            yield key, tokens


def _plain_text_keydict(path):
    out_dict = {}  # key: token_list
    for key, tokens in plain_text_reader(path):
        out_dict[key] = tokens
    return out_dict


def _utt2spk_keydict(path):
    utt2spk = {}
    with open(path, "r") as fi:
        for line in fi:
            utt, spk = line.strip().split()
            utt2spk[utt] = spk
    return utt2spk


def wer_details_by_utterance(
    ref_reader, hyp_dict, compute_alignments=False, scoring_mode="strict"
):
    """
    Description:
        Computes a wealth of salient info about each single utterance.
        This info can then be used to compute summary details (WER, SER).
    Input:
        ref_reader (type: generator) Should yield a tuple of utterance id
            (hashable) and reference tokens (iterable)
        hyp_dict (type: dict) Should be indexable by utterance ids, and return
            the hypothesis tokens for each utterance id (iterable)
        compute_alignments (type: bool) Whether alignments
            should also be saved.
            This also saves the tokens themselves, as the they are probably
            required for printing the alignments.
        scoring_mode (type: str, one of: ['strict', 'all', 'present'])
            How to deal with missing hypotheses (reference utterance id
                not found in hyp_dict)
            'strict': raise error for missing hypotheses
            'all': score missing hypotheses as empty
            'present': only score existing hypotheses
    Output:
        details_by_utterance (type: list of dicts) A list with one entry
            for every reference utterance. Each entry is a dict with keys:
                "key": utterance id
                "scored": bool, whether utterance was scored
                "hyp_absent": bool, true if a hypothesis was NOT found
                "hyp_empty": bool, true if hypothesis was considered empty
                    (either because it was empty, or not found and mode 'all')
                "num_edits": int, number of edits in total
                "num_ref_tokens": int, number of tokens in the reference
                "WER": float, word error rate of the utterance
                "insertions": int, number of insertions
                "deletions": int, number of deletions
                "substitutions": int, number of substitutions
                "alignment": if compute_alignments is True, alignment as list,
                    see speechbrain.utils.edit_distance.alignment
                    if compute_alignments is False, this is None
                "ref_tokens": iterable, the reference tokens,
                    only saved if alignments were computed, else None
                "hyp_tokens": iterable, the hypothesis tokens,
                    only saved if alignments were computed, else None
    Raises:
        KeyError if scoring mode is 'strict' and a hypothesis is not found
    Author:
        Aku Rouhe
    """
    details_by_utterance = []
    for key, ref_tokens in ref_reader:
        # Initialize utterance_details
        utterance_details = {
            "key": key,
            "scored": False,
            "hyp_absent": None,
            "hyp_empty": None,
            "num_edits": None,
            "num_ref_tokens": len(ref_tokens),
            "WER": None,
            "insertions": None,
            "deletions": None,
            "substitutions": None,
            "alignment": None,
            "ref_tokens": ref_tokens if compute_alignments else None,
            "hyp_tokens": None,
        }
        if key in hyp_dict:
            utterance_details.update({"hyp_absent": False})
            hyp_tokens = hyp_dict[key]
        elif scoring_mode == "all":
            utterance_details.update({"hyp_absent": True})
            hyp_tokens = []
        elif scoring_mode == "present":
            utterance_details.update({"hyp_absent": True})
            details_by_utterance.append(utterance_details)
            continue  # Skip scoring this utterance
        elif scoring_mode == "strict":
            raise KeyError(
                "Key "
                + key
                + " in reference but missing in hypothesis and strict mode on."
            )
        else:
            raise ValueError("Invalid scoring mode: " + scoring_mode)
        # Compute edits for this utterance
        op_table = edit_distance.op_table(ref_tokens, hyp_tokens)
        ops = edit_distance.count_ops(op_table)
        # Update the utterance-level details if we got this far:
        utterance_details.update(
            {
                "scored": True,
                "hyp_empty": True
                if len(hyp_tokens) == 0
                else False,  # This also works for e.g. torch tensors
                "num_edits": sum(ops),
                "num_ref_tokens": len(ref_tokens),
                "WER": 100.0 * sum(ops) / len(ref_tokens),
                "insertions": ops["insertions"],
                "deletions": ops["deletions"],
                "substitutions": ops["substitutions"],
                "alignment": edit_distance.alignment(op_table)
                if compute_alignments
                else None,
                "ref_tokens": ref_tokens if compute_alignments else None,
                "hyp_tokens": hyp_tokens if compute_alignments else None,
            }
        )
        details_by_utterance.append(utterance_details)
    return details_by_utterance


def wer_summary(details_by_utterance):
    """
    Description:
        Computes Word Error Rate and other salient statistics
        over the whole set of utterances.
    Input:
        details_by_utterance (type: list of dicts) See the output of
            wer_details_by_utterance
    Output:
        wer_details (type: dict) Dictionary with keys:
            "WER": float, word error rate
            "SER": float, sentence error rate (percentage of utterances
                which had at least one error)
            "num_edits": int, total number of edits
            "num_scored_tokens": int, total number of tokens in scored
                reference utterances (a missing hypothesis might still
                    have been scored with 'all' scoring mode)
            "num_erraneous_sents": int, total number of utterances
                which had at least one error
            "num_scored_sents": int, total number of utterances
                which were scored
            "num_absent_sents": int, hypotheses which were not found
            "num_ref_sents": int, number of all reference utterances
            "insertions": int, total number of insertions
            "deletions": int, total number of deletions
            "substitutions": int, total number of substitutions
            NOTE: Some cases lead to ambiguity over number of
                insertions, deletions and substitutions. We
                aim to replicate Kaldi compute_wer numbers.
    Author:
        Aku Rouhe
    """
    # Build the summary details:
    ins = dels = subs = 0
    num_scored_tokens = (
        num_scored_sents
    ) = num_edits = num_erraneous_sents = num_absent_sents = num_ref_sents = 0
    for dets in details_by_utterance:
        num_ref_sents += 1
        if dets["scored"]:
            num_scored_sents += 1
            num_scored_tokens += dets["num_ref_tokens"]
            ins += dets["insertions"]
            dels += dets["deletions"]
            subs += dets["substitutions"]
            num_edits += dets["num_edits"]
            if dets["num_edits"] > 0:
                num_erraneous_sents += 1
        if dets["hyp_absent"]:
            num_absent_sents += 1
    wer_details = {
        "WER": 100.0 * num_edits / num_scored_tokens,
        "SER": 100.0 * num_erraneous_sents / num_scored_sents,
        "num_edits": num_edits,
        "num_scored_tokens": num_scored_tokens,
        "num_erraneous_sents": num_erraneous_sents,
        "num_scored_sents": num_scored_sents,
        "num_absent_sents": num_absent_sents,
        "num_ref_sents": num_ref_sents,
        "insertions": ins,
        "deletions": dels,
        "substitutions": subs,
    }
    return wer_details


def wer_details_by_speaker(details_by_utterance, utt2spk):
    """
    Description:
        Compute word error rate and other salient info grouping by speakers.
    Input:
        details_by_utterance (type: list of dicts) See the output of
            wer_details_by_utterance
        utt2spk (type: dict) Map from utterance id to speaker id
    Output:
        details_by_speaker (type: dict of dicts) Maps speaker id
            to a dictionary of the statistics, with keys:
            "speaker": speaker id,
            "num_edits": int, number of edits in total by this speaker
            "insertions": int, number insertions by this speaker
            "dels": int, number of deletions by this speaker
            "subs": int, number of substitutions by this speaker
            "num_scored_tokens": int, number of scored reference
                tokens by this speaker (a missing hypothesis might still
                    have been scored with 'all' scoring mode)
            "num_scored_sents": int, number of scored utterances
                by this speaker
            "num_erraneous_sents": int, number of utterance with at least
                one error, by this speaker
            "num_absent_sents": int, number of utterances for which no
                hypothesis was found, by this speaker
            "num_ref_sents": int, number of utterances by this speaker
                in total
    Author:
        Aku Rouhe
    """
    # Build the speakerwise details:
    details_by_speaker = {}
    for dets in details_by_utterance:
        speaker = utt2spk[dets["key"]]
        spk_dets = details_by_speaker.setdefault(
            speaker,
            collections.Counter(
                {
                    "speaker": speaker,
                    "insertions": 0,
                    "dels": 0,
                    "subs": 0,
                    "num_scored_tokens": 0,
                    "num_scored_sents": 0,
                    "num_edits": 0,
                    "num_erraneous_sents": 0,
                    "num_absent_sents": 0,
                    "num_ref_sents": 0,
                }
            ),
        )
        utt_stats = collections.Counter()
        if dets["hyp_absent"]:
            utt_stats.update({"num_absent_sents": 1})
        if dets["scored"]:
            utt_stats.update(
                {
                    "num_scored_sents": 1,
                    "num_scored_tokens": dets["num_ref_tokens"],
                    "insertions": dets["insertions"],
                    "dels": dets["deletions"],
                    "subs": dets["substitutions"],
                    "num_edits": dets["num_edits"],
                }
            )
            if dets["num_edits"] > 0:
                utt_stats.update({"num_erraneous_sents": 1})
        spk_dets.update(utt_stats)
    # We will in the end return a list of normal dicts
    # We want the output to be sortable
    details_by_speaker_dicts = []
    # Now compute speakerwise summary details
    for speaker, spk_dets in details_by_speaker.items():
        spk_dets["speaker"] = speaker
        if spk_dets["num_scored_sents"] > 0:
            spk_dets["WER"] = (
                100.0 * spk_dets["num_edits"] / spk_dets["num_scored_tokens"]
            )
            spk_dets["SER"] = (
                100.0
                * spk_dets["num_erraneous_sents"]
                / spk_dets["num_scored_sents"]
            )
        else:
            spk_dets["WER"] = None
            spk_dets["SER"] = None
        details_by_speaker_dicts.append(spk_dets)
    return details_by_speaker_dicts


def top_wer_utts(details_by_utterance, top_k=20):
    """
    Description:
        Finds the k utterances with highest word error rates.
        Useful for diagnostic purposes, to see where the system
        is making the most mistakes.
        Returns results utterances which were not empty
        i.e. had to have been present in the hypotheses, with output produced
    Input:
        details_by_utterance (type: list of dicts) See output
            of wer_details_by_utterance
        top_k (type: int) Number of utterances to return
    Output:
        top_non_empty (type: list of dicts) List of at most K utterances,
                with the highest word error rates, which were not empty
                The utterance dict has the same keys as
                details_by_utterance
    Author:
        Aku Rouhe
    """
    scored_utterances = [
        dets for dets in details_by_utterance if dets["scored"]
    ]
    utts_by_wer = sorted(
        scored_utterances, key=lambda d: d["WER"], reverse=True
    )
    top_non_empty = []
    while utts_by_wer and len(top_non_empty) < top_k:
        utt = utts_by_wer.pop(0)
        if not utt["hyp_empty"]:
            top_non_empty.append(utt)
    return top_non_empty


def top_wer_spks(details_by_speaker, top_k=10):
    """
    Description:
        Finds the K speakers with highest word error rates.
        Useful for diagnostic purposes.
    Input:
        details_by_speaker (type: list of dicts) See output of
            wer_details_by_speaker
        top_k (type: int) Number of seakers to return
    Output:
        spks_by_wer (type: list of dicts) List of at most K
            dicts (with the same keys as details_by_speaker)
            of speakers sorted by WER.
    Author:
        Aku Rouhe
    """
    scored_speakers = [
        dets for dets in details_by_speaker if dets["num_scored_sents"] > 0
    ]
    spks_by_wer = sorted(scored_speakers, key=lambda d: d["WER"], reverse=True)
    if len(spks_by_wer) >= top_k:
        return spks_by_wer[:top_k]
    else:
        return spks_by_wer


# The following internal functions are used to print the computed statistics
# with human readable formatting.
def _print_wer_summary(wer_details, file=sys.stdout):
    # This function essentially mirrors the Kaldi compute-wer output format
    print(
        "%WER {WER:.2f} [ {num_edits} / {num_scored_tokens}, {ins} ins, {del} del, {sub} sub ]".format(  # noqa
            **wer_details
        ),
        file=file,
        end="",
    )
    print(
        " [PARTIAL]"
        if wer_details["num_scored_sents"] < wer_details["num_ref_sents"]
        else "",
        file=file,
    )
    print(
        "%SER {SER:.2f} [ {num_erraneous_sents} / {num_scored_sents} ]".format(
            **wer_details
        ),
        file=file,
    )
    print(
        "Scored {num_scored_sents} sentences, {num_absent_sents} not present in hyp.".format(  # noqa
            **wer_details
        ),
        file=file,
    )


def _print_top_wer_utts(top_non_empty, file=sys.stdout):
    print("=" * 80, file=file)
    print("UTTERANCES WITH HIGHEST WER", file=file)
    if top_non_empty:
        print(
            "Non-empty hypotheses -- utterances for which output was produced:",  # noqa
            file=sys.stdout,
        )
        for dets in top_non_empty:
            print("{key} %WER {WER:.2f}".format(**dets))


def _print_top_wer_spks(spks_by_wer, file=sys.stdout):
    print("=" * 80, file=file)
    print("SPEAKERS WITH HIGHEST WER", file=file)
    for dets in spks_by_wer:
        print("{speaker} %WER {WER:.2f}".format(**dets))


def _print_alignment(
    alignment, a, b, empty_symbol="<eps>", separator=" ; ", file=sys.stdout
):
    # First, get equal length text for all:
    a_padded = []
    b_padded = []
    ops_padded = []
    for op, i, j in alignment:  # i indexes a, j indexes b
        op_string = str(op)
        a_string = str(a[i]) if i is not None else empty_symbol
        b_string = str(b[j]) if j is not None else empty_symbol
        # NOTE: the padding does not actually compute printed length,
        # but hopefully we can assume that printed length is
        # at most the str len
        pad_length = max(len(op_string), len(a_string), len(b_string))
        a_padded.append(a_string.center(pad_length))
        b_padded.append(b_string.center(pad_length))
        ops_padded.append(op_string.center(pad_length))
    # Then print, in the order Ref, op, Hyp
    print(separator.join(a_padded), file=file)
    print(separator.join(ops_padded), file=file)
    print(separator.join(b_padded), file=file)


def _print_alignments_global_header(
    empty_symbol="<eps>", separator=" ; ", file=sys.stdout
):
    print("=" * 80, file=file)
    print("ALIGNMENTS", file=file)
    print("", file=file)
    print("Format:", file=file)
    print("<utterance-id>, WER DETAILS", file=file)
    # Print the format with the actual
    # print_alignment function, using artificial data:
    a = ["reference", "on", "the", "first", "line"]
    b = ["and", "hypothesis", "on", "the", "third"]
    alignment = [
        (edit_distance.EDIT_SYMBOLS["ins"], None, 0),
        (edit_distance.EDIT_SYMBOLS["sub"], 0, 1),
        (edit_distance.EDIT_SYMBOLS["eq"], 1, 2),
        (edit_distance.EDIT_SYMBOLS["eq"], 2, 3),
        (edit_distance.EDIT_SYMBOLS["sub"], 3, 4),
        (edit_distance.EDIT_SYMBOLS["del"], 4, None),
    ]
    _print_alignment(alignment, a, b)


def _print_alignment_header(wer_details, file=sys.stdout):
    print("=" * 80)
    print(
        "{key}, %WER {WER:.2f} [ {num_edits} / {num_ref_tokens}, {ins} ins, {del} del, {sub} sub ]".format(  # noqa
            **wer_details
        ),
        file=file,
    )


if __name__ == "__main__":
    import argparse

    # See: https://stackoverflow.com/a/22157136
    class SmartFormatter(argparse.HelpFormatter):
        def _split_lines(self, text, width):
            if text.startswith("R|"):
                return text[2:].splitlines()
            return argparse.HelpFormatter._split_lines(self, text, width)

    parser = argparse.ArgumentParser(
        description=("Compute word error rate or a Levenshtein alignment"
                     "between a hypothesis and a reference."),
        formatter_class=SmartFormatter,
    )
    parser.add_argument(
        "ref",
        help="The ground truth to compare against. \
            Text file with utterance-ID on the first column.",
    )
    parser.add_argument(
        "hyp",
        help="The hypothesis, for which WER is computed. \
            Text file with utterance-ID on the first column.",
    )
    parser.add_argument(
        "--mode",
        help="R|How to treat missing hypotheses.\n"
        " 'present': only score hypotheses that were found\n"
        " 'all': treat missing hypotheses as empty\n"
        " 'strict': raise KeyError if a hypothesis is missing",
        choices=["present", "all", "strict"],
        default="strict",
    )
    parser.add_argument(
        "--print-top-wer",
        action="store_true",
        help="Print a list of utterances with the highest WER.",
    )
    parser.add_argument(
        "--print-alignments",
        action="store_true",
        help=("Print alignments for between all refs and hyps."
              "Also has details for individual hyps. Outputs a lot of text."),
    )
    parser.add_argument(
        "--align-separator",
        default=" ; ",
        help=("When printing alignments, separate tokens with this."
              "Note the spaces in the default."),
    )
    parser.add_argument(
        "--align_empty",
        default="<eps>",
        help="When printing alignments, empty spaces are filled with this.",
    )
    parser.add_argument(
        "--utt2spk",
        help="Provide a mapping from utterance ids to speaker ids."
        "If provided, print a list of speakers with highest WER.",
    )
    args = parser.parse_args()
    details_by_utterance = wer_details_by_utterance(
        _plain_text_reader(args.ref),
        _plain_text_keydict(args.hyp),
        compute_alignments=args.print_alignments,
        scoring_mode=args.mode,
    )
    summary_details = wer_summary(details_by_utterance)
    _print_wer_summary(summary_details)
    if args.print_top_wer:
        top_non_empty = top_wer_utts(details_by_utterance)
        _print_top_wer_utts(top_non_empty)
    if args.utt2spk:
        utt2spk = _utt2spk_keydict(args.utt2spk)
        details_by_speaker = wer_details_by_speaker(
            details_by_utterance, utt2spk
        )
        top_spks = top_wer_spks(details_by_speaker)
        _print_top_wer_spks(top_spks)
    if args.print_alignments:
        _print_alignments_global_header(
            separator=args.align_separator, empty_symbol=args.align_empty
        )
        for dets in details_by_utterance:
            if dets["scored"]:
                _print_alignment_header(dets)
                _print_alignment(
                    dets["alignment"],
                    dets["ref_tokens"],
                    dets["hyp_tokens"],
                    separator=args.align_separator,
                )
