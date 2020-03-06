#!/usr/bin/env python3
import collections

EDIT_SYMBOLS = {
    "eq": "=",  # when tokens are equal
    "ins": "I",
    "del": "D",
    "sub": "S",
}


# NOTE: There is a danger in using mutables as default arguments, as they are
# only initialized once, and not every time the function is run. However,
# here the default is not actually ever mutated,
# and simply serves as an empty Counter.
def accumulatable_wer_stats(refs, hyps, stats=collections.Counter()):
    """
    Description:
        Computes word error rate and the related counts for a batch.
        Can also be used to accumulate the counts over many batches, by passing
        the output back to the function in the call for the next batch.
    Input:
        ref: (type: iterable of iterables) Batch of reference sequences
        hyp: (type: iterable of iterables) Batch of hypothesis sequences
        stats (type: collections.Counter) The running statistics.
            Pass the output of this function back as this parameter
            to accumulate the counts. It may be cleanest to initialize
            the stats yourself; then an empty collections.Counter() should
            be used.
    Output:
        updated_stats: (type: collections.Counter) The updated running
            statistics, with keys:
                "WER" - word error rate
                "insertions" - number of insertions
                "deletions" - number of deletions
                "substitutions" - number of substitutions
                "num_ref_tokens" - number of reference tokens
    Example:
        from speechbrain.utils.edit_distance import accumulatable_wer_stats
        import collections
        batches = [[[[1,2,3],[4,5,6]], [[1,2,4],[5,6]]],
                    [[[7,8], [9]],     [[7,8],  [10]]]]
        stats = collections.Counter()
        for batch in batches:
            refs, hyps = batch
            stats = accumulatable_wer_stats(refs, hyps, stats)
        print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))
        # %WER 33.33, 9 ref tokens
    Author:
        Aku Rouhe
    """
    updated_stats = stats + _batch_stats(refs, hyps)
    if updated_stats["num_ref_tokens"] == 0:
        updated_stats["WER"] = float("nan")
    else:
        num_edits = sum(
            [
                updated_stats["insertions"],
                updated_stats["deletions"],
                updated_stats["substitutions"],
            ]
        )
        updated_stats["WER"] = (
            100.0 * num_edits / updated_stats["num_ref_tokens"]
        )
    return updated_stats


def _batch_stats(refs, hyps):
    """
    Description:
        Internal function which actually computes the counts.
        Used by accumulatable_wer_stats
    Input:
        ref: (type: iterable of iterables) Batch of reference sequences
        hyp: (type: iterable of iterables) Batch of hypothesis sequences
    Output:
        stats: (type: collections.Counter) Edit statistics over the batch,
            with keys:
                "insertions" - number of insertions
                "deletions" - number of deletions
                "substitutions" - number of substitutions
                "num_ref_tokens" - number of reference tokens
    Example:
        from speechbrain.utils.edit_distance import _batch_stats
        batch = [[[1,2,3],[4,5,6]], [[1,2,4],[5,6]]]
        refs, hyps = batch
        print(_batch_stats(refs, hyps))
        ## Counter({'num_ref_tokens': 6, 'substitutions': 1, 'deletions': 1})
    Author:
        Aku Rouhe
    """
    if len(refs) != len(hyps):
        raise ValueError(
            "The reference and hypothesis batches are not of the same size"
        )
    stats = collections.Counter()
    for ref_tokens, hyp_tokens in zip(refs, hyps):
        table = op_table(ref_tokens, hyp_tokens)
        edits = count_ops(table)
        stats += edits
        stats["num_ref_tokens"] += len(ref_tokens)
    return stats


def op_table(a, b):
    """
    Description:
        Solves for the table of edit operations, which is mainly used to
        compute word error rate. The table is of size [|a|+1, |b|+1],
        and each point (i, j) in the table has an edit operation. The
        edit operations can be deterministically followed backwards to
        find the shortest edit path to from a[:i-1] to b[:j-1]. Indexes
        of zero (i=0 or j=0) correspond to an empty sequence.

        The algorithm itself is well known, see
            https://en.wikipedia.org/wiki/Levenshtein_distance
        Note that in some cases there are multiple valid edit operation
        paths which lead to the same edit distance minimum.
    Input:
        a and b: (type: any iterable) sequences between which the edit
            operations are solved for.
    Output:
        op_table: (type: list of lists, as matrix) Table of edit operations
    Example:
        from speechbrain.utils.edit_distance import op_table
        ref = [1,2,3]
        hyp = [1,2,4]
        print(op_table(ref, hyp))
        # [['I', 'I', 'I', 'I'],
        #  ['D', '=', 'I', 'I'],
        #  ['D', 'D', '=', 'I'],
        #  ['D', 'D', 'D', 'S']]
    Author:
        Aku Rouhe
    """
    # For the dynamic programming algorithm, only two rows are really needed:
    # the one currently being filled in, and the previous one
    # The following is also the right initialization
    prev_row = [j for j in range(len(b) + 1)]
    curr_row = [0] * (len(b) + 1)  # Just init to zero
    # For the edit operation table we will need the whole matrix.
    # We will initialize the table with no-ops, so that we only need to change
    # where an edit is made.
    op_table = [
        [EDIT_SYMBOLS["eq"] for j in range(len(b) + 1)]
        for i in range(len(a) + 1)
    ]
    # We already know the operations on the first row and column:
    for i in range(len(a) + 1):
        op_table[i][0] = EDIT_SYMBOLS["del"]
    for j in range(len(b) + 1):
        op_table[0][j] = EDIT_SYMBOLS["ins"]
    # The rest of the table is filled in row-wise:
    for i, a_token in enumerate(a, start=1):
        curr_row[0] += 1  # This trick just deals with the first column.
        for j, b_token in enumerate(b, start=1):
            # The dynamic programming algorithm cost rules
            insertion_cost = curr_row[j - 1] + 1
            deletion_cost = prev_row[j] + 1
            substitution = 0 if a_token == b_token else 1
            substitution_cost = prev_row[j - 1] + substitution
            # Here copying the Kaldi compute-wer comparison order, which in
            # ties prefers:
            # insertion > deletion > substitution
            if (
                substitution_cost < insertion_cost
                and substitution_cost < deletion_cost
            ):
                curr_row[j] = substitution_cost
                # Again, note that if not substitution, the edit table already
                # has the correct no-op symbol.
                if substitution:
                    op_table[i][j] = EDIT_SYMBOLS["sub"]
            elif deletion_cost < insertion_cost:
                curr_row[j] = deletion_cost
                op_table[i][j] = EDIT_SYMBOLS["del"]
            else:
                curr_row[j] = insertion_cost
                op_table[i][j] = EDIT_SYMBOLS["ins"]
        # Move to the next row:
        prev_row[:] = curr_row[:]
    return op_table


def alignment(op_table):
    """
    Description:
        Walks back an edit operations table, produced by calling
            op_table(a, b),
        and collects the edit distance alignment of a to b. The alignment
        shows which token in a corresponds to which token in b. Note that the
        alignment is monotonic, one-to-zero-or-one.
    Input:
        op_table: (type: list of lists) Edit operations table from
            op_table(a, b)
    Output:
        alignment (type: [(str <edit-op>, int-or-None <i>, int-or-None <j>),])
            List of edit operations, and the corresponding indices to a and b.
            See the EDIT_SYMBOLS dict for the edit-ops.
            i indexes a, j indexes b, and the indices can be None, which means
            aligning to nothing
    Example:
        from speechbrain.utils.edit_distance import alignment
        # table for a=[1,2,3], b=[1,2,4]
        table = [['I', 'I', 'I', 'I'],
                 ['D', '=', 'I', 'I'],
                 ['D', 'D', '=', 'I'],
                 ['D', 'D', 'D', 'S']]
        print(alignment(table))
        # [('=', 0, 0), ('=', 1, 1), ('S', 2, 2)]
    Author:
        Aku Rouhe
    """
    # The alignment will be the size of the longer sequence.
    # form: [(op, a_index, b_index)], index is None when aligned to empty
    alignment = []
    # Now we'll walk back the op_table to get the alignment.
    i = len(op_table) - 1
    j = len(op_table[0]) - 1
    while not (i == 0 and j == 0):
        if i == 0:
            j -= 1
            alignment.insert(0, (EDIT_SYMBOLS["ins"], None, j))
        elif j == 0:
            i -= 1
            alignment.insert(0, (EDIT_SYMBOLS["del"], i, None))
        else:
            if op_table[i][j] == EDIT_SYMBOLS["ins"]:
                j -= 1
                alignment.insert(0, (EDIT_SYMBOLS["ins"], None, j))
            elif op_table[i][j] == EDIT_SYMBOLS["del"]:
                i -= 1
                alignment.insert(0, (EDIT_SYMBOLS["del"], i, None))
            elif op_table[i][j] == EDIT_SYMBOLS["sub"]:
                i -= 1
                j -= 1
                alignment.insert(0, (EDIT_SYMBOLS["sub"], i, j))
            else:
                i -= 1
                j -= 1
                alignment.insert(0, (EDIT_SYMBOLS["eq"], i, j))
    return alignment


def count_ops(op_table):
    """
    Description:
        Walks back an edit operations table produced by op_table(a, b) and
        counts the number of insertions, deletions, and substitutions in the
        shortest edit path. This information is typically used in speech
        recognition to report the number of different error types separately.
    Input:
        op_table: (type: list of lists) Edit operations table from
            op_table(a, b)
    Output:
        edits: (type: collections.Counter) The counts of the edit operations,
            with keys:
                "insertions"
                "deletions"
                "substitutions"
            NOTE: not all of the keys might appear explicitly in the output,
                but for the missing keys collections.Counter will return 0
    Example:
        from speechbrain.utils.edit_distance import count_ops
        table = [['I', 'I', 'I', 'I'],
                 ['D', '=', 'I', 'I'],
                 ['D', 'D', '=', 'I'],
                 ['D', 'D', 'D', 'S']]
        print(count_ops(table))
        # Counter({'substitutions': 1})
    Author:
        Aku Rouhe
    """
    edits = collections.Counter()
    # Walk back the table, gather the ops.
    i = len(op_table) - 1
    j = len(op_table[0]) - 1
    while not (i == 0 and j == 0):
        if i == 0:
            ins += 1
            j -= 1
        elif i == 0:
            dels += 1
            j -= 1
        else:
            if op_table[i][j] == EDIT_SYMBOLS["ins"]:
                edits["insertions"] += 1
                j -= 1
            elif op_table[i][j] == EDIT_SYMBOLS["del"]:
                edits["deletions"] += 1
                i -= 1
            else:
                if op_table[i][j] == EDIT_SYMBOLS["sub"]:
                    edits["substitutions"] += 1
                i -= 1
                j -= 1
    return edits
