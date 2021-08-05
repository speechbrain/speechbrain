import k2

def nbest_decoding(lats: k2.Fsa, num_paths: int):
    '''
    (Ideas of this function are from Dan)

    It implements something like CTC prefix beam search using n-best lists

    The basic idea is to first extra n-best paths from the given lattice,
    build a word seqs from these paths, and compute the total scores
    of these sequences in the log-semiring. The one with the max score
    is used as the decoding output.
    '''

    # First, extract `num_paths` paths for each sequence.
    # paths is a k2.RaggedInt with axes [seq][path][arc_pos]
    paths = k2.random_paths(lats, num_paths=num_paths, use_double_scores=True)

    # word_seqs is a k2.RaggedInt sharing the same shape as `paths`
    # but it contains word IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.

    word_seqs = k2.index(lats.aux_labels, paths)
    # Note: the above operation supports also the case when
    # lats.aux_labels is a ragged tensor. In that case,
    # `remove_axis=True` is used inside the pybind11 binding code,
    # so the resulting `word_seqs` still has 3 axes, like `paths`.
    # The 3 axes are [seq][path][word]

    # Remove epsilons and -1 from word_seqs
    word_seqs = k2.ragged.remove_values_leq(word_seqs, 0)

    # Remove repeated sequences to avoid redundant computation later.
    #
    # Since k2.ragged.unique_sequences will reorder paths within a seq,
    # `new2old` is a 1-D torch.Tensor mapping from the output path index
    # to the input path index.
    # new2old.numel() == unique_word_seqs.num_elements()
    unique_word_seqs, _, new2old = k2.ragged.unique_sequences(
        word_seqs, need_num_repeats=False, need_new2old_indexes=True)
    # Note: unique_word_seqs still has the same axes as word_seqs

    seq_to_path_shape = k2.ragged.get_layer(unique_word_seqs.shape(), 0)

    # path_to_seq_map is a 1-D torch.Tensor.
    # path_to_seq_map[i] is the seq to which the i-th path
    # belongs.
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    # Remove the seq axis.
    # Now unique_word_seqs has only two axes [path][word]
    unique_word_seqs = k2.ragged.remove_axis(unique_word_seqs, 0)

    # word_fsas is an FsaVec with axes [path][state][arc]
    word_fsas = k2.linear_fsa(unique_word_seqs)

    word_fsas_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsas)

    # lats has phone IDs as labels and word IDs as aux_labels.
    # inv_lats has word IDs as labels and phone IDs as aux_labels
    inv_lats = k2.invert(lats)
    inv_lats = k2.arc_sort(inv_lats) # no-op if inv_lats is already arc-sorted

    path_lats = k2.intersect_device(inv_lats,
                                    word_fsas_with_epsilon_loops,
                                    b_to_a_map=path_to_seq_map,
                                    sorted_match_a=True)
    # path_lats has word IDs as labels and phone IDs as aux_labels

    path_lats = k2.top_sort(k2.connect(path_lats.to('cpu')).to(lats.device))

    tot_scores = path_lats.get_tot_scores(True, True)
    # RaggedFloat currently supports float32 only.
    # We may bind Ragged<double> as RaggedDouble if needed.
    ragged_tot_scores = k2.RaggedFloat(seq_to_path_shape,
                                       tot_scores.to(torch.float32))

    argmax_indexes = k2.ragged.argmax_per_sublist(ragged_tot_scores)

    # Since we invoked `k2.ragged.unique_sequences`, which reorders
    # the index from `paths`, we use `new2old`
    # here to convert argmax_indexes to the indexes into `paths`.
    #
    # Use k2.index here since argmax_indexes' dtype is torch.int32
    best_path_indexes = k2.index(new2old, argmax_indexes)

    paths_2axes = k2.ragged.remove_axis(paths, 0)

    # best_paths is a k2.RaggedInt with 2 axes [path][arc_pos]
    best_paths = k2.index(paths_2axes, best_path_indexes)

    # labels is a k2.RaggedInt with 2 axes [path][phone_id]
    # Note that it contains -1s.
    labels = k2.index(lats.labels.contiguous(), best_paths)

    labels = k2.ragged.remove_values_eq(labels, -1)

    # lats.aux_labels is a k2.RaggedInt tensor with 2 axes, so
    # aux_labels is also a k2.RaggedInt with 2 axes
    aux_labels = k2.index(lats.aux_labels, best_paths.values())

    best_path_fsas = k2.linear_fsa(labels)
    best_path_fsas.aux_labels = aux_labels

    return best_path_fsas