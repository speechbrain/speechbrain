import csv
import numbers
import numpy as np

from scipy import sparse
from scipy.sparse.csgraph import connected_components

np.random.seed(1234)


def prepare_subset_csv(full_diary_csv, rec_id, out_csv_file):
    """Prepares csv for a recording ID
    """
    out_csv_head = [full_diary_csv[0]]
    entry = []
    for row in full_diary_csv:
        if row[0].startswith(rec_id):
            entry.append(row)

    out_csv = out_csv_head + entry

    with open(out_csv_file, mode="w") as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for r in out_csv:
            csv_writer.writerow(r)

    # msg = "Prepared CSV file: " + out_csv_file
    # logger.info(msg)


def is_overlapped(end1, start2):
    """Returns True if segments are overlapping
    """
    if start2 > end1:
        return False
    else:
        return True


def merge_ssegs_same_speaker(lol):
    """Merge adjacent sub-segs from a same speaker.

    Arguments
    ---------
    lol : list of list
        Each list -  [rec_id, sseg_start, sseg_end, spkr_id]
    """
    new_lol = []

    # Start from the first sub-seg
    sseg = lol[0]

    for i in range(1, len(lol)):
        next_sseg = lol[i]

        # IF sub-segments overlap AND has same speaker THEN merge
        if is_overlapped(sseg[2], next_sseg[1]) and sseg[3] == next_sseg[3]:
            sseg[2] = next_sseg[2]  # just update the end time

        else:
            new_lol.append(sseg)
            sseg = next_sseg

    return new_lol


def distribute_overlap(lol):
    """Distributes the overlapped speech equally among the adjacent segments with different speakers.
    Input list of list: structure [rec_id, sseg_start, sseg_end, spkr_id]
    """
    new_lol = []
    sseg = lol[0]

    # Add first sub-segment here to avoid error at: "if new_lol[-1] != sseg:" when new_lol is empty
    # new_lol.append(sseg)

    for i in range(1, len(lol)):
        next_sseg = lol[i]
        # No need to check if they are different speakers
        # Because if segments are overlapped then they always have different speakers
        # This is because similar speaker's adjacent sub-segments are already merged by "merge_ssegs_same_speaker()"

        if is_overlapped(sseg[2], next_sseg[1]):

            # Get overlap duration
            # Now this overlap will be divided equally between adjacent segments
            overlap = sseg[2] - next_sseg[1]

            # Update end time of old seg
            sseg[2] = sseg[2] - (overlap / 2.0)

            # Update start time of next seg
            next_sseg[1] = next_sseg[1] + (overlap / 2.0)

            if len(new_lol) == 0:
                # For first sub-segment entry
                new_lol.append(sseg)
            else:
                # To avoid duplicate entries
                if new_lol[-1] != sseg:
                    new_lol.append(sseg)

            # Current sub-segment is next sub-segment
            sseg = next_sseg

        else:
            # For the first sseg
            if len(new_lol) == 0:
                new_lol.append(sseg)
            else:
                # To avoid duplicate entries
                if new_lol[-1] != sseg:
                    new_lol.append(sseg)

            # Update the current sub-segment
            sseg = next_sseg

    # Add the remaning last sub-segment
    new_lol.append(next_sseg)

    return new_lol


def write_rttm(segs_list, out_rttm_file):
    """Writes the segment list in RTTM format.
    """
    rttm = []
    rec_id = segs_list[0][0]

    for seg in segs_list:
        new_row = [
            "SPEAKER",
            rec_id,
            "0",
            str(round(seg[1], 4)),
            str(round(seg[2] - seg[1], 4)),
            "<NA>",
            "<NA>",
            seg[3],
            "<NA>",
            "<NA>",
        ]
        rttm.append(new_row)

    with open(out_rttm_file, "w") as f:
        for row in rttm:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)

    # logger.info("Output RTTM saved at: " + out_rttm_file)


#######################################


def graph_connected_component(graph, node_id):
    """Find the largest graph connected components that contains one
    given node
    Parameters
    ----------
    graph : array-like, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes
    node_id : int
        The index of the query node of the graph
    Returns
    -------
    connected_components_matrix : array-like, shape: (n_samples,)
        An array of bool value indicating the indexes of the nodes
        belonging to the largest connected components of the given query
        node
    """
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        # speed up row-wise access to boolean connection mask
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def graph_is_connected(graph):
    """ Return whether the graph is connected (True) or Not (False)
    Parameters
    ----------
    graph : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes
    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not
    """
    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        # Fix this later
        return (
            _graph_connected_component(graph, 0).sum()  # noqa F821
            == graph.shape[0]  # noqa F821
        )


def set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition
    Parameters
    ----------
    laplacian : array or sparse matrix
        The graph laplacian
    value : float
        The value of the diagonal
    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not
    Returns
    -------
    laplacian : array or sparse matrix
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[:: n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def deterministic_vector_sign_flip(u):
    """Modify the sign of vectors for reproducibility.
    Flips the sign of elements of all the vectors (rows of u) such that
    the absolute maximum element of each vector is positive.
    Parameters
    ----------
    u : ndarray
        Array with vectors as its rows.
    Returns
    -------
    u_flipped : ndarray with same shape as u
        Array with the sign flipped vectors as its rows.
    """
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a np.random.RandomState" " instance" % seed
    )
