#!/usr/bin/python
"""
This recipe implements diarization baseline
using deep embedding extraction followed by spectral clustering.
We use nearest-neighbor based affinity matrix.

Condition: Oracle VAD and Oracle number of speakers.

Note: There are multiple ways to write this recipe. We chose to iterate over individual files.
This method is less GPU memory demanding and also makes code easy to understand.
"""

import os
import sys
import torch
import logging
import speechbrain as sb
import numpy
import numpy as np
import pickle
import csv
import glob
import shutil
import numbers
import warnings
import time
from tqdm.contrib import tqdm

from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

from speechbrain.utils.data_utils import download_file
from speechbrain.data_io.data_io import DataLoaderFactory
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.utils.DER import DER

# Logger setup
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from ami_prepare import prepare_ami  # noqa E402

try:
    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster._kmeans import k_means
except ImportError:
    err_msg = "The optional dependency sklearn is used in this module\n"
    err_msg += "Cannot import sklearn. \n"
    err_msg += "Please follow the below instructions\n"
    err_msg += "=============================\n"
    err_msg += "Using pip:\n"
    err_msg += "pip install sklearn\n"
    err_msg += "================================ \n"
    err_msg += "Using conda:\n"
    err_msg += "conda install sklearn"
    raise ImportError(err_msg)


def compute_embeddings(wavs, lens):
    """Definition of the steps for computation of embeddings from the waveforms
    """
    with torch.no_grad():
        wavs = wavs.to(params["device"])
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, lens)
        emb = params["embedding_model"](feats, lens=lens)
        emb = params["mean_var_norm_emb"](
            emb, torch.ones(emb.shape[0], device=params["device"])
        )

    return emb


def download_and_pretrain():
    """Downloads pre-trained model
    """
    save_model_path = params["model_dir"] + "/emb.ckpt"
    download_file(params["embedding_file"], save_model_path)
    params["embedding_model"].load_state_dict(
        torch.load(save_model_path), strict=True
    )


def embedding_computation_loop(split, set_loader, stat_file):
    """Extracts embeddings for a given dataset loader
    """

    # Extract embeddings (skip if already done)
    if not os.path.isfile(stat_file):

        embeddings = numpy.empty(
            shape=[0, params["emb_dim"]], dtype=numpy.float64
        )
        modelset = []
        segset = []
        with tqdm(set_loader, dynamic_ncols=True) as t:
            # different data may have different statistics
            params["mean_var_norm_emb"].count = 0

            for wav in t:
                ids, wavs, lens = wav[0]

                mod = [x for x in ids]
                seg = [x for x in ids]
                modelset = modelset + mod
                segset = segset + seg

                # embedding computation
                emb = compute_embeddings(wavs, lens).squeeze(1).cpu().numpy()
                embeddings = numpy.concatenate((embeddings, emb), axis=0)

        modelset = numpy.array(modelset, dtype="|O")
        segset = numpy.array(segset, dtype="|O")

        # Intialize variables for start, stop and stat0
        s = numpy.array([None] * embeddings.shape[0])
        b = numpy.array([[1.0]] * embeddings.shape[0])

        stat_obj = StatObject_SB(
            modelset=modelset,
            segset=segset,
            start=s,
            stop=s,
            stat0=b,
            stat1=embeddings,
        )
        logger.info(f"Saving Embeddings...")
        stat_obj.save_stat_object(stat_file)

    else:
        logger.info(f"Skipping embedding extraction (as already present)")
        logger.info(f"Loading previously saved embeddings")

        with open(stat_file, "rb") as in_file:
            stat_obj = pickle.load(in_file)

    return stat_obj


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
        for l in out_csv:
            csv_writer.writerow(l)

    msg = "Prepared CSV file: " + out_csv_file
    logger.info(msg)


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

    logger.info("Output RTTM saved at: " + out_rttm_file)


################################################


def _graph_connected_component(graph, node_id):
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


def _graph_is_connected(graph):
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
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def _set_diag(laplacian, value, norm_laplacian):
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


def spectral_embedding_sb(
    adjacency, n_components=8, norm_laplacian=True, drop_first=True,
):

    # random_state = check_random_state(random_state)

    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    if not _graph_is_connected(adjacency):
        warnings.warn(
            "Graph is not fully connected, spectral embedding"
            " may not work as expected."
        )

    laplacian, dd = csgraph_laplacian(
        adjacency, normed=norm_laplacian, return_diag=True
    )

    laplacian = _set_diag(laplacian, 1, norm_laplacian)

    laplacian *= -1
    # v0 = random_state.uniform(-1, 1, laplacian.shape[0])

    # vals, diffusion_map = eigsh(
    #    laplacian, k=n_components, sigma=1.0, which="LM", tol=eigen_tol, v0=v0
    # )

    vals, diffusion_map = eigsh(
        laplacian, k=n_components, sigma=1.0, which="LM",
    )

    embedding = diffusion_map.T[n_components::-1]

    if norm_laplacian:
        embedding = embedding / dd

    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T


def _deterministic_vector_sign_flip(u):
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
        "%r cannot be used to seed a numpy.random.RandomState"
        " instance" % seed
    )


def spectral_clustering_sb(
    affinity,
    n_clusters=8,
    n_components=None,
    eigen_solver=None,
    random_state=None,
    n_init=10,
    eigen_tol=0.0,
    assign_labels="kmeans",
):

    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    maps = spectral_embedding_sb(
        affinity, n_components=n_components, drop_first=False,
    )

    _, labels, _ = k_means(
        maps, n_clusters, random_state=random_state, n_init=n_init
    )

    return labels


def do_spec_clustering(diary_obj_eval, out_rttm_file, rec_id, k=4):
    """Performs spectral clustering on embeddings
    """
    clust_obj = Spec_Cluster(
        n_clusters=k,
        assign_labels="kmeans",
        random_state=params["seed"],
        affinity="nearest_neighbors",
    )

    clust_obj.perform_sc(diary_obj_eval.stat1)

    labels = clust_obj.labels_

    # Convert labels to speaker boundaries
    subseg_ids = diary_obj_eval.segset
    lol = []

    for i in range(labels.shape[0]):
        spkr_id = rec_id + "_" + str(labels[i])

        sub_seg = subseg_ids[i]

        splitted = sub_seg.rsplit("_", 2)
        rec_id = str(splitted[0])
        sseg_start = float(splitted[1])
        sseg_end = float(splitted[2])

        a = [rec_id, sseg_start, sseg_end, spkr_id]
        lol.append(a)

    # Sorting based on start time of sub-segment
    lol.sort(key=lambda x: float(x[1]))

    # Merge and split in 2 simple steps: (i) Merge sseg of same speakers then (ii) split different speakers
    # Step 1: Merge adjacent sub-segments that belong to same speaker (or cluster)
    lol = merge_ssegs_same_speaker(lol)

    # Step 2: Distribute duration of adjacent overlapping sub-segments belonging to different speakers (or cluster)
    # Taking mid-point as the splitting time location.
    lol = distribute_overlap(lol)

    logger.info("Completed diarizing " + rec_id)
    write_rttm(lol, out_rttm_file)


def get_oracle_num_spkrs(rec_id, spkr_info):
    """Returns actual number of speakers in a recording
    """

    num_spkrs = 0
    for line in spkr_info:
        if rec_id in line:
            # Since rec_id is prefix for each speaker
            num_spkrs += 1

    return num_spkrs


class Spec_Cluster(SpectralClustering):
    def perform_sc(self, X):
        """Performs spectral clustering using sklearn on embeddings (X).

        Arguments
        ---------
        X : array (n_samples, n_features)
            Embeddings to be clustered
        """

        # Computation of affinity matrix
        connectivity = kneighbors_graph(
            X,
            n_neighbors=params["n_neighbors"],
            include_self=params["include_self"],
        )
        self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)

        # Perform spectral clustering on affinity matrix
        self.labels_ = spectral_clustering_sb(
            self.affinity_matrix_,
            n_clusters=self.n_clusters,
            assign_labels=self.assign_labels,
        )
        return self


def diarize_dataset(full_csv, split_type, n_lambdas):
    """Diarizes all the recordings in a given dataset
    """

    # Prepare `spkr_info` only once when Oracle num of speakers is selected
    if params["oracle_n_spkrs"] is True:
        full_ref_rttm_file = (
            params["ref_rttm_dir"] + "/fullref_ami_" + split_type + ".rttm"
        )
        RTTM = []
        with open(full_ref_rttm_file, "r") as f:
            for line in f:
                entry = line[:-1]
                RTTM.append(entry)

        spkr_info = list(  # noqa F841
            filter(lambda x: x.startswith("SPKR-INFO"), RTTM)
        )

    # Get all recording IDs in this dataset
    A = [row[0].rstrip().split("_")[0] for row in full_csv]
    all_rec_ids = list(set(A[1:]))
    all_rec_ids.sort()

    N = str(len(all_rec_ids))
    split = "AMI_" + split_type
    i = 1

    # Pretrain model
    if "https://" in params["embedding_file"]:
        download_and_pretrain()
    else:
        params["embedding_model"].load_state_dict(
            torch.load(params["embedding_file"]), strict=True
        )

    # Setting eval modality
    params["embedding_model"].eval()

    for rec_id in all_rec_ids:

        tag = "[" + str(split_type) + ": " + str(i) + "/" + N + "]"
        i = i + 1

        msg = "Diarizing %s : %s " % (tag, rec_id)
        logger.info(msg)

        if not os.path.exists(os.path.join(params["embedding_dir"], split)):
            os.makedirs(os.path.join(params["embedding_dir"], split))

        diary_stat_file = os.path.join(
            params["embedding_dir"], split, rec_id + "_xv_stat.pkl"
        )

        # Prepare a csv for a recording
        new_csv_file = os.path.join(
            params["embedding_dir"], split, rec_id + ".csv"
        )
        prepare_subset_csv(full_csv, rec_id, new_csv_file)

        # Setup a dataloader for above one recording (above csv)
        diary_set = DataLoaderFactory(
            new_csv_file,
            params["diary_loader_eval"].batch_size,
            params["diary_loader_eval"].csv_read,
            params["diary_loader_eval"].sentence_sorting,
        )

        diary_set_loader = diary_set.forward().get_dataloader()

        # Putting modules on the device
        params["compute_features"].to(params["device"])
        params["mean_var_norm"].to(params["device"])
        params["embedding_model"].to(params["device"])
        params["mean_var_norm_emb"].to(params["device"])

        # Compute Embeddings
        diary_obj_dev = embedding_computation_loop(
            "diary", diary_set_loader, diary_stat_file
        )

        # Perform spectral clustering
        out_rttm_dir = os.path.join(params["sys_rttm_dir"], split)
        if not os.path.exists(out_rttm_dir):
            os.makedirs(out_rttm_dir)
        out_rttm_file = out_rttm_dir + "/" + rec_id + ".rttm"

        if params["oracle_n_spkrs"] is True:
            # Oracle num of speakers
            num_spkrs = get_oracle_num_spkrs(rec_id, spkr_info)
        else:
            # Num of speakers tunned on dev set
            num_spkrs = n_lambdas

        do_spec_clustering(diary_obj_dev, out_rttm_file, rec_id, k=num_spkrs)

    # Concatenate individual RTTM files
    # This is not needed but just staying with the standards
    concate_rttm_file = out_rttm_dir + "/sys_output.rttm"

    # logger.info("Concatenating individual RTTM files...")
    with open(concate_rttm_file, "w") as cat_file:
        for f in glob.glob(out_rttm_dir + "/*.rttm"):
            if f == concate_rttm_file:
                continue
            with open(f, "r") as indi_rttm_file:
                shutil.copyfileobj(indi_rttm_file, cat_file)

    msg = "The system generated RTTM file for %s set : %s" % (
        split_type,
        concate_rttm_file,
    )
    logger.info(msg)

    return concate_rttm_file


def dev_tuner(full_csv, split_type):
    """Tuning n_compenents on dev set. (Basic tunning).
    Returns:
        n_lambdas = n_components
    """

    DER_list = []
    for n_lambdas in range(1, params["max_num_spkrs"] + 1):

        # Process whole dataset for value of n_lambdas
        concate_rttm_file = diarize_dataset(full_csv, split_type, n_lambdas)

        ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_ami_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
        )

        msg = "[Tuner]: n_lambdas= %d , DER= %s\n" % (
            n_lambdas,
            str(round(DER_, 2)),
        )

        logger.info(msg)
        DER_list.append(DER_)

    # Take n_lambdas with minmum DER
    tuned_n_lambdas = DER_list.index(min(DER_list)) + 1

    return tuned_n_lambdas


# Begin!
if __name__ == "__main__":  # noqa: C901

    # Load hyperparameters file with command-line overrides
    params_file, overrides = sb.core.parse_arguments(sys.argv[1:])

    with open(params_file) as fin:
        params = sb.yaml.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Few more experiment directories (to have cleaner structure)
    exp_dirs = [
        params["model_dir"],
        params["embedding_dir"],
        params["csv_dir"],
        params["ref_rttm_dir"],
        params["sys_rttm_dir"],
    ]
    for dir_ in exp_dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    # Prepare data for AMI
    logger.info(
        "AMI: Data preparation [Prepares both, the reference RTTMs and the CSVs]"
    )
    prepare_ami(
        data_folder=params["data_folder"],
        manual_annot_folder=params["manual_annot_folder"],
        save_folder=params["save_folder"],
        split_type=params["split_type"],
        skip_TNO=params["skip_TNO"],
        mic_type=params["mic_type"],
        vad_type=params["vad_type"],
        max_subseg_dur=params["max_subseg_dur"],
        overlap=params["overlap"],
    )

    # AMI Dev Set
    full_csv = []
    with open(params["csv_diary_dev"], "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            full_csv.append(row)

    # TUNING for num of lambdas
    if params["oracle_n_spkrs"] is False:
        a = time.time()
        n_lambdas = dev_tuner(full_csv, "dev")
        msg = "Tuning completed! Total time spent in tuning = %s seconds\n" % (
            str(round(time.time() - a, 2))
        )
        logger.info(msg)
    else:
        msg = "Running for Oracle number of speakers"
        logger.info(msg)
        n_lambdas = None  # will be taken from groundtruth

    out_boundaries = diarize_dataset(full_csv, "dev", n_lambdas=n_lambdas)

    # Evaluating on DEV set
    logger.info("Evaluating for AMI Dev. set")
    ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_ami_dev.rttm")
    sys_rttm = out_boundaries
    [MS_dev, FA_dev, SER_dev, DER_dev] = DER(
        ref_rttm,
        sys_rttm,
        params["ignore_overlap"],
        params["forgiveness_collar"],
    )
    msg = "AMI Dev set: Diarization Error Rate = %s %%\n" % (
        str(round(DER_dev, 2))
    )
    logger.info(msg)

    # AMI Eval Set
    full_csv = []
    with open(params["csv_diary_eval"], "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            full_csv.append(row)

    out_boundaries = diarize_dataset(full_csv, "eval", n_lambdas=n_lambdas)

    # Evaluating on EVAL set
    logger.info("Evaluating for AMI Eval. set")
    ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_ami_eval.rttm")
    sys_rttm = out_boundaries
    [MS_eval, FA_eval, SER_eval, DER_eval] = DER(
        ref_rttm,
        sys_rttm,
        params["ignore_overlap"],
        params["forgiveness_collar"],
    )
    msg = "AMI Eval set: Diarization Error Rate = %s %%\n" % (
        str(round(DER_eval, 2))
    )
    logger.info(msg)

    msg = (
        "Final Diarization Error Rate (%%) on AMI corpus: Dev = %s %% | Eval = %s %%\n"
        % (str(round(DER_dev, 2)), str(round(DER_eval, 2)))
    )
    logger.info(msg)
