#!/usr/bin/python
"""
This recipe implements diarization baseline
using deep embedding extraction followed by spectral clustering.
We use cosine similarity based affinity matrix.

Condition: Oracle VAD and Oracle number of speakers.

Note: There are multiple ways to write this recipe. We chose to iterate over individual files.
This method is less GPU memory demanding and also makes code easy to understand.
"""

import os
import sys
import torch
import logging
import speechbrain as sb
import numpy as np
import pickle
import csv
import glob
import shutil
import scipy
from tqdm.contrib import tqdm

from speechbrain.utils.data_utils import download_file
from speechbrain.data_io.data_io import DataLoaderFactory
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.utils.DER import DER

np.random.seed(1234)

# Logger setup
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from ami_prepare import prepare_ami  # noqa E402

try:
    import sklearn
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


def compute_embeddings(wavs, lens, init_params=False):
    """Definition of the steps for computation of embeddings from the waveforms
    """
    with torch.no_grad():
        wavs = wavs.to(params.device)
        feats = params.compute_features(wavs, init_params=init_params)
        feats = params.mean_var_norm(feats, lens)
        emb = params.embedding_model(feats, lens, init_params=init_params)
        emb = params.mean_var_norm_emb(
            emb, torch.ones(emb.shape[0]).to("cuda:0")
        )

    return emb


def download_and_pretrain():
    """Downloads pre-trained model
    """
    save_model_path = params.model_dir + "/emb.ckpt"
    download_file(params.embedding_file, save_model_path)
    params.embedding_model.load_state_dict(
        torch.load(save_model_path), strict=True
    )


def embedding_computation_loop(split, set_loader, stat_file):
    """Extracts embeddings for a given dataset loader
    """

    # Extract embeddings (skip if already done)
    if not os.path.isfile(stat_file):

        embeddings = np.empty(shape=[0, params.emb_dim], dtype=np.float64)
        modelset = []
        segset = []
        with tqdm(set_loader, dynamic_ncols=True) as t:
            # different data may have different statistics
            params.mean_var_norm_emb.count = 0

            for wav in t:
                ids, wavs, lens = wav[0]

                mod = [x for x in ids]
                seg = [x for x in ids]
                modelset = modelset + mod
                segset = segset + seg

                # embedding computation
                emb = compute_embeddings(wavs, lens).squeeze().cpu().np()
                embeddings = np.concatenate((embeddings, emb), axis=0)

        modelset = np.array(modelset, dtype="|O")
        segset = np.array(segset, dtype="|O")

        # Intialize variables for start, stop and stat0
        s = np.array([None] * embeddings.shape[0])
        b = np.array([[1.0]] * embeddings.shape[0])

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


def getLamdaGaplist(lambdas):
    lambda_gap_list = []
    for i in range(len(lambdas) - 1):
        lambda_gap_list.append(float(lambdas[i + 1]) - float(lambdas[i]))
    return lambda_gap_list


def get_gaps_n_num_spk(Mat, spk_max):

    lambdas, eig_vecs = scipy.sparse.linalg.eigs(Mat, spk_max)
    lambdas = np.sort(lambdas)

    lambda_gap_list = getLamdaGaplist(lambdas)
    num_of_spk = (
        np.argmax(lambda_gap_list[: min(spk_max, len(lambda_gap_list))]) + 1
    )

    return lambda_gap_list, num_of_spk


class Standard_SC:
    def __init__(self):
        self.max_num_spkrs = 10

    def do_sc(self, X, k, PVAL):

        sim_mat = self.get_sim_mat(X)

        p_val = PVAL  # params.p_val
        # Requires PVAL
        prunned_sim_mat = self.p_pruning(sim_mat, p_val)

        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)
        lap = self.get_laplacian(sym_prund_sim_mat)

        # Requires n_spkrs (or estimates it)
        emb, num_of_spk = self.get_spec_emb(lap, k)

        self.cluster_me(emb, num_of_spk)

    def get_sim_mat(self, X):
        # Cosine similarities
        M = sklearn.metrics.pairwise.cosine_similarity(X, X)
        return M

    def p_pruning(self, A, pval):

        n_elems = int((1 - pval) * A.shape[0])

        for i in range(A.shape[0]):
            # For each row in a affinity matrix
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]
            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0

        return A

    def get_laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0
        A = M
        D = np.sum(np.abs(A), axis=1)
        D = np.diag(D)
        L = D - A
        return L

    def get_spec_emb(self, L, k_oracle=4):
        lambdas, eig_vecs = scipy.linalg.eigh(L)

        if params.oracle_n_spkrs is True:
            num_of_spk = k_oracle
        else:
            # Ignore this else part. It will be update in another PR
            # TODO: Some issues with max eigen gap. Fix this later
            print("\nlambdas = ", lambdas[0:8])

            # Ignore the first eigen value
            lambda_gap_list = getLamdaGaplist(lambdas[1:10])
            print("gaps= ", np.around(lambda_gap_list, 3))
            spk_max = 10
            num_of_spk = (
                np.argmax(lambda_gap_list[: min(spk_max, len(lambda_gap_list))])
                + 1
            )
            max_gap = max(  # noqa F841
                lambda_gap_list[: min(spk_max, len(lambda_gap_list))]
            )

            # k =  num_of_spk
            # Sort
            # indx = np.argsort(lambdas)
            # indx = indx[0:k]
            # emb = eig_vecs[:,indx]

        emb = eig_vecs[:, 0:num_of_spk]

        return emb, num_of_spk

    def cluster_me(self, emb, k):
        _, self.labels_, _ = k_means(emb, k)


def do_spec_clustering(diary_obj_eval, out_rttm_file, rec_id, k=4, PVAL=0.01):
    """Performs spectral clustering on embeddings
    """

    clust_obj = Standard_SC()
    clust_obj.do_sc(diary_obj_eval.stat1, k, PVAL)

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


def diarize_dataset(full_csv, split_type, p_val):
    """Diarizes all the recordings in a given dataset
    """

    # Prepare `spkr_info` only once when Oracle num of speakers is selected
    if params.oracle_n_spkrs is True:
        full_ref_rttm_file = (
            params.ref_rttm_dir + "/fullref_ami_" + split_type + ".rttm"
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
    init_params = True

    for rec_id in all_rec_ids:

        tag = "[" + str(split_type) + ": " + str(i) + "/" + N + "]"
        i = i + 1

        msg = "Diarizing %s : %s " % (tag, rec_id)
        logger.info(msg)

        if not os.path.exists(os.path.join(params.embedding_dir, split)):
            os.makedirs(os.path.join(params.embedding_dir, split))

        diary_stat_file = os.path.join(
            params.embedding_dir, split, rec_id + "_xv_stat.pkl"
        )

        # Prepare a csv for a recording
        new_csv_file = os.path.join(
            params.embedding_dir, split, rec_id + ".csv"
        )
        prepare_subset_csv(full_csv, rec_id, new_csv_file)

        # Setup a dataloader for above one recording (above csv)
        diary_set = DataLoaderFactory(
            new_csv_file,
            params.diary_loader_eval.batch_size,
            params.diary_loader_eval.csv_read,
            params.diary_loader_eval.sentence_sorting,
        )

        diary_set_loader = diary_set.forward()

        # Dir to store embeddings
        if not os.path.exists(os.path.join(params.embedding_dir, split)):
            os.makedirs(os.path.join(params.embedding_dir, split))

        if init_params:
            _, wavs, lens = next(iter(diary_set_loader))[0]

            # Initialize the model and perform pre-training
            _ = compute_embeddings(wavs, lens, init_params=True)

            # Download models from the web if needed
            if "https://" in params.embedding_file:
                download_and_pretrain()
            else:
                params.embedding_model.load_state_dict(
                    torch.load(params.embedding_file), strict=True
                )

            init_params = False
            params.embedding_model.eval()

        # Compute Embeddings
        diary_obj_dev = embedding_computation_loop(
            "diary", diary_set_loader, diary_stat_file
        )

        # Perform spectral clustering
        out_rttm_dir = os.path.join(params.sys_rttm_dir, split)
        if not os.path.exists(out_rttm_dir):
            os.makedirs(out_rttm_dir)
        out_rttm_file = out_rttm_dir + "/" + rec_id + ".rttm"

        if params.oracle_n_spkrs is True:
            # Oracle num of speakers
            PVAL = p_val
            num_spkrs = get_oracle_num_spkrs(rec_id, spkr_info)
        else:
            PVAL = p_val
            num_spkrs = None  # Will be estimated later

        # Note this is for ONE recording
        do_spec_clustering(
            diary_obj_dev, out_rttm_file, rec_id, num_spkrs, PVAL
        )

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


def dev_p_tuner(full_csv, split_type):
    """Tuning n_compenents on dev set. (Basic tunning).
    Returns:
        n_lambdas = n_components
    """

    DER_list = []
    prange = [0.0025, 0.0050, 0.0075, 0.010, 0.025, 0.050, 0.075, 0.100]

    for p_v in prange:
        # Process whole dataset for value of n_lambdas
        concate_rttm_file = diarize_dataset(full_csv, split_type, p_v)

        ref_rttm = os.path.join(params.ref_rttm_dir, "fullref_ami_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm, sys_rttm, params.ignore_overlap, params.forgiveness_collar
        )

        msg = "\n[Tuner]: p_val= %f , DER= %s\n" % (p_v, str(round(DER_, 2)),)

        logger.info(msg)
        DER_list.append(DER_)

    # Take p_val that gave minmum DER on Dev dataset
    tuned_p_val = prange[DER_list.index(min(DER_list))]

    # return tuned_n_lambdas
    return tuned_p_val


# Begin!
if __name__ == "__main__":  # noqa: C901

    # Load hyperparameters file with command-line overrides
    params_file, overrides = sb.core.parse_arguments(sys.argv[1:])

    with open(params_file) as fin:
        params = sb.yaml.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params.output_folder,
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Few more experiment directories (to have cleaner structure)
    exp_dirs = [
        params.model_dir,
        params.embedding_dir,
        params.csv_dir,
        params.ref_rttm_dir,
        params.sys_rttm_dir,
    ]
    for dir_ in exp_dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    # Prepare data for AMI
    logger.info(
        "AMI: Data preparation [Prepares both, the reference RTTMs and the CSVs]"
    )
    prepare_ami(
        data_folder=params.data_folder,
        manual_annot_folder=params.manual_annot_folder,
        save_folder=params.save_folder,
        split_type=params.split_type,
        skip_TNO=params.skip_TNO,
        mic_type=params.mic_type,
        vad_type=params.vad_type,
        max_subseg_dur=params.max_subseg_dur,
        overlap=params.overlap,
    )

    # AMI Dev Set
    full_csv = []
    with open(params.csv_diary_dev, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            full_csv.append(row)

    # Tuning p_val for affinity matrix on dev data
    best_p_val = dev_p_tuner(full_csv, "dev")

    # Following is not needed but lets keep this as results dir will have final best RTTMs
    out_boundaries = diarize_dataset(full_csv, "dev", best_p_val)

    # Evaluating on DEV set
    logger.info("Evaluating for AMI Dev. set")
    ref_rttm = os.path.join(params.ref_rttm_dir, "fullref_ami_dev.rttm")
    sys_rttm = out_boundaries
    [MS_dev, FA_dev, SER_dev, DER_dev] = DER(
        ref_rttm, sys_rttm, params.ignore_overlap, params.forgiveness_collar
    )
    msg = "AMI Dev set: Diarization Error Rate = %s %%\n" % (
        str(round(DER_dev, 2))
    )
    logger.info(msg)

    # AMI Eval Set
    full_csv = []
    with open(params.csv_diary_eval, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            full_csv.append(row)

    out_boundaries = diarize_dataset(full_csv, "eval", best_p_val)

    # Evaluating on EVAL set
    logger.info("Evaluating for AMI Eval. set")
    ref_rttm = os.path.join(params.ref_rttm_dir, "fullref_ami_eval.rttm")
    sys_rttm = out_boundaries
    [MS_eval, FA_eval, SER_eval, DER_eval] = DER(
        ref_rttm, sys_rttm, params.ignore_overlap, params.forgiveness_collar
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
