#!/usr/bin/python
"""
This recipe implements diarization baseline.
Condition: Oracle VAD and Oracle number of speakers.

Note: There could be multiple ways to write this recipe. We chose to iterate over individual files.
This method is less GPU memory demanding and also makes code easy to understand.
"""

import os
import sys
import torch
import logging
import speechbrain as sb
import numpy
import pickle
import copy
import csv
import glob
import shutil
from tqdm.contrib import tqdm

from speechbrain.utils.DER import DER
from speechbrain.utils.data_utils import download_file
from speechbrain.data_io.data_io import DataLoaderFactory
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.processing.PLDA_LDA import LDA  # noqa F401

# Logger setup
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from voxceleb_prepare import prepare_voxceleb  # noqa E402
from ami_prepare import prepare_ami  # noqa E402
from ami_splits import get_AMI_split  # noqa E402


# Definition of the steps for xvector computation from the waveforms
def compute_x_vectors(wavs, lens, init_params=False):
    with torch.no_grad():
        wavs = wavs.to(params.device)
        feats = params.compute_features(wavs, init_params=init_params)
        feats = params.mean_var_norm(feats, lens)
        x_vect = params.xvector_model(feats, lens=lens, init_params=init_params)
        x_vect = params.mean_var_norm_xvect(
            x_vect, torch.ones(x_vect.shape[0]).to("cuda:0")
        )

    return x_vect


# Function for pre-trained model downloads
def download_and_pretrain():
    save_model_path = params.model_dir + "/xvect.ckpt"
    download_file(params.xvector_f, save_model_path)
    params.xvector_model.load_state_dict(
        torch.load(save_model_path), strict=True
    )


# Function to get mod and seg
def get_utt_ids_for_test(ids, data_dict):
    mod = [data_dict[x]["wav1"]["data"] for x in ids]
    seg = [data_dict[x]["wav2"]["data"] for x in ids]

    return mod, seg


# Computes xvectors for given split
def xvect_computation_loop(split, set_loader, stat_file):

    # Extract xvectors (skip if already done)
    if not os.path.isfile(stat_file):
        init_params = True

        xvectors = numpy.empty(shape=[0, params.xvect_dim], dtype=numpy.float64)
        modelset = []
        segset = []
        with tqdm(set_loader, dynamic_ncols=True) as t:

            for wav in t:
                ids, wavs, lens = wav[0]

                mod = [x for x in ids]
                seg = [x for x in ids]
                modelset = modelset + mod
                segset = segset + seg

                # Initialize the model and perform pre-training
                if init_params:
                    xvects = compute_x_vectors(wavs, lens, init_params=True)
                    params.mean_var_norm_xvect.glob_mean = torch.zeros_like(
                        xvects[0, 0, :]
                    )
                    params.mean_var_norm_xvect.count = 0

                    # Download models from the web if needed
                    if "https://" in params.xvector_f:
                        download_and_pretrain()
                    else:
                        params.xvector_model.load_state_dict(
                            torch.load(params.xvector_f), strict=True
                        )

                    init_params = False
                    params.xvector_model.eval()

                # xvector computation
                xvects = compute_x_vectors(wavs, lens)
                xv = xvects.squeeze().cpu().numpy()
                xvectors = numpy.concatenate((xvectors, xv), axis=0)

        modelset = numpy.array(modelset, dtype="|O")
        segset = numpy.array(segset, dtype="|O")

        # Intialize variables for start, stop and stat0
        s = numpy.array([None] * xvectors.shape[0])
        b = numpy.array([[1.0]] * xvectors.shape[0])

        stat_obj = StatObject_SB(
            modelset=modelset,
            segset=segset,
            start=s,
            stop=s,
            stat0=b,
            stat1=xvectors,
        )
        logger.info(f"Saving Embeddings...")
        stat_obj.save_stat_object(stat_file)

    else:
        logger.info(f"Skipping embedding extraction (as already present)")
        logger.info(f"Loading previously saved embeddings")

        with open(stat_file, "rb") as in_file:
            stat_obj = pickle.load(in_file)

    return stat_obj


def prepare_subset_csv(full_diary_csv, rec_id, out_dir):
    out_csv_head = [full_diary_csv[0]]
    entry = []
    for row in full_diary_csv:
        if row[0].startswith(rec_id):
            entry.append(row)

    out_csv = out_csv_head + entry

    out_csv_file = out_dir + "/" + rec_id + ".csv"
    with open(out_csv_file, mode="w") as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for l in out_csv:
            csv_writer.writerow(l)

    msg = "Prepared CSV file: " + out_csv_file
    logger.info(msg)


def trace_back_cluster_labels(
    cluster_map_traverse, threshold=0.9, oracle_num_spkrs=4
):

    N = len(cluster_map_traverse) + 1

    # Cluster dictionery for maintaining cluster IDs and segments inside each cluster ID
    # example: clusters['10'] : [1,4,7] denotes cluster number 10 has segment ids 1, 4 and 7
    clusters = dict()

    # Initialize clusters (cluster IDs starts from 0 )
    for i in range(N):
        clusters[str(i)] = [i]

    i = 0
    while i < N - 1:

        a = str(int(cluster_map_traverse[i, 0]))
        b = str(int(cluster_map_traverse[i, 1]))

        # dist = cluster_map_traverse[i, 2]
        new_id = str(N + i)

        # Concatenate the list of cluster IDs from old clusters 'a' and 'b'
        clusters[new_id] = clusters[a] + clusters[b]

        # Remove old cluster from the "clusters" dictionary
        clusters.pop(a)
        clusters.pop(b)
        i += 1

        # Stop when number of speakers are reached
        if oracle_num_spkrs and len(clusters) <= oracle_num_spkrs:
            break

    return clusters


def is_overlapped(end1, start2):
    """
    Returns True if segments are overlapping
    """
    if start2 > end1:
        return False
    else:
        return True


def merge_ssegs_same_speaker(lol):
    """
    Merge adjacent sub-segs from a same speaker
    Input lol structure: [rec_id, sseg_start, sseg_end, spkr_id]
    """
    new_lol = []

    # Start from the first sub-seg
    sseg = lol[0]

    # Loop over all sub-segments from 1 (not 0)
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
    """
    Distributes the overlapped speech equally among the adjacent segments with different speakers.
    Input lol structure: [rec_id, sseg_start, sseg_end, spkr_id]
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
            overlap = sseg[2] - next_sseg[1]

            # Update end time of old seg
            sseg[2] = sseg[2] - (overlap / 2.0)

            # Update start time of next seg
            next_sseg[1] = next_sseg[1] + (overlap / 2.0) + 0.001

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


def do_sc(diary_obj_eval, out_rttm_file, rec_id, k=4):

    xvects = copy.deepcopy(diary_obj_eval.stat1)

    clustering = SpectralClustering(
        n_clusters=4,
        assign_labels="kmeans",
        random_state=1234,
        affinity="nearest_neighbors",
    ).fit(xvects)

    labels = clustering.labels_

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


def diarizer(full_csv, split_type):

    split = "AMI_" + split_type

    A = [row[0].rstrip().split("_")[0] for row in full_csv]
    all_rec_ids = list(set(A[1:]))

    all_rec_ids.sort()

    N = str(len(all_rec_ids))
    i = 1

    # Loop through each recording
    for rec_id in all_rec_ids:

        ss = "[" + str(i) + "/" + N + "]"
        i = i + 1

        msg = "[%s] Diarizing %s : %s " % (split_type, ss, rec_id)
        logger.info(msg)

        if not os.path.exists(os.path.join(params.xvect_dir, split)):
            os.makedirs(os.path.join(params.xvect_dir, split))

        diary_stat_file = os.path.join(
            params.xvect_dir, split, rec_id + "_xv_stat.pkl"
        )

        # diary_ndx_file = os.path.join(
        #    params.xvect_dir, split, rec_id + "_ndx.pkl"
        # )

        # Prepare a csv for a recording
        out_csv_dir = os.path.join(params.xvect_dir, split)
        prepare_subset_csv(full_csv, rec_id, out_csv_dir)
        new_csv_file = os.path.join(params.xvect_dir, split, rec_id + ".csv")

        # Setup a dataloader for above one recording (above csv)
        diary_set = DataLoaderFactory(
            new_csv_file,
            params.diary_loader_eval.batch_size,
            params.diary_loader_eval.csv_read,
            params.diary_loader_eval.sentence_sorting,
        )

        diary_set_loader = diary_set.forward()

        if not os.path.exists(os.path.join(params.xvect_dir, split)):
            os.makedirs(os.path.join(params.xvect_dir, split))

        # msg = "Extracting xvectors for AMI " + split_type
        # logger.info(msg)

        # Compute Xvectors
        diary_obj_dev = xvect_computation_loop(
            "diary", diary_set_loader, diary_stat_file
        )

        """
        if params.len_norm is True:
            logger.info('len_norm')
            diary_obj_dev.norm_stat1()
        """
        """
        if params.do_lda is True:
            diary_obj_eval = lda_vox.do_lda(
                stat_server=diary_obj_eval,
                reduced_dim=params.lda_dim,
                transform_mat=lda_vox.transform_mat,
            )
        """

        # Whiten using Vox's mean and Sigma
        # diary_obj_eval.whiten_stat1(mean_vox, Sigma_vox)

        # Perform sc on each recording
        out_rttm_dir = os.path.join(params.sys_rttm_dir, split)
        if not os.path.exists(out_rttm_dir):
            os.makedirs(out_rttm_dir)
        out_rttm_file = out_rttm_dir + "/" + rec_id + ".rttm"

        # logger.info("Performing SC")
        do_sc(diary_obj_dev, out_rttm_file, rec_id)

    # Concatenate individual RTTM files
    # This is not needed but just staying with standards
    concate_rttm_file = out_rttm_dir + "/sys_output.rttm"

    logger.info("Concatenating individual RTTM files...")
    with open(concate_rttm_file, "w") as cat_file:
        for f in glob.glob(out_rttm_dir + "/*.rttm"):
            if f == concate_rttm_file:
                continue
            with open(f, "r") as indi_rttm_file:
                shutil.copyfileobj(indi_rttm_file, cat_file)

    msg = "Final system generated RTTM file for %s set : %s" % (
        split_type,
        concate_rttm_file,
    )
    logger.info(msg)

    return concate_rttm_file


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
        params.xvect_dir,
        params.csv_dir,
        params.ref_rttm_dir,
        params.sys_rttm_dir,
    ]
    for dir_ in exp_dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    try:
        from sklearn.cluster import SpectralClustering

    except ImportError:
        err_msg = "The optional dependency sklearn to use this module\n"
        err_msg += "cannot import sklearn. \n"
        err_msg += "Please follow the instructions below\n"
        err_msg += "=============================\n"
        err_msg += "Using pip:\n"
        err_msg += "pip install sklearn\n"
        err_msg += "================================ \n"
        err_msg += "Using conda:\n"
        err_msg += "conda install sklearn"
        raise ImportError(err_msg)

    # Prepare data for AMI
    logger.info(
        "AMI: Data preparation [Prepares both, the reference RTTMs and the CSVs]"
    )
    prepare_ami(
        data_folder=params.data_folder_ami,
        save_folder=params.save_folder,
        split_type=params.split_type,
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

    out_boundaries = diarizer(full_csv, "dev")

    logger.info("Evaluating for AMI Dev. set")
    ref_rttm = os.path.join(params.ref_rttm_dir, "fullref_ami_dev.rttm")
    sys_rttm = out_boundaries
    [MS, FA, SER, DER_] = DER(
        ref_rttm, sys_rttm, params.ignore_overlap, params.forgiveness_collar
    )
    msg = "DER (Dev. set)= %s %%\n" % (str(round(DER_, 2)))
    logger.info(msg)

    # AMI Eval Set
    full_csv = []
    with open(params.csv_diary_eval, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            full_csv.append(row)

    out_boundaries = diarizer(full_csv, "eval")

    logger.info("Evaluating for AMI Eval. set")
    ref_rttm = os.path.join(params.ref_rttm_dir, "fullref_ami_eval.rttm")
    sys_rttm = out_boundaries
    [MS, FA, SER, DER_] = DER(
        ref_rttm, sys_rttm, params.ignore_overlap, params.forgiveness_collar
    )
    msg = "DER (Eval. set) = %s %%\n" % (str(round(DER_, 2)))
    logger.info(msg)
