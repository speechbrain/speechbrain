#!/usr/bin/python
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
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from speechbrain.utils.DER import DER  # noqa F401
from speechbrain.utils.data_utils import download_file
from speechbrain.data_io.data_io import DataLoaderFactory
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.processing.PLDA_LDA import Ndx
from speechbrain.processing.PLDA_LDA import fast_PLDA_scoring

# Logger setup
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))


from voxceleb_prepare import prepare_voxceleb  # noqa E402
from ami_prepare import prepare_ami  # noqa E402
from ami_splits import get_AMI_split  # noqa E402

"""
# Verify correctness of the final segments
# Simply run the DER evaluation and check for MISSED SPEECH and FALARM SPEECH

MISSED SPEECH =      0.00 secs (  0.0 percent of scored time)
FALARM SPEECH =      0.00 secs (  0.0 percent of scored time)
MISSED WORDS =      0         (100.0 percent of scored words)
---------------------------------------------
SCORED SPEAKER TIME =    496.47 secs (100.0 percent of scored speech)
MISSED SPEAKER TIME =      0.00 secs (  0.0 percent of scored speaker time)
FALARM SPEAKER TIME =      0.00 secs (  0.0 percent of scored speaker time)
"""

"""
Note:
Though variable/funtion names are quite interpretable, the following points are useful in clearly understanding this code.
- "segments" reffer to the big chunks that comes from the VAD or groundtruth.
- "sub-segment" or "sseg" denotes smaller duration segments that are generated from "segments"
- There are multiple ways to write this recipe. We chose to iterate individual file.
This method is less GPU memory demnading and easy to understand and update the code.
"""


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
    save_model_path = params.output_folder + "/save/models/xvect.ckpt"
    download_file(params.xvector_f, save_model_path)
    params.xvector_model.load_state_dict(
        torch.load(save_model_path), strict=True
    )


# Function to get mod and seg
def get_utt_ids_for_test(ids, data_dict):
    mod = [data_dict[x]["wav1"]["data"] for x in ids]
    seg = [data_dict[x]["wav2"]["data"] for x in ids]

    return mod, seg


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
        logger.info(f"Saving stat obj for {split}")
        stat_obj.save_stat_object(stat_file)

    else:
        logger.info(
            f"Skipping Xvector Extraction for {split} (as already present)"
        )
        logger.info(f"Loading previously saved stat_object for {split}")

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


def trace_back_cluster_labels(links, threshold=0.9, oracle_num_spkrs=4):

    N = len(links) + 1

    # Cluster dictionery for maintaining cluster IDs and segments inside each cluster ID
    # example: clusters['10'] : [1,4,7] denotes cluster number 10 has segment ids 1, 4 and 7
    clusters = dict()

    # Initialize clusters (cluster IDs starts from 0 )
    for i in range(N):
        clusters[str(i)] = [i]

    i = 0
    while i < N - 1:

        a = str(int(links[i, 0]))
        b = str(int(links[i, 1]))

        # dist = links[i, 2]
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
        # Stop when PLDA threshold is reached
        # elif dist >= threshold:
        #    break

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
            next_sseg[1] = next_sseg[1] + (overlap / 2.0)  # + 0.001

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
            # For first sseg
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


def do_ahc(score_matrix, out_rttm_file, rec_id):

    ## Agglomerative Hierarchical Clustering using linkage
    scores = copy.deepcopy(score_matrix)

    # Ignoring diagonal scores and normalizing as per remaining pairs
    mn = numpy.tril(scores).min()
    mx = numpy.tril(scores).max()
    scores = (scores - mn) / (mx - mn)

    # Prepare distance matrix
    dist_mat = (scores + scores.T) / 2.0 * -1.0
    dist_mat = dist_mat - numpy.tril(dist_mat).min()
    numpy.fill_diagonal(dist_mat, 0.0)
    dist_mat = squareform(dist_mat)

    links = linkage(dist_mat, method="complete")

    # Convert the links into interpretable clusters
    clusters = trace_back_cluster_labels(links)

    # Clusters IDs to segment labels
    # clus = dict()
    subseg_ids = diary_obj.modelset

    i = 0  # speaker/cluster ID
    lol = []
    for c in clusters:
        clus_elements = clusters[c]

        for elem in clus_elements:

            sub_seg = subseg_ids[elem]

            splitted = sub_seg.rsplit("_", 2)
            rec_id = str(splitted[0])
            sseg_start = float(splitted[1])
            sseg_end = float(splitted[2])
            spkr_id = rec_id + "_" + str(i)

            a = [rec_id, sseg_start, sseg_end, spkr_id]
            lol.append(a)
        i += 1

    # Sorting based on start time of sub-segment
    lol.sort(key=lambda x: float(x[1]))

    # Proceed in simple 2 steps: (i) Merge sseg of same speakers then (ii) splits different speakers
    # Merge adjacent sub-segments that belong to same speaker (or cluster)
    lol = merge_ssegs_same_speaker(lol)

    # Distribute duration of adjacent overlapping sub-segments belonging to different speakers (or cluster)
    # Taking mid-point as the splitting time location.
    lol = distribute_overlap(lol)

    logger.info("Completed diarizing " + rec_id)
    write_rttm(lol, out_rttm_file)


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

    # Few more experiment directories
    models_dir = os.path.join(params.save_folder, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    xvect_dir = os.path.join(params.save_folder, "xvectors")
    if not os.path.exists(xvect_dir):
        os.makedirs(xvect_dir)

    csv_dir = os.path.join(params.save_folder, "csv")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    ref_dir = os.path.join(params.save_folder, "ref_RTTM")
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)

    rttm_dir = os.path.join(params.save_folder, "sys_RTTM")
    if not os.path.exists(rttm_dir):
        os.makedirs(rttm_dir)

    """
    # Prepare data from dev of Voxceleb1
    logger.info("Vox: Data preparation")
    prepare_voxceleb(
        data_folder=params.data_folder,
        save_folder=params.save_folder,
        splits=["train", "test"],
        split_ratio=[90, 10],
        seg_dur=300,
        vad=False,
        rand_seed=params.seed,
    )
    """

    # Prepare data for AMI
    # TODO: Shift this below (to improve code readability)
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

    # PLDA inputs for Train data
    modelset, segset = [], []
    xvectors = numpy.empty(shape=[0, params.xvect_dim], dtype=numpy.float64)

    # Train set
    train_set = params.train_loader_vox()
    ind2lab = params.train_loader_vox.label_dict["spk_id"]["index2lab"]

    # Xvector file for train data
    if not os.path.exists(os.path.join(xvect_dir, "Vox1_train")):
        os.makedirs(os.path.join(xvect_dir, "Vox1_train"))

    xv_file = os.path.join(
        xvect_dir, "Vox1_train/VoxCeleb1_train_xvectors_stat_obj.pkl"
    )

    # Skip extraction of train if already extracted
    if not os.path.exists(xv_file):
        logger.info("[Vox] Extracting xvectors from Training set..")
        with tqdm(train_set, dynamic_ncols=True) as t:
            init_params = True
            for wav, spk_id in t:
                _, wav, lens = wav
                id, spk_id, lens = spk_id

                # For modelset
                spk_id_str = convert_index_to_lab(spk_id, ind2lab)

                # Flattening speaker ids
                spk_ids = [sid[0] for sid in spk_id_str]
                modelset = modelset + spk_ids

                # For segset
                segset = segset + id

                if init_params:
                    xvect = compute_x_vectors(wav, lens, init_params=True)
                    params.mean_var_norm_xvect.glob_mean = torch.zeros_like(
                        xvect[0, 0, :]
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
                xvect = compute_x_vectors(wav, lens)

                xv = xvect.squeeze().cpu().numpy()
                xvectors = numpy.concatenate((xvectors, xv), axis=0)

        # Speaker IDs and utterance IDs
        modelset = numpy.array(modelset, dtype="|O")
        segset = numpy.array(segset, dtype="|O")

        # intialize variables for start, stop and stat0
        s = numpy.array([None] * xvectors.shape[0])
        b = numpy.array([[1.0]] * xvectors.shape[0])

        xvectors_stat = StatObject_SB(
            modelset=modelset,
            segset=segset,
            start=s,
            stop=s,
            stat0=b,
            stat1=xvectors,
        )

        del xvectors

        # Save TRAINING Xvectors in StatObject_SB object
        xvectors_stat.save_stat_object(xv_file)

    else:
        # Load the saved stat object for train xvector
        logger.info(
            "Skipping Xvector Extraction for Vox1 train set (as already present)"
        )
        logger.info("Loading previously saved stat_object for train xvectors..")
        with open(xv_file, "rb") as in_file:
            xvectors_stat = pickle.load(in_file)

    # Training Gaussian PLDA model
    # (Using Xvectors extracted from Voxceleb)

    if params.whiten is True:

        # Compute Xvectors for AMI dev set (will use it to whiten)
        if not os.path.exists(os.path.join(xvect_dir, "AMI_dev")):
            os.makedirs(os.path.join(xvect_dir, "AMI_dev"))

        diary_stat_file = os.path.join(xvect_dir, "AMI_dev/ami_dev_xv_stat.pkl")

        logger.info("Extracting xvectors for AMI DEV")
        diary_obj = xvect_computation_loop(
            "diary", params.diary_loader_dev(), diary_stat_file
        )

        logger.info(
            "Training PLDA model using Vox1 and whitening using AMI dev"
        )
        params.compute_plda.plda(
            xvectors_stat, whiten=True, w_stat_server=diary_obj
        )
        del diary_obj
    else:
        logger.info("Training PLDA model using Vox1 (no whitening)")
        params.compute_plda.plda(xvectors_stat)

    logger.info("PLDA training completed")

    # EVAL
    # Get all recording IDs in AMI test split
    full_diary_csv = []
    with open(params.csv_diary_eval, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            full_diary_csv.append(row)

    A = [row[0].rstrip().split("_")[0] for row in full_diary_csv]
    all_rec_ids = list(set(A[1:]))

    # Set directories to store xvectors (not neccesaary) and rttms (must save)
    # xvect_dir = os.path.join(params.save_folder, "xvectors")
    rttm_dir = os.path.join(params.save_folder, "sys_RTTM")

    if not os.path.exists(rttm_dir):
        os.makedirs(rttm_dir)

    all_rec_ids.sort()

    N = str(len(all_rec_ids))
    i = 1

    # Loop through all recordings in AMI Test split
    for rec_id in all_rec_ids:

        ss = "[" + str(i) + "/" + N + "]"
        i = i + 1

        # Replace with tqdm??? (doesn't make any difference!!)
        msg = "Diarizing %s : %s " % (ss, rec_id)
        logger.info(msg)

        if not os.path.exists(os.path.join(xvect_dir, "AMI_eval")):
            os.makedirs(os.path.join(xvect_dir, "AMI_eval"))

        diary_stat_file = os.path.join(
            xvect_dir, "AMI_eval", rec_id + "_xv_stat.pkl"
        )
        diary_ndx_file = os.path.join(
            xvect_dir, "AMI_eval", rec_id + "_ndx.pkl"
        )

        # Prepare a csv for a recording
        prepare_subset_csv(full_diary_csv, rec_id, xvect_dir + "/AMI_eval/")
        new_csv_file = os.path.join(xvect_dir, "AMI_eval", rec_id + ".csv")

        # Setup a dataloader for above csv
        diary_set = DataLoaderFactory(
            new_csv_file,
            params.diary_loader_eval.batch_size,
            params.diary_loader_eval.csv_read,
            params.diary_loader_eval.sentence_sorting,
        )

        diary_set_loader = diary_set.forward()

        # Compute Xvectors
        diary_obj = xvect_computation_loop(
            "diary", diary_set_loader, diary_stat_file
        )

        # Prepare Ndx Object
        if not os.path.isfile(diary_ndx_file):
            models = diary_obj.modelset
            testsegs = diary_obj.modelset

            logger.info("Preparing Ndx")
            ndx_obj = Ndx(models=models, testsegs=testsegs)
            logger.info("Saving ndx obj...")
            ndx_obj.save_ndx_object(diary_ndx_file)
        else:
            logger.info("Skipping Ndx preparation")
            logger.info("Loading Ndx from disk")
            with open(diary_ndx_file, "rb") as in_file:
                ndx_obj = pickle.load(in_file)

        logger.info("Performing PLDA scoring")
        scores_plda = fast_PLDA_scoring(
            diary_obj,
            diary_obj,
            ndx_obj,
            params.compute_plda.mean,
            params.compute_plda.F,
            params.compute_plda.Sigma,
        )

        # print(scores_plda.scoremat)

        # Do AHC
        out_rttm_file = rttm_dir + "/" + rec_id + ".rttm"
        logger.info("Performing AHC")
        do_ahc(scores_plda.scoremat, out_rttm_file, rec_id)

    # (Optional) Concatenate individual RTTM files
    concate_rttm_file = rttm_dir + "/sys_output.rttm"

    with open(concate_rttm_file, "w") as cat_file:
        for f in glob.glob(rttm_dir + "/*.rttm"):
            if f == concate_rttm_file:
                continue
            with open(f, "r") as indi_rttm_file:
                shutil.copyfileobj(indi_rttm_file, cat_file)

    logger.info(
        "Final system generated concatenated RTTM saved at : "
        + concate_rttm_file
    )
