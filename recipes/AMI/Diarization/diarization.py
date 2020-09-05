#!/usr/bin/python
import os
import sys
import torch
import logging
import speechbrain as sb
import numpy
import pickle
import copy
from tqdm.contrib import tqdm
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from speechbrain.utils.EER import EER  # noqa F401
from speechbrain.utils.data_utils import download_file
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.processing.PLDA_LDA import Ndx
from speechbrain.processing.PLDA_LDA import fast_PLDA_scoring

# Logger setup
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

from voxceleb_prepare import prepare_voxceleb  # noqa E402
from ami_prepare import prepare_ami  # noqa E402


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
    save_model_path = params.output_folder + "/save/xvect.ckpt"
    download_file(params.xvector_file, save_model_path)
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
                    if "https://" in params.xvector_file:
                        download_and_pretrain()
                    else:
                        params.xvector_model.load_state_dict(
                            torch.load(params.xvector_file), strict=True
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
        logger.info(f"Skipping Xvector Extraction for {split}")
        logger.info(f"Loading previously saved stat_object for {split}")

        with open(stat_file, "rb") as inumpyut:
            stat_obj = pickle.load(inumpyut)

    return stat_obj


def trace_back_cluster_labels(links, threshold=0.9, oracle_num_spkrs=4):

    N = len(links) + 1

    # Create cluster dictionery
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
        clusters[new_id] = clusters[a] + clusters[b]
        clusters.pop(a)
        clusters.pop(b)
        i += 1
        if oracle_num_spkrs and len(clusters) <= oracle_num_spkrs:
            break
        # elif dist >= threshold:
        #    break

    return clusters


# Manage overlaped xvectors and prepare RTTM
def is_overlapped(end1, start2):
    if start2 > end1:
        return False
    else:
        return True


def merge_ssegs_same_speaker(lol):
    """
    Merge adjacent sub-segs from a same speaker
    """
    new_lol = []
    seg = lol[0]
    for i in range(1, len(lol)):
        new_seg = lol[i]
        if is_overlapped(seg[2], new_seg[1]) and seg[3] == new_seg[3]:
            seg[2] = new_seg[2]  # update end time
        else:
            new_lol.append(seg)
            seg = new_seg

    return new_lol


def distribute_overlap(lol):
    """
    Distributes the overlapped speech equally among the adjacent segments with different speakers.
    """
    new_lol = []
    seg = lol[0]
    for i in range(1, len(lol)):
        new_seg = lol[i]
        # No need to check if they are different speakers
        # Because if segments are overlapped then alway different speakers
        # This was taken care by merge_ssegs_same_speaker()
        if is_overlapped(seg[2], new_seg[1]):
            # divide
            overlap = seg[2] - new_seg[1]

            # update end time of old seg
            seg[2] = seg[2] - (overlap / 2.0)

            # Update start time of new seg
            new_seg[1] = new_seg[1] + (overlap / 2.0)  # + 0.001

            # To avoid duplicate entries
            if new_lol[-1] != seg:
                new_lol.append(seg)

            # The new_seg should always be added
            new_lol.append(new_seg)
            seg = new_seg
        else:
            # For first sseg
            if len(new_lol) == 0:
                new_lol.append(seg)

            # To avoid duplicate entries
            if new_lol[-1] != new_seg:
                new_lol.append(new_seg)
            seg = new_seg

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

    for i in rttm:
        print(i)
    with open(out_rttm_file, "w") as f:
        for row in rttm:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)


def do_ahc(score_matrix, out_rttm_file):

    ## Agglomerative Hierarchical Clustering Starts here
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

    print("AHC started...\n")
    links = linkage(dist_mat, method="complete")

    clusters = trace_back_cluster_labels(links)

    # Clusters IDs to segment labels
    # clus = dict()
    subseg_ids = diary_obj.modelset

    i = 0
    lol = []
    for c in clusters:
        clus_elements = clusters[c]
        # temp = []
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

    lol.sort(key=lambda x: float(x[1]))

    lol = merge_ssegs_same_speaker(lol)
    lol = distribute_overlap(lol)

    write_rttm(lol, out_rttm_file)


if __name__ == "__main__":

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params.output_folder,
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    """
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
    logger.info("AMI: Data preparation")
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
    train_set = params.train_loader()
    ind2lab = params.train_loader.label_dict["spk_id"]["index2lab"]

    # Xvector file for train data
    xv_file = os.path.join(
        params.save_folder, "VoxCeleb1_train_xvectors_stat_obj.pkl"
    )

    # Skip extraction of train if already extracted
    if not os.path.exists(xv_file):
        logger.info("Extracting xvectors from Training set..")
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
                    if "https://" in params.xvector_file:
                        download_and_pretrain()
                    else:
                        params.xvector_model.load_state_dict(
                            torch.load(params.xvector_file), strict=True
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
        logger.info("Skipping Xvector Extraction for training set")
        logger.info("Loading previously saved stat_object for train xvectors..")
        with open(xv_file, "rb") as inumpyut:
            xvectors_stat = pickle.load(inumpyut)

    # Training Gaussina PLDA model
    # (Using Xvectors from Voxceleb)
    logger.info("Training PLDA model")
    params.compute_plda.plda(xvectors_stat)
    logger.info("PLDA training completed")

    # xvector files
    diary_stat_file = os.path.join(params.save_folder, "diary_stat_enrol.pkl")
    diary_ndx_file = os.path.join(params.save_folder, "diary_ndx.pkl")

    # Data loader
    diary_set_loader = params.diary_loader()

    # Compute Xvectors
    diary_obj = xvect_computation_loop(
        "diary", diary_set_loader, diary_stat_file
    )

    # Loop for PLDA scoring per meeting
    all_rec_ids = set(diary_obj.modelset)

    # Prepare Ndx Object
    if not os.path.isfile(diary_ndx_file):
        models = diary_obj.modelset
        testsegs = diary_obj.modelset  # test_obj.modelset

        logger.info("Preparing Ndx")
        ndx_obj = Ndx(models=models, testsegs=testsegs)
        logger.info("Saving ndx obj...")
        ndx_obj.save_ndx_object(diary_ndx_file)
    else:
        logger.info("Skipping Ndx preparation")
        logger.info("Loading Ndx from disk")
        with open(diary_ndx_file, "rb") as inumpyut:
            ndx_obj = pickle.load(inumpyut)

    logger.info("PLDA scoring...")
    scores_plda = fast_PLDA_scoring(
        diary_obj,
        diary_obj,
        ndx_obj,
        params.compute_plda.mean,
        params.compute_plda.F,
        params.compute_plda.Sigma,
    )

    print("PLDA scoring completed...")
    print(scores_plda.scoremat)

    # Function to do AHC
    out_rttm_file = "results/save/abc.rttm"
    do_ahc(scores_plda.scoremat, out_rttm_file)
