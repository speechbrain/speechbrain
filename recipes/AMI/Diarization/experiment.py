#!/usr/bin/python3
"""This recipe implements diarization baseline
using deep embedding extraction followed by spectral clustering.

To run this recipe, do the following:
> python experiment.py hyperparams.yaml

Condition: Oracle VAD

Note: There are multiple ways to write this recipe. We chose to iterate over individual files.
This method is less GPU memory demanding and also makes code easy to understand.


Authors
 * Nauman Dawalatabad 2020
"""

import os
import sys
import torch
import logging
import pickle
import csv
import glob
import shutil
import numpy as np
import speechbrain as sb
from tqdm.contrib import tqdm

from speechbrain.utils.data_utils import download_file
from speechbrain.data_io.data_io import DataLoaderFactory
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.processing import diarization as diar
from speechbrain.utils.DER import DER

np.random.seed(1234)

# Logger setup
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))


try:
    import sklearn  # noqa F401
except ImportError:
    err_msg = (
        "Cannot import optional dependency `sklearn` used in this module\n"
    )
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
        emb = params["embedding_model"](feats, lens)
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
        logger.debug("Extracting deep embeddings and diarizing")
        embeddings = np.empty(shape=[0, params["emb_dim"]], dtype=np.float64)
        modelset = []
        segset = []

        # Different data may have different statistics
        params["mean_var_norm_emb"].count = 0

        for wav in set_loader:  # t:
            ids, wavs, lens = wav[0]

            mod = [x for x in ids]
            seg = [x for x in ids]
            modelset = modelset + mod
            segset = segset + seg

            # Embedding computation
            emb = compute_embeddings(wavs, lens).squeeze(1).cpu().numpy()
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
        logger.debug("Saving Embeddings...")
        stat_obj.save_stat_object(stat_file)

    else:
        logger.debug("Skipping embedding extraction (as already present)")
        logger.debug("Loading previously saved embeddings")

        with open(stat_file, "rb") as in_file:
            stat_obj = pickle.load(in_file)

    return stat_obj


def diarize_dataset(full_csv, split_type, n_lambdas, pval, n_neighbors=10):
    """Diarizes all the recordings in a given dataset
    """

    # Prepare `spkr_info` only once when Oracle num of speakers is selected
    if params["oracle_n_spkrs"] is True:
        full_ref_rttm_file = (
            params["ref_rttm_dir"] + "/fullref_ami_" + split_type + ".rttm"
        )

        rttm = diar.read_rttm(full_ref_rttm_file)

        spkr_info = list(  # noqa F841
            filter(lambda x: x.startswith("SPKR-INFO"), rttm)
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
    msg = "Diarizing " + split_type + " set"
    logger.info(msg)
    for rec_id in tqdm(all_rec_ids):

        tag = "[" + str(split_type) + ": " + str(i) + "/" + N + "]"
        i = i + 1

        msg = "Diarizing %s : %s " % (tag, rec_id)
        logger.debug(msg)

        if not os.path.exists(os.path.join(params["embedding_dir"], split)):
            os.makedirs(os.path.join(params["embedding_dir"], split))

        diary_stat_file = os.path.join(
            params["embedding_dir"], split, rec_id + "_xv_stat.pkl"
        )

        # Prepare a csv for a recording
        new_csv_file = os.path.join(
            params["embedding_dir"], split, rec_id + ".csv"
        )
        diar.prepare_subset_csv(full_csv, rec_id, new_csv_file)

        # Setup a dataloader for above one recording (above csv)
        diary_set = DataLoaderFactory(
            new_csv_file,
            params["diary_loader"].batch_size,
            params["diary_loader"].csv_read,
            params["diary_loader"].sentence_sorting,
        )

        diary_set_loader = diary_set.forward().get_dataloader()

        # Putting modules on the device
        params["compute_features"].to(params["device"])
        params["mean_var_norm"].to(params["device"])
        params["embedding_model"].to(params["device"])
        params["mean_var_norm_emb"].to(params["device"])

        # Compute Embeddings
        diary_obj = embedding_computation_loop(
            "diary", diary_set_loader, diary_stat_file
        )

        # Perform spectral clustering
        out_rttm_dir = os.path.join(params["sys_rttm_dir"], split)
        if not os.path.exists(out_rttm_dir):
            os.makedirs(out_rttm_dir)
        out_rttm_file = out_rttm_dir + "/" + rec_id + ".rttm"

        if params["oracle_n_spkrs"] is True:
            # Oracle num of speakers
            num_spkrs = diar.get_oracle_num_spkrs(rec_id, spkr_info)
        else:
            if params["affinity"] == "nn":
                # Num of speakers tunned on dev set
                num_spkrs = n_lambdas
            else:
                # Will be estimated using max eigen gap for cos based affinity
                num_spkrs = None

        diar.do_spec_clustering(
            diary_obj,
            out_rttm_file,
            rec_id,
            num_spkrs,
            pval,
            params["affinity"],
            n_neighbors,
        )

    # Concatenate individual RTTM files
    # This is not needed but just staying with the standards
    concate_rttm_file = out_rttm_dir + "/sys_output.rttm"

    logger.debug("Concatenating individual RTTM files...")
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
    logger.debug(msg)

    return concate_rttm_file


def dev_p_tuner(full_csv, split_type):
    """Tuning p_value affinity matrix
    """

    DER_list = []
    prange = [
        0.0025,
        0.0050,
        0.006,
        0.007,
        0.008,
        0.009,
        0.010,
        0.025,
        0.050,
        0.075,
        0.100,
    ]
    n_lambdas = None
    for p_v in prange:
        # Process whole dataset for value of p_v
        concate_rttm_file = diarize_dataset(
            full_csv, split_type, n_lambdas, p_v
        )

        ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_ami_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
        )

        DER_list.append(DER_)

    # Take p_val that gave minmum DER on Dev dataset
    tuned_p_val = prange[DER_list.index(min(DER_list))]

    return tuned_p_val


def dev_nn_tuner(full_csv, split_type):
    """Tuning n_neighbors on dev set.
    Assuming oracle num of speakers.
    """

    DER_list = []
    pval = None
    for nn in range(5, 15):

        # Fix this later. Now assumming oracle num of speakers
        n_lambdas = 4

        # Process whole dataset for value of n_lambdas
        concate_rttm_file = diarize_dataset(
            full_csv, split_type, n_lambdas, pval, nn
        )

        ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_ami_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
        )

        DER_list.append([nn, DER_])

    DER_list.sort(key=lambda x: x[1])
    tunned_nn = DER_list[0]

    return tunned_nn[0]


def dev_tuner(full_csv, split_type):
    """Tuning n_components on dev set.
    Note: This is a very basic tunning for nn based affinity.
    This is work in progress till we find a better way.
    """

    DER_list = []
    pval = None
    for n_lambdas in range(1, params["max_num_spkrs"] + 1):

        # Process whole dataset for value of n_lambdas
        concate_rttm_file = diarize_dataset(
            full_csv, split_type, n_lambdas, pval
        )

        ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_ami_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
        )

        DER_list.append(DER_)

    # Take n_lambdas with minmum DER
    tuned_n_lambdas = DER_list.index(min(DER_list)) + 1

    return tuned_n_lambdas


# Begin!
if __name__ == "__main__":  # noqa: C901

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])

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
        params["der_dir"],
    ]
    for dir_ in exp_dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    # AMI Dev Set
    full_csv = []
    with open(params["csv_diary_dev"], "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            full_csv.append(row)

    best_nn = None
    if params["affinity"] == "nn":
        logger.info("Tuning for nn (Multiple iterations over AMI Dev set)")
        best_nn = dev_nn_tuner(full_csv, "dev")

    n_lambdas = None
    best_pval = None
    if params["affinity"] == "cos":  # oracle num_spkrs or not doesn't matter
        # cos: Tune for best pval
        logger.info("Tuning for p-value (Multiple iterations over AMI Dev set)")
        best_pval = dev_p_tuner(full_csv, "dev")
    else:
        if params["oracle_n_spkrs"] is False:
            # nn: Tune num of number of components (to be updated later)
            logger.info(
                "Tuning for number of eigen components (Multiple iterations over AMI Dev set)"
            )
            n_lambdas = dev_tuner(full_csv, "dev")

    # Running once more of dev set (optional)
    out_boundaries = diarize_dataset(
        full_csv,
        "dev",
        n_lambdas=n_lambdas,
        pval=best_pval,
        n_neighbors=best_nn,
    )

    # Evaluating on DEV set
    logger.info("Evaluating for AMI Dev. set")
    ref_rttm_dev = os.path.join(params["ref_rttm_dir"], "fullref_ami_dev.rttm")
    sys_rttm_dev = out_boundaries
    [MS_dev, FA_dev, SER_dev, DER_dev] = DER(
        ref_rttm_dev,
        sys_rttm_dev,
        params["ignore_overlap"],
        params["forgiveness_collar"],
        individual_file_scores=True,
    )
    msg = "AMI Dev set: Diarization Error Rate = %s %%\n" % (
        str(round(DER_dev[-1], 2))
    )
    logger.info(msg)

    # AMI Eval Set
    full_csv = []
    with open(params["csv_diary_eval"], "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            full_csv.append(row)

    out_boundaries = diarize_dataset(
        full_csv,
        "eval",
        n_lambdas=n_lambdas,
        pval=best_pval,
        n_neighbors=best_nn,
    )

    # Evaluating on EVAL set
    logger.info("Evaluating for AMI Eval. set")
    ref_rttm_eval = os.path.join(
        params["ref_rttm_dir"], "fullref_ami_eval.rttm"
    )
    sys_rttm_eval = out_boundaries
    [MS_eval, FA_eval, SER_eval, DER_eval] = DER(
        ref_rttm_eval,
        sys_rttm_eval,
        params["ignore_overlap"],
        params["forgiveness_collar"],
        individual_file_scores=True,
    )
    msg = "AMI Eval set: Diarization Error Rate = %s %%\n" % (
        str(round(DER_eval[-1], 2))
    )
    logger.info(msg)

    msg = (
        "Final Diarization Error Rate (%%) on AMI corpus: Dev = %s %% | Eval = %s %%\n"
        % (str(round(DER_dev[-1], 2)), str(round(DER_eval[-1], 2)))
    )
    logger.info(msg)

    # Writing der for individual files
    t0 = "oracle" if params["oracle_n_spkrs"] else "est"
    tag = t0 + "_" + str(params["affinity"]) + ".txt"

    out_der_file = os.path.join(params["der_dir"], "dev_DER_" + tag)
    diar.write_ders_file(ref_rttm_dev, DER_dev, out_der_file)

    out_der_file = os.path.join(params["der_dir"], "eval_DER_" + tag)
    diar.write_ders_file(ref_rttm_eval, DER_eval, out_der_file)
