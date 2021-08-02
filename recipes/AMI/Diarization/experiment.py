#!/usr/bin/python3
"""This recipe implements diarization system using deep embedding extraction followed by spectral clustering.

To run this recipe:
> python experiment.py hparams/<your_hyperparams_file.yaml>
 e.g., python experiment.py hparams/ecapa_tdnn.yaml

Condition: Oracle VAD (speech regions taken from the groundtruth).

Note: There are multiple ways to write this recipe. We iterate over individual recordings.
 This approach is less GPU memory demanding and also makes code easy to understand.

Citation: This recipe is based on the following paper,
 N. Dawalatabad, M. Ravanelli, F. Grondin, J. Thienpondt, B. Desplanques, H. Na,
 "ECAPA-TDNN Embeddings for Speaker Diarization," arXiv:2104.01466, 2021.

Authors
 * Nauman Dawalatabad 2020
"""

import os
import sys
import torch
import logging
import pickle
import json
import glob
import shutil
import numpy as np
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.processing import diarization as diar
from speechbrain.utils.DER import DER
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.dataio import read_audio_multichannel

np.random.seed(1234)

# Logger setup
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))


try:
    import sklearn  # noqa F401
except ImportError:
    err_msg = (
        "Cannot import optional dependency `sklearn` used in this module.\n"
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
    """Definition of the steps for computation of embeddings from the waveforms.
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


def embedding_computation_loop(split, set_loader, stat_file):
    """Extracts embeddings for a given dataset loader.
    """

    # Note: We use speechbrain.processing.PLDA_LDA.StatObject_SB type to store embeddings.
    # Extract embeddings (skip if already done).
    if not os.path.isfile(stat_file):
        logger.debug("Extracting deep embeddings and diarizing")
        embeddings = np.empty(shape=[0, params["emb_dim"]], dtype=np.float64)
        modelset = []
        segset = []

        # Different data may have different statistics.
        params["mean_var_norm_emb"].count = 0

        for batch in set_loader:
            ids = batch.id
            wavs, lens = batch.sig

            mod = [x for x in ids]
            seg = [x for x in ids]
            modelset = modelset + mod
            segset = segset + seg

            # Embedding computation.
            emb = (
                compute_embeddings(wavs, lens)
                .contiguous()
                .squeeze(1)
                .cpu()
                .numpy()
            )
            embeddings = np.concatenate((embeddings, emb), axis=0)

        modelset = np.array(modelset, dtype="|O")
        segset = np.array(segset, dtype="|O")

        # Intialize variables for start, stop and stat0.
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
        logger.debug("Skipping embedding extraction (as already present).")
        logger.debug("Loading previously saved embeddings.")

        with open(stat_file, "rb") as in_file:
            stat_obj = pickle.load(in_file)

    return stat_obj


def prepare_subset_json(full_meta_data, rec_id, out_meta_file):
    """Prepares metadata for a given recording ID.

    Arguments
    ---------
    full_meta_data : json
        Full meta (json) containing all the recordings
    rec_id : str
        The recording ID for which meta (json) has to be prepared
    out_meta_file : str
        Path of the output meta (json) file.
    """

    subset = {}
    for key in full_meta_data:
        k = str(key)
        if k.startswith(rec_id):
            subset[key] = full_meta_data[key]

    with open(out_meta_file, mode="w") as json_f:
        json.dump(subset, json_f, indent=2)


def diarize_dataset(full_meta, split_type, n_lambdas, pval, n_neighbors=10):
    """This function diarizes all the recordings in a given dataset. It performs
    computation of embedding and clusters them using spectral clustering (or other backends).
    The output speaker boundary file is stored in the RTTM format.
    """

    # Prepare `spkr_info` only once when Oracle num of speakers is selected.
    # spkr_info is essential to obtain number of speakers from groundtruth.
    if params["oracle_n_spkrs"] is True:
        full_ref_rttm_file = (
            params["ref_rttm_dir"] + "/fullref_ami_" + split_type + ".rttm"
        )

        rttm = diar.read_rttm(full_ref_rttm_file)

        spkr_info = list(  # noqa F841
            filter(lambda x: x.startswith("SPKR-INFO"), rttm)
        )

    # Get all the recording IDs in this dataset.
    all_keys = full_meta.keys()
    A = [word.rstrip().split("_")[0] for word in all_keys]
    all_rec_ids = list(set(A[1:]))
    all_rec_ids.sort()
    split = "AMI_" + split_type
    i = 1

    # Setting eval modality.
    params["embedding_model"].eval()
    msg = "Diarizing " + split_type + " set"
    logger.info(msg)

    if len(all_rec_ids) <= 0:
        msg = "No recording IDs found! Please check if meta_data json file is properly generated."
        logger.error(msg)
        sys.exit()

    # Diarizing different recordings in a dataset.
    for rec_id in tqdm(all_rec_ids):
        # This tag will be displayed in the log.
        tag = (
            "["
            + str(split_type)
            + ": "
            + str(i)
            + "/"
            + str(len(all_rec_ids))
            + "]"
        )
        i = i + 1

        # Log message.
        msg = "Diarizing %s : %s " % (tag, rec_id)
        logger.debug(msg)

        # Embedding directory.
        if not os.path.exists(os.path.join(params["embedding_dir"], split)):
            os.makedirs(os.path.join(params["embedding_dir"], split))

        # File to store embeddings.
        emb_file_name = rec_id + "." + params["mic_type"] + ".emb_stat.pkl"
        diary_stat_emb_file = os.path.join(
            params["embedding_dir"], split, emb_file_name
        )

        # Prepare a metadata (json) for one recording. This is basically a subset of full_meta.
        # Lets keep this meta-info in embedding directory itself.
        json_file_name = rec_id + "." + params["mic_type"] + ".json"
        meta_per_rec_file = os.path.join(
            params["embedding_dir"], split, json_file_name
        )

        # Write subset (meta for one recording) json metadata.
        prepare_subset_json(full_meta, rec_id, meta_per_rec_file)

        # Prepare data loader.
        diary_set_loader = dataio_prep(params, meta_per_rec_file)

        # Putting modules on the device.
        params["compute_features"].to(params["device"])
        params["mean_var_norm"].to(params["device"])
        params["embedding_model"].to(params["device"])
        params["mean_var_norm_emb"].to(params["device"])

        # Compute Embeddings.
        diary_obj = embedding_computation_loop(
            "diary", diary_set_loader, diary_stat_emb_file
        )

        # Adding tag for directory path.
        type_of_num_spkr = "oracle" if params["oracle_n_spkrs"] else "est"
        tag = (
            type_of_num_spkr
            + "_"
            + str(params["affinity"])
            + "_"
            + params["backend"]
        )
        out_rttm_dir = os.path.join(
            params["sys_rttm_dir"], params["mic_type"], split, tag
        )
        if not os.path.exists(out_rttm_dir):
            os.makedirs(out_rttm_dir)
        out_rttm_file = out_rttm_dir + "/" + rec_id + ".rttm"

        # Processing starts from here.
        if params["oracle_n_spkrs"] is True:
            # Oracle num of speakers.
            num_spkrs = diar.get_oracle_num_spkrs(rec_id, spkr_info)
        else:
            if params["affinity"] == "nn":
                # Num of speakers tunned on dev set (only for nn affinity).
                num_spkrs = n_lambdas
            else:
                # Num of speakers will be estimated using max eigen gap for cos based affinity.
                # So adding None here. Will use this None later-on.
                num_spkrs = None

        if params["backend"] == "kmeans":
            diar.do_kmeans_clustering(
                diary_obj, out_rttm_file, rec_id, num_spkrs, pval,
            )

        if params["backend"] == "SC":
            # Go for Spectral Clustering (SC).
            diar.do_spec_clustering(
                diary_obj,
                out_rttm_file,
                rec_id,
                num_spkrs,
                pval,
                params["affinity"],
                n_neighbors,
            )

        # Can used for AHC later. Likewise one can add different backends here.
        if params["backend"] == "AHC":
            # call AHC
            threshold = pval  # pval for AHC is nothing but threshold.
            diar.do_AHC(diary_obj, out_rttm_file, rec_id, num_spkrs, threshold)

    # Once all RTTM outputs are generated, concatenate individual RTTM files to obtain single RTTM file.
    # This is not needed but just staying with the standards.
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


def dev_pval_tuner(full_meta, split_type):
    """Tuning p_value for affinity matrix.
    The p_value used so that only p% of the values in each row is retained.
    """

    DER_list = []
    prange = np.arange(0.002, 0.015, 0.001)

    n_lambdas = None  # using it as flag later.
    for p_v in prange:
        # Process whole dataset for value of p_v.
        concate_rttm_file = diarize_dataset(
            full_meta, split_type, n_lambdas, p_v
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

        if params["oracle_n_spkrs"] is True and params["backend"] == "kmeans":
            # no need of p_val search. Note p_val is needed for SC for both oracle and est num of speakers.
            # p_val is needed in oracle_n_spkr=False when using kmeans backend.
            break

    # Take p_val that gave minmum DER on Dev dataset.
    tuned_p_val = prange[DER_list.index(min(DER_list))]

    return tuned_p_val


def dev_ahc_threshold_tuner(full_meta, split_type):
    """Tuning threshold for affinity matrix. This function is called when AHC is used as backend.
    """

    DER_list = []
    prange = np.arange(0.0, 1.0, 0.1)

    n_lambdas = None  # using it as flag later.

    # Note: p_val is threshold in case of AHC.
    for p_v in prange:
        # Process whole dataset for value of p_v.
        concate_rttm_file = diarize_dataset(
            full_meta, split_type, n_lambdas, p_v
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

        if params["oracle_n_spkrs"] is True:
            break  # no need of threshold search.

    # Take p_val that gave minmum DER on Dev dataset.
    tuned_p_val = prange[DER_list.index(min(DER_list))]

    return tuned_p_val


def dev_nn_tuner(full_meta, split_type):
    """Tuning n_neighbors on dev set. Assuming oracle num of speakers.
    This is used when nn based affinity is selected.
    """

    DER_list = []
    pval = None

    # Now assumming oracle num of speakers.
    n_lambdas = 4

    for nn in range(5, 15):

        # Process whole dataset for value of n_lambdas.
        concate_rttm_file = diarize_dataset(
            full_meta, split_type, n_lambdas, pval, nn
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

        if params["oracle_n_spkrs"] is True and params["backend"] == "kmeans":
            break

    DER_list.sort(key=lambda x: x[1])
    tunned_nn = DER_list[0]

    return tunned_nn[0]


def dev_tuner(full_meta, split_type):
    """Tuning n_components on dev set. Used for nn based affinity matrix.
    Note: This is a very basic tunning for nn based affinity.
    This is work in progress till we find a better way.
    """

    DER_list = []
    pval = None
    for n_lambdas in range(1, params["max_num_spkrs"] + 1):

        # Process whole dataset for value of n_lambdas.
        concate_rttm_file = diarize_dataset(
            full_meta, split_type, n_lambdas, pval
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

    # Take n_lambdas with minmum DER.
    tuned_n_lambdas = DER_list.index(min(DER_list)) + 1

    return tuned_n_lambdas


def dataio_prep(hparams, json_file):
    """Creates the datasets and their data processing pipelines.
    This is used for multi-mic processing.
    """

    # 1. Datasets
    data_folder = hparams["data_folder"]
    dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=json_file, replacements={"data_root": data_folder},
    )

    # 2. Define audio pipeline.
    if params["mic_type"] == "Array1":
        # Multi-mic (Microphone Array)
        @sb.utils.data_pipeline.takes("wav")
        @sb.utils.data_pipeline.provides("sig")
        def audio_pipeline(wav):
            mics_signals = read_audio_multichannel(wav).unsqueeze(0)
            sig = params["multimic_beamformer"](mics_signals)
            sig = sig.squeeze()
            return sig

    else:
        # Single microphone
        @sb.utils.data_pipeline.takes("wav")
        @sb.utils.data_pipeline.provides("sig")
        def audio_pipeline(wav):
            sig = read_audio(wav)
            return sig

    sb.dataio.dataset.add_dynamic_item([dataset], audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys([dataset], ["id", "sig"])

    # 4. Create dataloader:
    dataloader = sb.dataio.dataloader.make_dataloader(
        dataset, **params["dataloader_opts"]
    )

    return dataloader


# Begin experiment!
if __name__ == "__main__":  # noqa: C901

    # Load hyperparameters file with command-line overrides.
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])

    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Dataset prep (peparing metadata files)
    from ami_prepare import prepare_ami  # noqa

    run_on_main(
        prepare_ami,
        kwargs={
            "data_folder": params["data_folder"],
            "save_folder": params["save_folder"],
            "ref_rttm_dir": params["ref_rttm_dir"],
            "meta_data_dir": params["meta_data_dir"],
            "manual_annot_folder": params["manual_annot_folder"],
            "split_type": params["split_type"],
            "skip_TNO": params["skip_TNO"],
            "mic_type": params["mic_type"],
            "vad_type": params["vad_type"],
            "max_subseg_dur": params["max_subseg_dur"],
            "overlap": params["overlap"],
        },
    )

    # Create experiment directory.
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Few more experiment directories inside results/ (to maintain cleaner structure).
    exp_dirs = [
        params["embedding_dir"],
        params["sys_rttm_dir"],
        params["der_dir"],
    ]
    for dir_ in exp_dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    # We download the pretrained Model from HuggingFace (or elsewhere depending on
    # the path given in the YAML file).
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected()
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])

    # AMI Dev Set: Tune hyperparams on dev set.
    # Read the meta-data file for dev set generated during data_prep
    dev_meta_file = os.path.join(
        params["meta_data_dir"],
        "ami_dev." + params["mic_type"] + ".subsegs.json",
    )
    with open(dev_meta_file, "r") as f:
        meta_dev = json.load(f)

    full_meta = meta_dev

    # Processing starts from here
    # Following few lines selects option for different backend and affinity matrices. Finds best values for hyperameters using dev set.
    best_nn = None
    if params["affinity"] == "nn":
        logger.info("Tuning for nn (Multiple iterations over AMI Dev set)")
        best_nn = dev_nn_tuner(full_meta, "dev")

    n_lambdas = None
    best_pval = None

    if params["affinity"] == "cos" and (
        params["backend"] == "SC" or params["backend"] == "kmeans"
    ):
        # oracle num_spkrs or not, doesn't matter for kmeans and SC backends
        # cos: Tune for the best pval for SC /kmeans (for unknown num of spkrs)
        logger.info(
            "Tuning for p-value for SC (Multiple iterations over AMI Dev set)"
        )
        best_pval = dev_pval_tuner(full_meta, "dev")

    elif params["backend"] == "AHC":
        logger.info("Tuning for threshold-value for AHC")
        best_threshold = dev_ahc_threshold_tuner(full_meta, "dev")
        best_pval = best_threshold
    else:
        # NN for unknown num of speakers (can be used in future)
        if params["oracle_n_spkrs"] is False:
            # nn: Tune num of number of components (to be updated later)
            logger.info(
                "Tuning for number of eigen components for NN (Multiple iterations over AMI Dev set)"
            )
            # dev_tuner used for tuning num of components in NN. Can be used in future.
            n_lambdas = dev_tuner(full_meta, "dev")

    # Load 'dev' and 'eval' metadata files.
    full_meta_dev = full_meta  # current full_meta is for 'dev'
    eval_meta_file = os.path.join(
        params["meta_data_dir"],
        "ami_eval." + params["mic_type"] + ".subsegs.json",
    )
    with open(eval_meta_file, "r") as f:
        full_meta_eval = json.load(f)

    # Tag to be appended to final output DER files. Writing DER for individual files.
    type_of_num_spkr = "oracle" if params["oracle_n_spkrs"] else "est"
    tag = (
        type_of_num_spkr
        + "_"
        + str(params["affinity"])
        + "."
        + params["mic_type"]
    )

    # Perform final diarization on 'dev' and 'eval' with best hyperparams.
    final_DERs = {}
    for split_type in ["dev", "eval"]:
        if split_type == "dev":
            full_meta = full_meta_dev
        else:
            full_meta = full_meta_eval

        # Performing diarization.
        msg = "Diarizing using best hyperparams: " + split_type + " set"
        logger.info(msg)
        out_boundaries = diarize_dataset(
            full_meta,
            split_type,
            n_lambdas=n_lambdas,
            pval=best_pval,
            n_neighbors=best_nn,
        )

        # Computing DER.
        msg = "Computing DERs for " + split_type + " set"
        logger.info(msg)
        ref_rttm = os.path.join(
            params["ref_rttm_dir"], "fullref_ami_" + split_type + ".rttm"
        )
        sys_rttm = out_boundaries
        [MS, FA, SER, DER_vals] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
            individual_file_scores=True,
        )

        # Writing DER values to a file. Append tag.
        der_file_name = split_type + "_DER_" + tag
        out_der_file = os.path.join(params["der_dir"], der_file_name)
        msg = "Writing DER file to: " + out_der_file
        logger.info(msg)
        diar.write_ders_file(ref_rttm, DER_vals, out_der_file)

        msg = (
            "AMI "
            + split_type
            + " set DER = %s %%\n" % (str(round(DER_vals[-1], 2)))
        )
        logger.info(msg)
        final_DERs[split_type] = round(DER_vals[-1], 2)

    # Final print DERs
    msg = (
        "Final Diarization Error Rate (%%) on AMI corpus: Dev = %s %% | Eval = %s %%\n"
        % (str(final_DERs["dev"]), str(final_DERs["eval"]))
    )
    logger.info(msg)
