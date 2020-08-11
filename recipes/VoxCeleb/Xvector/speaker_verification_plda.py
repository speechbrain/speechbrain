#!/usr/bin/python
import os
import sys
import torch
import logging
import speechbrain as sb
import numpy
import pickle

from tqdm.contrib import tqdm
from speechbrain.utils.EER import EER
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

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    hyperparams_to_save=params_file,
    overrides=overrides,
)

# Prepare data from dev of Voxceleb1
logger.info("Data preparation")
prepare_voxceleb(
    data_folder=params.data_folder,
    save_folder=params.save_folder,
    splits=["train", "dev", "enrol"],
    split_ratio=[90, 10],
    seg_dur=300,
    vad=False,
    rand_seed=params.seed,
)


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
    with open(xv_file, "rb") as input:
        xvectors_stat = pickle.load(input)


# Training Gaussina PLDA model
logger.info("Training PLDA model")
params.compute_plda.plda(xvectors_stat)
logger.info("PLDA training completed")

# Enroll and Test xvector
enrol_stat_file = os.path.join(params.save_folder, "stat_enrol.pkl")
test_stat_file = os.path.join(params.save_folder, "stat_test.pkl")
ndx_file = os.path.join(params.save_folder, "ndx.pkl")

# Data loader
enrol_set_loader = params.enrol_loader()
test_set_loader = params.test_loader()


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

                # Enrolment and test xvectors
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

        with open(stat_file, "rb") as input:
            stat_obj = pickle.load(input)

    return stat_obj


enrol_obj = xvect_computation_loop("enrol", enrol_set_loader, enrol_stat_file)
test_obj = xvect_computation_loop("test", test_set_loader, test_stat_file)

# Prepare Ndx Object
if not os.path.isfile(ndx_file):
    models = enrol_obj.modelset
    testsegs = test_obj.modelset

    logger.info("Preparing Ndx")
    ndx_obj = Ndx(models=models, testsegs=testsegs)
    logger.info("Saving ndx obj...")
    ndx_obj.save_ndx_object(ndx_file)
else:
    logger.info("Skipping Ndx preparation")
    logger.info("Loading Ndx from disk")
    with open(ndx_file, "rb") as input:
        ndx_obj = pickle.load(input)


logger.info("PLDA scoring...")
scores_plda = fast_PLDA_scoring(
    enrol_obj,
    test_obj,
    ndx_obj,
    params.compute_plda.mean,
    params.compute_plda.F,
    params.compute_plda.Sigma,
)

# Positive and Negative scores
positive_scores = []
negative_scores = []

gt_file = os.path.join(params.data_folder, "meta", "veri_test.txt")

for line in open(gt_file):
    lab = int(line.split(" ")[0].rstrip().split(".")[0].strip())
    enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
    test_id = line.split(" ")[2].rstrip().split(".")[0].strip()

    # Assuming enrol_id and test_id are unique
    i = int(numpy.where(scores_plda.modelset == enrol_id)[0][0])
    j = int(numpy.where(scores_plda.segset == test_id)[0][0])

    s = float(scores_plda.scoremat[i, j])

    if lab == 1:
        positive_scores.append(s)
    else:
        negative_scores.append(s)


logger.info("Computing EER... ")
del scores_plda
del xvectors_stat
del enrol_obj
del test_obj
eer = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
logger.info("EER=%f", eer)
