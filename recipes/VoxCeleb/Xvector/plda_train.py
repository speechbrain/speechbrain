#!/usr/bin/python
import os
import sys
import torch
import logging
import speechbrain as sb
import numpy
import pickle
import time  # noqa F401

from tqdm.contrib import tqdm
from speechbrain.utils.EER import EER  # noqa F401
from speechbrain.utils.data_utils import download_file
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.processing.PLDA_LDA import PLDA
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
    params_to_save=params_file,
    overrides=overrides,
)

# Prepare data from dev of Voxceleb1
"""
# Code for train-valid split needs update
# Check this later
prepare_voxceleb(
    data_folder=params.train_data_folder,
    save_folder=params.save_folder,
    splits=["train", "dev"],
    split_ratio=[90, 10],
    seg_dur=300,
    vad=False,
    rand_seed=params.seed,
)
"""


# Prepare data from test Voxceleb1
prepare_voxceleb(
    data_folder=params.test_data_folder,
    save_folder=params.save_folder,
    splits=["test"],
    rand_seed=params.seed,
)


# Cosine similarity initialization
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

# Data loaders
test_set = params.test_loader()
# index2label = params.test_loader.label_dict["lab_verification"]["index2lab"]


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


# Some PLDA inputs
modelset, segset = [], []
xvectors = numpy.empty(shape=[0, 512], dtype=numpy.float64)

# Train set
train_set = params.train_loader()
ind2lab = params.train_loader.label_dict["spk_id"]["index2lab"]


# Get Xvectors for train data
xv_file = os.path.join(
    params.save_folder, "VoxCeleb1_train_xvectors_stat_obj.pkl"
)
# skip extraction if already extracted
if not os.path.exists(xv_file):
    print("Extracting xvectors from Training set..")
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
                params.classifier.eval()
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
    # Save TRAINING Xvectors in StatObject_SB object and load them during PLDA scoring
    xvectors_stat.save_stat_object(xv_file)
else:
    # Load the saved stat object for train xvector
    print("Skipping Xvector Extraction for training set")
    print("Loading previously saved stat_object for train xvectors..")
    with open(xv_file, "rb") as input:
        xvectors_stat = pickle.load(input)

plda = PLDA()

# Training Gaussina PLDA model
print("Training PLDA model")
plda.plda(xvectors_stat)

print("\nTesting...")

# Some PLDA inputs


modelset_test, segset_test = [], []
xvect_test = numpy.empty(shape=[0, 512], dtype=numpy.float64)

# verification_truth = []

# Enroll and Test xvector
enrol_stat_file = os.path.join(params.save_folder, "stat_enrol.pkl")
test_stat_file = os.path.join(params.save_folder, "stat_test.pkl")
ndx_file = os.path.join(params.save_folder, "ndx.pkl")


# ENROL
enrol_set = params.enrol_loader()
xvect_enrol = numpy.empty(shape=[0, 512], dtype=numpy.float64)
modelset_enrol = []
segset_enrol = []
# Extract xvectors for Enrolment set
if not os.path.isfile(enrol_stat_file):
    init_params = True
    with tqdm(enrol_set, dynamic_ncols=True) as t:
        # with tqdm(test_set, dynamic_ncols=True) as t:
        # for wav1, wav2, label_verification in t:
        for wav in t:
            ids, wavs, lens = wav[0]
            mod = [x for x in ids]
            seg = [x for x in ids]
            modelset_enrol = modelset_enrol + mod
            segset_enrol = segset_enrol + seg
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
                params.classifier.eval()

            # Enrolment and test xvectors
            xvects = compute_x_vectors(wavs, lens)
            xv_enrol = xvects.squeeze().cpu().numpy()
            xvect_enrol = numpy.concatenate((xvect_enrol, xv_enrol), axis=0)

    modelset_enrol = numpy.array(modelset_enrol, dtype="|O")
    segset_enrol = numpy.array(segset_enrol, dtype="|O")

    # intialize variables for start, stop and stat0
    s = numpy.array([None] * xvect_enrol.shape[0])
    b = numpy.array([[1.0]] * xvect_enrol.shape[0])

    enrol_obj = StatObject_SB(
        modelset=modelset_enrol,
        segset=segset_enrol,
        start=s,
        stop=s,
        stat0=b,
        stat1=xvect_enrol,
    )
    print("Saving enrol xvectors...")
    enrol_obj.save_stat_object(enrol_stat_file)

else:
    print("Skipping Xvector Extraction for enrol set")
    print("Loading previously saved stat_object for enrol xvectors..")

    with open(enrol_stat_file, "rb") as input:
        enrol_obj = pickle.load(input)


# TEST
test_set = params.test_loader()
xvect_test = numpy.empty(shape=[0, 512], dtype=numpy.float64)
modelset_test = []
segset_test = []
# Extract xvectors for Enrolment set
if not os.path.isfile(test_stat_file):
    init_params = True
    print("Extracting xvectors for test set")
    with tqdm(test_set, dynamic_ncols=True) as t:
        # with tqdm(test_set, dynamic_ncols=True) as t:
        # for wav1, wav2, label_verification in t:
        for wav in t:
            ids, wavs, lens = wav[0]
            mod = [x for x in ids]
            seg = [x for x in ids]
            modelset_test = modelset_test + mod
            segset_test = segset_test + seg
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
                params.classifier.eval()

            # Enrolment and test xvectors
            xvects = compute_x_vectors(wavs, lens)
            xv_test = xvects.squeeze().cpu().numpy()
            xvect_test = numpy.concatenate((xvect_test, xv_test), axis=0)

    modelset_test = numpy.array(modelset_test, dtype="|O")
    segset_test = numpy.array(segset_test, dtype="|O")

    # intialize variables for start, stop and stat0
    s = numpy.array([None] * xvect_test.shape[0])
    b = numpy.array([[1.0]] * xvect_test.shape[0])

    test_obj = StatObject_SB(
        modelset=modelset_test,
        segset=segset_test,
        start=s,
        stop=s,
        stat0=b,
        stat1=xvect_test,
    )
    print("Saving test stat obj xvectors...")
    test_obj.save_stat_object(test_stat_file)

else:
    print("Skipping Xvector Extraction for test set")
    print("Loading previously saved stat_object for test xvectors..")

    with open(test_stat_file, "rb") as input:
        test_obj = pickle.load(input)


# Prepare Ndx Object
if not os.path.isfile(ndx_file):
    models = enrol_obj.modelset
    testsegs = test_obj.modelset

    # Current numpy is strict with boolean dimension mismatch
    # This hack is needed to make it compatible with numpy (>=1.17.0)
    d = models.shape[0] - testsegs.shape[0]
    print("d: ", d)
    if d != 0:
        print("here")
        if d > 0:
            print("TEST")
            pad = testsegs[-d:]
            testsegs = numpy.concatenate((testsegs, pad), axis=0)
        else:
            print("MMMD")
            pad = models[-d:]
            models = numpy.concatenate((models, pad), axis=0)

    print("\n\nPreparing Ndx file")
    a = time.time()
    ndx_obj = Ndx(models=models, testsegs=testsegs)
    print("NDX time: ", time.time() - a)
    print("Saving ndx obj...")
    ndx_obj.save_ndx_object(ndx_file)
else:
    print("Skipping Ndx preparation")
    print("Loading Ndx from disk")
    with open(ndx_file, "rb") as input:
        ndx_obj = pickle.load(input)


print("Started PLDA scoring...")
scores_plda = fast_PLDA_scoring(
    enrol_obj, test_obj, ndx_obj, plda.mean, plda.F, plda.Sigma
)
print("Completed PLDA scoring...")

# Assuming same number of enrol and test (as it is in voxceleb verification split)
# print("Getting diagonal elements...")
# scores = scores_plda.scoremat.diagonal()

positive_scores = []
negative_scores = []

"""
print ("PLDA SCORES: ", scores_plda.scoremat.shape,  scores_plda.modelset.shape, scores_plda.segset.shape)
print ("mod: ", scores_plda.modelset)
print ("seg: ", scores_plda.segset)
print ("PLDA score mat:")
print (scores_plda.scoremat)
"""


print("COMPLETE")
sys.exit()

"""
# Assuming same number of enrol and test (as it is in voxceleb verification split)
print("Getting diagonal elements...")
scores = scores_plda.scoremat.diagonal()

positive_scores = []
negative_scores = []

# Adding score to positive or negative lists
print("Preparing positive and negative scores")
for i in range(len(label_verification)):
    if int(label_verification[i]) == 1:
        positive_scores.append(scores[i])
    else:
        negative_scores.append(scores[i])

# Normalize scores
print("Normalizing PLDA scores")
pmin = min(positive_scores)
pmax = max(negative_scores)
nmin = min(negative_scores)
nmax = max(negative_scores)
print("here")
positive_scores = (positive_scores - pmin) / (pmax - pmin)
print("Normal...... ")
negative_scores = (negative_scores - nmin) / (nmax - nmin)

# Compute the EER
print("Computing EER... ")
eer = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
logger.info("EER=%f", eer)
"""
