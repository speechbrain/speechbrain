#!/usr/bin/python
import os
import sys
import torch
import logging
import speechbrain as sb
from tqdm.contrib import tqdm
from speechbrain.utils.EER import EER
from speechbrain.utils.data_utils import download_file
from speechbrain.processing.PLDA import StatObject_SB, PLDA

# We should add the code here (now it is just a copy of the verification one)


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
prepare_voxceleb(
    data_folder=params.data_folder,
    save_folder=params.save_folder,
    splits=["dev", "test"],
    rand_seed=params.seed,
)

# Cosine similarity initialization
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

# Data loaders
train_set = params.train_loader()
test_set = params.test_loader()
index2label = params.test_loader.label_dict["lab_verification"]["index2lab"]


# Definition of the steps for xvector computation from the waveforms
def compute_x_vectors(wavs, lens, init_params=False):
    with torch.no_grad():
        wavs = wavs.to(params.device)
        feats = params.compute_features(wavs, init_params=init_params)
        feats = params.mean_var_norm(feats, lens)
        x_vect = params.xvector_model(feats, lens=lens, init_params=init_params)
        # x_vect = params.mean_var_norm_xvect(
        #     x_vect, torch.ones(x_vect.shape[0]).to("cuda:0")
        # )
    return x_vect


# Function for pre-trained model downloads
def download_and_pretrain():
    save_model_path = params.output_folder + "/save/xvect.ckpt"
    download_file(params.xvector_file, save_model_path)
    params.xvector_model.load_state_dict(
        torch.load(save_model_path), strict=True
    )


# computing xvectors on the training dataset (to train PLDA)
logger.info("Extracting xvectors...")

tr_xvectors = []
tr_ids = []

with tqdm(train_set, dynamic_ncols=True) as t:

    init_params = True

    for wav, spk_id in t:
        id, wav, lens = wav
        id, spk_id, _ = spk_id

    # Initialize the model and perform pre-training
    if init_params:
        xvect = compute_x_vectors(wav, lens, init_params=True)
        # Download models from the web if needed
        if "https://" in params.xvector_file:
            download_and_pretrain()
        else:
            params.xvector_model.load_state_dict(
                torch.load(params.xvector_file), strict=True
            )

        init_params = False
        params.xvector_model.eval()

    # computing xvector
    xvect = compute_x_vectors(wav, lens)

    # Add xvectors and snt_id in a lst
    tr_xvectors.append(xvect)
    tr_ids.append(id)


# Xvectors concatenation
tr_xvectors = torch.stack(tr_xvectors)

# Creating StatObject_SB (i.e. the data structure needed for PLDA training)
# Nauman, please add the other fields
train_obj = StatObject_SB(stat1=tr_xvectors)


# Maybe we save this structure on disk. This way we can avoid computing
# xvectors every time we run this script.
# Train the model
plda = PLDA()
plda.plda(train_obj)


# Loop over all test sentences
with tqdm(test_set, dynamic_ncols=True) as t:

    positive_scores = []
    negative_scores = []
    init_params = True

    for wav1, wav2, label_verification in t:
        id, wav1, lens1 = wav1
        id, wav2, lens2 = wav2
        id, label_verification, _ = label_verification

        # Initialize the model and perform pre-training
        if init_params:
            xvect1 = compute_x_vectors(wav1, lens1, init_params=True)
            params.mean_var_norm_xvect.glob_mean = torch.zeros_like(
                xvect1[0, 0, :]
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
        xvect1 = compute_x_vectors(wav1, lens1)
        xvect2 = compute_x_vectors(wav2, lens2)

        # Computing similarity
        score = similarity(xvect1, xvect2)

        # Adding score to positive or negative lists
        for i in range(len(label_verification)):

            logger.debug(
                "%s score=%f label=%s"
                % (id[i], score[i], index2label[int(label_verification[i])])
            )
            if index2label[int(label_verification[i])] == "1":
                positive_scores.append(score[i])
            else:
                negative_scores.append(score[i])

    # Compute the EER
    eer = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    logger.info("EER=%f", eer)
