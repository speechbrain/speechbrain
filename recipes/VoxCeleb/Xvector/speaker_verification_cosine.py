#!/usr/bin/python
import os
import sys
import torch
import logging
import speechbrain as sb
from tqdm.contrib import tqdm
from speechbrain.utils.EER import EER
from speechbrain.utils.data_utils import download_file


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
    splits=["test"],
    rand_seed=params.seed,
)

# Cosine similarity initialization
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

# Data loaders
test_set = params.test_loader()
index2label = params.test_loader.label_dict["lab_verification"]["index2lab"]


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
        print(eer)
    logger.info("EER=%f", eer)
