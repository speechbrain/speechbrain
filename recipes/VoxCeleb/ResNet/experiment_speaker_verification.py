#!/usr/bin/python
import os
import sys
import math
import torch
import logging
import speechbrain as sb
import torch.nn.functional as F
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
    hyperparams_to_save=params_file,
    overrides=overrides,
)

# Prepare data from dev of Voxceleb1
prepare_voxceleb(
    data_folder=params.data_folder,
    save_folder=params.save_folder,
    splits=["test"],
    rand_seed=params.seed,
    source=params.voxceleb_source
    if hasattr(params, "voxceleb_source")
    else None,
)

# Cosine similarity initialization
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

# Data loaders
test_set = params.test_loader()
index2label = params.test_loader.label_dict["lab_verification"]["index2lab"]


# Definition of the steps for resnet computation from the waveforms
# Note that ResNet models are trained with fixed segments w/o padding, so
# variable-length segments w/ zero-padding are not generalized well.
# Therefore, compute embeddings for each wavs.
# TODO(hwidong.na): support batch size > 1?
def compute_embedding(wavs, lens, init_params=False):
    with torch.no_grad():
        assert wavs.shape[0] == 1 and len(wavs.shape) == 2
        wavs = wavs.squeeze(0)
        audiosize = wavs.shape[0]
        max_audio = params.max_frames * 160 + 240
        if audiosize <= max_audio:
            shortage = math.floor((max_audio - audiosize + 1) / 2)
            wavs = F.pad(wavs, (shortage, shortage), "constant", value=0)
            audiosize = wavs.shape[0]

        if params.max_frames == 0:
            feats = wavs.unsqueeze(0)
        else:
            startframe = torch.linspace(
                0, audiosize - max_audio, params.num_eval
            )
            feats = torch.zeros([params.num_eval, max_audio])
            for i, asf in enumerate(startframe):
                feats[i] = wavs[int(asf) : int(asf) + max_audio]
            lens = lens.repeat(params.num_eval)
        feats = feats.to(params.device)
        feats = params.compute_features(feats, init_params=init_params)
        feats = params.mean_var_norm(feats, lens)
        feats = feats.unsqueeze(1)
        emb = params.resnet_model(feats, init_params=init_params)
        emb = params.mean_var_norm_emb(
            emb, torch.ones(emb.shape[0]).to(params.device)
        )
    return emb


# Function for pre-trained model downloads
def download_and_pretrain():
    save_model_path = params.output_folder + "/save/resnet.ckpt"
    download_file(params.resnet_file, save_model_path)
    params.resnet_model.load_state_dict(
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
            emb = compute_embedding(wav1, lens1, init_params=True)
            params.mean_var_norm_emb.glob_mean = torch.zeros_like(emb[0, 0, :])
            params.mean_var_norm_emb.count = 0

            # Download models from the web if needed
            if "https://" in params.resnet_file:
                download_and_pretrain()
            else:
                params.resnet_model.load_state_dict(
                    torch.load(params.resnet_file), strict=True
                )

            init_params = False
            params.resnet_model.eval()

        # Enrolment and test resnets
        emb1 = compute_embedding(wav1, lens1)
        emb2 = compute_embedding(wav2, lens2)

        # Computing similarity
        score = similarity(emb1, emb2).mean(dim=0)

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
