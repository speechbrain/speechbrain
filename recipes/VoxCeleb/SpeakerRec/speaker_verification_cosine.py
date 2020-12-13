#!/usr/bin/python3
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml
Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
"""
import os
import sys
import torch
import logging
import speechbrain as sb
from tqdm.contrib import tqdm
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file


# Compute embeddings from the waveforms
def compute_embedding(wavs, lens):
    """Computes the embeddings of the input waveform batch
    """
    with torch.no_grad():
        wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, lens)
        emb = params["embedding_model"](feats, lens)
        emb = params["mean_var_norm_emb"](
            emb, torch.ones(emb.shape[0]).to(params["device"])
        )
    return emb


def compute_embedding_loop(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        for (batch,) in tqdm(data_loader, dynamic_ncols=True):
            seg_ids, wavs, lens = batch
            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
            emb = compute_embedding(wavs, lens)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict


def get_verification_scores(veri_test):
    """ Computes positive and negative scores given the verification split.
    """
    scores = []
    positive_scores = []
    negative_scores = []

    save_file = os.path.join(params["output_folder"], "scores.txt")
    s_file = open(save_file, "w")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # creating cohort for score normalization
    if "score_norm" in params:
        train_cohort = torch.stack(list(train_dict.values()))

    for i, line in enumerate(veri_test):

        # Reading verification file (enrol_file test_file label)
        lab_pair = int(line.split(" ")[0].rstrip().split(".")[0].strip())
        enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
        test_id = line.split(" ")[2].rstrip().split(".")[0].strip()
        enrol = enrol_dict[enrol_id]
        test = test_dict[test_id]

        if "score_norm" in params:
            # Getting norm stats for enrol impostors
            enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
            score_e_c = similarity(enrol_rep, train_cohort)
            mean_e_c = torch.mean(score_e_c, dim=0)
            std_e_c = torch.std(score_e_c, dim=0)

            # Getting norm stats for test impostors
            test_rep = test.repeat(train_cohort.shape[0], 1, 1)
            score_t_c = similarity(test_rep, train_cohort)
            mean_t_c = torch.mean(score_t_c, dim=0)
            std_t_c = torch.std(score_t_c, dim=0)

        # Compute the score for the given sentence
        score = similarity(enrol, test)[0]

        # Perform score normalization
        if "score_norm" in params:
            if params["score_norm"] == "z-norm":
                score = (score - mean_e_c) / std_e_c
            elif params["score_norm"] == "t-norm":
                score = (score - mean_t_c) / std_t_c
            elif params["score_norm"] == "s-norm":
                score = (score - mean_e_c) / std_e_c
                score += (score - mean_t_c) / std_t_c
                score = 0.5 * score

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score))
        scores.append(score)

        if lab_pair == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    s_file.close()
    return positive_scores, negative_scores


# Function for pre-trained model downloads
def download_and_pretrain():
    """ Downloads the specified pre-trained model
    """
    if "http" in params["embedding_file"]:
        save_model_path = params["output_folder"] + "/save/embedding_model.ckpt"
        download_file(params["embedding_file"], save_model_path)
    else:
        save_model_path = params["embedding_file"]
    params["embedding_model"].load_state_dict(
        torch.load(save_model_path), strict=True
    )


if __name__ == "__main__":
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
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    prepare_voxceleb(
        data_folder=params["data_folder"],
        save_folder=params["save_folder"],
        splits=["train", "dev", "test"] if "score_norm" in params else ["test"],
        split_ratio=[90, 10],
        seg_dur=300,
        rand_seed=params["seed"],
        source=params["voxceleb_source"]
        if "voxceleb_source" in params
        else None,
    )

    # Dictionary to store the last waveform read for each speaker
    wav_stored = {}

    # Data loaders
    if "train_loader" in params:
        train_set_loader = params["train_loader"]().get_dataloader()
    enrol_set_loader = params["enrol_loader"]().get_dataloader()
    test_set_loader = params["test_loader"]().get_dataloader()

    # Pretrain the model (download it if needed)
    download_and_pretrain()
    params["embedding_model"].eval()

    # Putting models on the specified device
    params["compute_features"].to(params["device"])
    params["mean_var_norm"].to(params["device"])
    params["embedding_model"].to(params["device"])
    params["mean_var_norm_emb"].to(params["device"])

    # Computing  enrollment and test embeddings
    print("Computing enroll/test embeddings...")

    # First run
    enrol_dict = compute_embedding_loop(enrol_set_loader)
    test_dict = compute_embedding_loop(test_set_loader)

    # Second run (normalization stats are more stable)
    enrol_dict = compute_embedding_loop(enrol_set_loader)
    test_dict = compute_embedding_loop(test_set_loader)

    if "score_norm" in params:
        train_dict = compute_embedding_loop(train_set_loader)

    # Compute the EER
    print("Computing EER..")
    # Reading standard verification split
    gt_file = os.path.join(params["data_folder"], "meta", "veri_test.txt")
    with open(gt_file) as f:
        veri_test = [line.rstrip() for line in f]

    positive_scores, negative_scores = get_verification_scores(veri_test)
    del enrol_dict, test_dict

    eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    logger.info("EER=%f", eer * 100)

    min_dcf, th = minDCF(
        torch.tensor(positive_scores), torch.tensor(negative_scores)
    )
    logger.info("minDCF=%f", min_dcf * 100)
