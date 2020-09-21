#!/usr/bin/python
import os
import sys
import torch
import logging
import speechbrain as sb
from tqdm.contrib import tqdm
from speechbrain.utils.EER import EER
from speechbrain.utils.data_utils import download_file


# Compute embeddings from the waveforms
def compute_embedding(wavs, lens, init_params=False):
    with torch.no_grad():
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        feats = params.compute_features(wavs, init_params=init_params)
        feats = params.mean_var_norm(feats, lens)
        emb = params.embedding_model(feats, init_params=init_params)
        emb = params.mean_var_norm_emb(
            emb, torch.ones(emb.shape[0]).to(params.device)
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
            wavs, lens = wavs.to(params.device), lens.to(params.device)
            emb = compute_embedding(wavs, lens)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict


def get_verification_scores(veri_test):
    """ computes positive and negative scores given the verification split.
    """
    scores = []
    labs = []
    positive_scores = []
    negative_scores = []
    cnt = 0

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # Loop over all the verification tests
    scores = []
    for i, line in enumerate(veri_test):

        # Reading verification file (enrol_file test_file label)
        labs.append(int(line.split(" ")[0].rstrip().split(".")[0].strip()))
        enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
        test_id = line.split(" ")[2].rstrip().split(".")[0].strip()
        enrol = enrol_dict[enrol_id]
        test = test_dict[test_id]
        scores.append(similarity(enrol, test)[0])

        # Gathering batches
        if cnt == params.batch_size - 1 or i == len(veri_test) - 1:
            # Putting scores in the corresponding lists
            for j, score in enumerate(scores):
                if labs[j] == 1:
                    positive_scores.append(score)
                else:
                    negative_scores.append(score)
            scores = []
            labs = []
            cnt = 0
            continue
        cnt = cnt + 1
    return positive_scores, negative_scores


# Function for pre-trained model downloads
def download_and_pretrain():
    """ Downloads the specified pre-trained model
    """
    if "http" in params.embedding_file:
        save_model_path = params.output_folder + "/save/embedding_model.ckpt"
        download_file(params.embedding_file, save_model_path)
    else:
        save_model_path = params.embedding_file
    params.embedding_model.load_state_dict(
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

    # Dictionary to store the last waveform read for each speaker
    wav_stored = {}

    # Data loaders
    enrol_set_loader = params.enrol_loader()
    test_set_loader = params.test_loader()

    # init params
    seg_ids, wavs, lens = next(iter(test_set_loader))[0]
    wavs, lens = wavs.to(params.device), lens.to(params.device)
    emb = compute_embedding(wavs, lens, init_params=True)
    params.mean_var_norm_emb.glob_mean = torch.zeros_like(emb[0, 0, :])
    params.mean_var_norm_emb.count = 0
    params.embedding_model.eval()
    download_and_pretrain()

    # Computing  enrollment and test embeddings
    print("Computing enroll/test embeddings...")
    enrol_dict = compute_embedding_loop(enrol_set_loader)
    test_dict = compute_embedding_loop(test_set_loader)
    # Compute the EER
    print("Computing EER..")
    # Reading standard verification split
    gt_file = os.path.join(params.data_folder, "meta", "veri_test.txt")
    with open(gt_file) as f:
        veri_test = [line.rstrip() for line in f]

    positive_scores, negative_scores = get_verification_scores(veri_test)
    del enrol_dict, test_dict

    eer = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    logger.info("EER=%f", eer * 100)
