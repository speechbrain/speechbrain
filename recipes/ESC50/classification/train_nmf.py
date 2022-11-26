import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from esc50_prepare import prepare_esc50
from train import dataio_prep
from sklearn.metrics import confusion_matrix
import numpy as np
from confusion_matrix_fig import create_cm_fig

import scipy.io.wavfile as wavf
from tqdm import tqdm
import matplotlib.pyplot as plt
from librosa.display import specshow
from drawnow import drawnow, figure
from speechbrain.processing.features import spectral_magnitude


def draw_fig():
    plt.subplot(211)
    specshow(Xhat.data.cpu().numpy())

    plt.subplot(212)
    specshow(Xs.squeeze().data.cpu().numpy())

    plt.savefig("nmf_results.png", format="png")


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    run_on_main(
        prepare_esc50,
        kwargs={
            "data_folder": hparams["data_folder"],
            "audio_data_folder": hparams["audio_data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "train_fold_nums": hparams["train_fold_nums"],
            "valid_fold_nums": hparams["valid_fold_nums"],
            "test_fold_nums": hparams["test_fold_nums"],
            "skip_manifest_creation": hparams["skip_manifest_creation"],
        },
    )

    datasets, _ = dataio_prep(hparams)
    nmf_model = hparams["nmf"].to(hparams["device"])
    nmf_encoder = hparams["nmf_encoder"].to(hparams["device"])
    opt = torch.optim.Adam(
        lr=1e-4,
        params=list(nmf_encoder.parameters()) + list(nmf_model.parameters()),
    )

    for e in range(100):
        for i, element in enumerate(datasets["train"]):
            # print(element["sig"].shape[0] / hparams["sample_rate"])

            opt.zero_grad()
            Xs = hparams["compute_stft"](
                element["sig"].unsqueeze(0).to(hparams["device"])
            )
            Xs = hparams["compute_stft_mag"](Xs)
            Xs = torch.log(Xs + 1).permute(0, 2, 1)
            z = nmf_encoder(Xs)

            Xhat = torch.matmul(nmf_model.return_W("torch"), z.squeeze())
            loss = ((Xs.squeeze() - Xhat) ** 2).mean()
            loss.backward()

            opt.step()
            if 1:
                if i in [100]:
                    draw_fig()
        print("loss is {}, epoch is {} ".format(loss.item(), e))

    torch.save(nmf_model.return_W("torch"), "nmf_decoder.pt")
