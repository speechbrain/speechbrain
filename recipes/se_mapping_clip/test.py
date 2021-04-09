#!/usr/bin/env/python3
"""Recipe for training a speech enhancement system with spectral masking.

To run this recipe, do the following:
> python train.py train.yaml --data_folder /path/to/save/mini_librispeech

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

Authors
 * Szu-Wei Fu 2020
 * Chien-Feng Liao 2020
 * Peter Plantinga 2021
"""
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mini_librispeech_prepare import prepare_mini_librispeech
from train import SEBrain


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    print(hparams_file)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Initialize the Brain object to prepare for mask training.
    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    wav = 'wav/49_3.wav'

    noisy_wavs = sb.dataio.dataio.read_audio(wav)
    noisy_wavs = noisy_wavs.reshape(1, -1)
    print(noisy_wavs.shape)

    noisy_feats, noisy_feats_contex = se_brain.compute_feats(noisy_wavs)

    print(noisy_feats.shape)

    se_brain.on_evaluate_start(max_key="stoi")
    se_brain.on_stage_start(sb.Stage.TEST, epoch=None)

    se_brain.modules.eval()

    noisy_feats = noisy_feats.to(se_brain.device)
    noisy_wavs = noisy_wavs.to(se_brain.device)
    noisy_feats_contex = noisy_feats_contex.to(se_brain.device)

    # Masking is done here with the "signal approximation (SA)" algorithm.
    # The masked input is compared directly with clean speech targets.
    mask = se_brain.modules.model(noisy_feats_contex)
    predict_spec = torch.mul(mask, noisy_feats)
    # predict_spec = mask
    print("predict_spec:{}".format(predict_spec.shape))
    print("predict_spec:{}".format(predict_spec.device))
    print("noisy_wavs:{}".format(noisy_wavs.shape))
    print("noisy_wavs:{}".format(noisy_wavs.device))

    # Also return predicted wav, for evaluation. Note that this could
    # also be used for a time-domain loss term.
    predict_wav = se_brain.hparams.resynth(
        torch.expm1(predict_spec), noisy_wavs
    )

    print(predict_wav.shape)
    sb.dataio.dataio.write_audio("wav/output/49_3_enh4.wav", torch.squeeze(predict_wav.to('cpu')), 16000)


