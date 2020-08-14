#!/usr/bin/python
import os
import torch
import speechbrain as sb
import speechbrain.processing.NMF as sb_nmf
from nmf_brain import NMF_Brain
from speechbrain.processing.features import spectral_magnitude

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/sourcesep_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

sb.core.create_experiment_directory(
    experiment_directory=hyperparams.output_folder,
    hyperparams_to_save=hyperparams_file,
)
torch.manual_seed(0)

NMF1 = NMF_Brain(hyperparams.train_loader1(), hyperparams)

print("fitting model 1")
NMF1.fit(
    train_set=hyperparams.train_loader1(),
    valid_set=None,
    epoch_counter=range(hyperparams.N_epochs),
    progressbar=False,
)
W1hat = NMF1.training_out[1]

NMF2 = NMF_Brain(hyperparams.train_loader2(), hyperparams)

print("fitting model 2")
NMF2.fit(
    train_set=hyperparams.train_loader2(),
    valid_set=None,
    epoch_counter=range(hyperparams.N_epochs),
    progressbar=False,
)
W2hat = NMF2.training_out[1]

# separate
mixture_loader = hyperparams.test_loader()
Xmix = list(mixture_loader)[0]

Xmix = hyperparams.compute_features(Xmix[0][1])
Xmix_mag = spectral_magnitude(Xmix, power=2)

X1hat, X2hat = sb_nmf.NMF_separate_spectra([W1hat, W2hat], Xmix_mag)

sb_nmf.reconstruct_results(
    X1hat,
    X2hat,
    Xmix.permute(0, 2, 1, 3),
    hyperparams.sample_rate,
    hyperparams.win_length,
    hyperparams.hop_length,
    use_absolute_path=False,
    copy_original_files=True,
    datapath="samples/audio_samples/sourcesep_samples/",
)
