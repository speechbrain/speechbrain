#!/usr/bin/python
import os
import sys
import speechbrain as sb
import torch
import torchaudio
import numpy as np
from tqdm.contrib import tqdm
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_average
from speechbrain.processing.features import spectral_magnitude
from joblib import Parallel, delayed
from pystoi.stoi import stoi
from pesq import pesq

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from voicebank_prepare import prepare_voicebank  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
)


def read_pesq(clean_folder, enhanced_file, sr):
    wave_name = enhanced_file.split("/")[-1]
    clean_file = clean_folder + wave_name

    clean_wav, _ = torchaudio.load(clean_file)
    enhanced_wav, _ = torchaudio.load(enhanced_file)

    pesq_score = pesq(
        sr,
        np.squeeze(clean_wav.numpy()),
        np.squeeze(enhanced_wav.numpy()),
        "wb",
    )
    return pesq_score


# Parallel computing for accelerating
def read_batch_PESQ(clean_folder, enhanced_list):
    pesq_score = Parallel(n_jobs=30)(
        delayed(read_pesq)(clean_folder, en, 16000) for en in enhanced_list
    )
    return pesq_score


def read_STOI(clean_folder, enhanced_file):
    wave_name = enhanced_file.split("/")[-1]
    clean_file = clean_folder + wave_name

    clean_wav, _ = torchaudio.load(clean_file)
    enhanced_wav, _ = torchaudio.load(enhanced_file)

    stoi_score = stoi(
        np.squeeze(clean_wav.numpy()),
        np.squeeze(enhanced_wav.numpy()),
        16000,
        extended=False,
    )
    return stoi_score


# Parallel computing for accelerating
def read_batch_STOI(clean_folder, enhanced_list):
    stoi_score = Parallel(n_jobs=30)(
        delayed(read_STOI)(clean_folder, en) for en in enhanced_list
    )
    return stoi_score


def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


if params.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    tensorboard_train_logger = TensorboardLogger(params.tensorboard_logs)

# Create the folder to save enhanced files
if not os.path.exists(params.enhanced_folder):
    os.mkdir(params.enhanced_folder)


class SEBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, lens = x
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        feats = params.compute_STFT(wavs)
        feats = spectral_magnitude(feats, power=0.5, log=True, eps=1)
        # feats = params.mean_var_norm(feats, lens)

        mask = params.model(feats, init_params=init_params)
        out = torch.mul(mask, feats)

        return out

    def compute_objectives(self, predictions, targets, stage="train"):
        ids, wavs, lens = targets
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        feats = params.compute_STFT(wavs)
        feats = spectral_magnitude(feats, power=0.5, log=True, eps=1)

        loss = params.compute_cost(predictions, feats, lens)

        stats = {}
        """
        if stage != "train":
            stats["loss"] = loss
        """
        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        if params.use_tensorboard:
            tensorboard_train_logger.log_stats(
                {"Epoch": epoch}, train_stats, valid_stats
            )

        params.train_logger.log_stats(
            {"Epoch": epoch}, train_stats, valid_stats
        )

        loss = summarize_average(valid_stats["loss"])
        params.checkpointer.save_and_keep_only(
            meta={"loss": loss},
            importance_keys=[ckpt_recency, lambda c: -c.meta["loss"]],
        )

    def generate_enhanced_waveform(self, data_set):
        with tqdm(data_set) as t:
            for i, batch in enumerate(t):
                inputs, _ = batch
                predict_mag = torch.exp(self.compute_forward(inputs)) - 1
                ids, noisy_wavs, lens = inputs
                noisy_wavs, lens = (
                    noisy_wavs.to(params.device),
                    lens.to(params.device),
                )
                noisy_stft = params.compute_STFT(noisy_wavs)
                noisy_phase = torch.atan2(
                    noisy_stft[:, :, :, 1], noisy_stft[:, :, :, 0]
                )

                real, imag = torch.cos(noisy_phase), torch.sin(noisy_phase)
                enhanced_stft = torch.mul(
                    torch.unsqueeze(predict_mag, -1),
                    torch.cat(
                        (torch.unsqueeze(real, -1), torch.unsqueeze(imag, -1)),
                        -1,
                    ),
                )
                enhanced_wave = params.compute_ISTFT(
                    enhanced_stft.to("cpu"), noisy_wavs.shape[1]
                )

                torchaudio.save(
                    os.path.join(params.enhanced_folder, ids[0] + ".wav"),
                    enhanced_wave.to("cpu").detach(),
                    16000,
                )


# Prepare data
prepare_voicebank(
    data_folder=params.data_folder, save_folder=params.data_folder,
)
train_set = params.train_loader()
valid_set = params.valid_loader()
test_set = params.test_loader()
first_x, first_y = next(iter(train_set))

se_brain = SEBrain(
    modules=[params.model], optimizer=params.optimizer, first_inputs=[first_x],
)

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
se_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
params.checkpointer.recover_if_possible(lambda c: -c.meta["loss"])
test_stats = se_brain.evaluate(params.test_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)


# Generate enhanced waveform on the test set
se_brain.generate_enhanced_waveform(test_set)

# evaluate the PESQ and STOI scores of the enhanced speech
STOI_scores = read_batch_STOI(
    params.test_clean_folder, get_filepaths(params.enhanced_folder)
)
print("STOI score: %.5f" % np.mean(STOI_scores))

PESQ_scores = read_batch_PESQ(
    params.test_clean_folder, get_filepaths(params.enhanced_folder)
)
print("PESQ score: %.5f" % np.mean(PESQ_scores))
