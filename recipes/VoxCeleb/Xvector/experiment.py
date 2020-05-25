#!/usr/bin/python
import os
import sys  # noqa F401
import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.sequential import Sequential
from speechbrain.utils.train_logger import (
    FileTrainLogger,
    summarize_average,
)

# To import data preparation script from ../
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from voxceleb1_prepare import VoxCelebPreparer  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
if "seed" in overrides:
    torch.manual_seed(overrides["seed"])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Creating directory for experiments
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder, params_to_save=params_file,
)

# Logger during training
train_logger = FileTrainLogger(
    save_file=params.train_log, summary_fns={"loss": summarize_average},
)

# Checkpointer
checkpointer = sb.utils.checkpoints.Checkpointer(
    checkpoints_dir=params.save_folder,
    recoverables={
        "model": params.model,
        "optimizer": params.optimizer,
        "normalizer": params.mean_var_norm,
        "counter": params.epoch_counter,
    },
)

# Data preparation
data_prepare = VoxCelebPreparer(
    data_folder=params.data_folder,
    splits=params.splits,
    save_folder=params.save_folder,
    seg_dur=params.seg_dur,
    vad=params.vad,
    rand_seed=params.seed,
)
data_prepare()


# Trains xvector model
class XvectorBrain(sb.core.Brain):
    def compute_forward(self, x, train_mode=True, init_params=False):
        id, wavs, lens = x

        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        x = params.model(feats, init_params)
        x = params.linear(x, init_params)

        outputs = params.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, train_mode=True):
        predictions, lens = predictions
        uttid, spkid, _ = targets

        loss = params.compute_cost(predictions, spkid, lens)

        if not train_mode:
            stats = {"error": params.compute_error(predictions, spkid, lens)}
            return loss, stats

        return loss

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid error: %.2f" % summarize_average(valid_stats["error"]))
        epoch_stats = {"epoch": epoch, "lr": params.lr}
        train_logger.log_stats(epoch_stats, train_stats)
        checkpointer.save_and_keep_only()


# Extracts xvector given data and truncated model
class Extractor(Sequential):
    def forward(self, x, model, init_params=False):
        emb = model(x)

        return emb

    def extract(self, x, model):
        id, wavs, lens = x

        feats = params.compute_features(wavs, init_params=False)
        feats = params.mean_var_norm(feats, lens)
        emb = self.forward(feats, model).detach()

        return emb


# All data loaders
train_set = params.train_loader()
valid_set = params.valid_loader()

modules = [params.model, params.linear]

# Object initialization for training xvector model
first_x, first_y = next(zip(*train_set))
xvect_brain = XvectorBrain(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)

# Load the latest checkpoint to resume training (if stopped earlier)
checkpointer.recover_if_possible()

# Train the model
xvect_brain.fit(
    params.epoch_counter, train_set=train_set, valid_set=valid_set,
)

print("Running Xvector Extractor")

# Copy the trained model partially to obtain embeddings
model_a = nn.Sequential(
    xvect_brain.modules[0].layers[0],
    xvect_brain.modules[0].layers[1],
    xvect_brain.modules[0].layers[2].layers[0],
)

# Instantiate Extract() object
ext_brain = Extractor()

# Extract xvectors
xvectors = ext_brain.extract(next(iter(valid_set[0])), model_a)

# Saving xvectors (Optional)
torch.save(xvectors, params.save_folder + "/xvectors.pt")
