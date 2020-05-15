#!/usr/bin/python
import sys
import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.sequential import Sequential

sys.path.append("..")
from voxceleb1_prepare import VoxCelebPreparer  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
if "seed" in overrides:
    torch.manual_seed(overrides["seed"])

# params file
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin)

# creating directory for experiments
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder, params_to_save=params_file,
)


# Trainer
class XvectorBrain(sb.core.Brain):
    def forward(self, x, init_params=False):
        id, wavs, lens = x

        # Data-augementation will come here!

        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        x = params.model(feats, init_params)
        outputs = params.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, train=True):
        predictions, lens = predictions
        uttid, spkid, _ = targets

        loss = params.compute_cost(predictions, spkid, lens)

        if not train:
            stats = {"error": params.compute_error(predictions, spkid, lens)}
            return loss, stats

        return loss

    def summarize(self, stats, write=False):
        summary = {"loss": float(sum(s["loss"] for s in stats) / len(stats))}

        if "error" in stats[0]:
            summary["error"] = float(
                sum(s["error"] for s in stats) / len(stats)
            )

        return summary


# Extractor
class Extractor(Sequential):
    def forward(self, x, model, init_params=False):
        emb = model(x)

        return emb

    def extract(self, x, model):
        id, wavs, lens = x

        feats = params.compute_features(wavs)
        feats = params.mean_var_norm(feats, lens)
        emb = self.forward(feats, model).detach()

        return emb


saver = sb.utils.checkpoints.Checkpointer(
    checkpoints_dir=params.save_folder,
    recoverables={
        "model": params.model,
        "optimizer": params.optimizer,
        "normalizer": params.normalize,
    },
)

xvect_brain = XvectorBrain(
    modules=[params.model], optimizer=params.optimizer, saver=saver,
)

data_prepare = VoxCelebPreparer(
    data_folder=params.data_folder,
    splits=["train", "dev"],
    save_folder=params.data_folder,
    vad=False,
    seg_dur=300,
    rand_seed=params.seed,
)
data_prepare()

xvect_brain.fit(
    train_set=params.train_loader(),
    valid_set=params.valid_loader(),
    number_of_epochs=params.number_of_epochs,
)

# Not needed in vox1 verification section
# xvect_brain.evaluate(params.test_loader())

print("Now Running Xvector Extractor")

# Embedding b is expected to be better than embedding a
model_b = nn.Sequential(
    xvect_brain.modules[0].layers[0],
    xvect_brain.modules[0].layers[1],
    xvect_brain.modules[0].layers[2].layers[0],
    xvect_brain.modules[0].layers[2].layers[1],
    xvect_brain.modules[0].layers[2].layers[2],
    xvect_brain.modules[0].layers[2].layers[3],
)

ext_brain = Extractor()
xvectors = ext_brain.extract(next(iter(params.valid_loader()[0])), model_b)

# Saving xvectors (Optional)
torch.save(xvectors, params.save_folder + "/xvectors.pt")
