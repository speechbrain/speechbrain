#!/usr/bin/python
import sys
import torch
import speechbrain as sb

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
    splits=["train", "dev", "test"],
    save_folder=params.data_folder,
)
data_prepare()


xvect_brain.fit(
    train_set=params.train_loader(),
    valid_set=params.valid_loader(),
    number_of_epochs=params.number_of_epochs,
)

xvect_brain.evaluate(params.test_loader())
