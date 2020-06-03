#!/usr/bin/python
import os
import sys
import speechbrain as sb
from speechbrain.nnet.containers import Sequential

# from speechbrain.utils.train_logger import summarize_average
# from speechbrain.utils.checkpoints import ckpt_recency

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from voxceleb1_prepare import prepare_voxceleb1  # noqa E402

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

# Prepare data
prepare_voxceleb1(
    data_folder=params.data_folder,
    save_folder=params.save_folder,
    splits=["train", "dev"],
    split_ratio=[70, 30],
    seg_dur=300,
    vad=False,
    rand_seed=1234,
)


# Trains xvector model
class XvectorBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x

        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        x = params.model(feats, init_params)
        x = params.output_linear(x, init_params)

        outputs = params.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        uttid, spkid, _ = targets

        loss = params.compute_cost(predictions, spkid, lens)

        stats = {}

        # if stage != "train":
        stats["error"] = params.compute_error(predictions, spkid, lens)

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        epoch_stats = {"epoch": epoch}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)
        params.checkpointer.save_and_keep_only()


# Extracts xvector given data and truncated model
class Extractor(Sequential):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_emb(self, feats):

        emb = self.model(feats)

        return emb

    def extract(self, x):
        id, wavs, lens = x

        feats = params.compute_features(wavs, init_params=False)
        feats = params.mean_var_norm(feats, lens)

        emb = self.get_emb(feats)
        emb = emb.detach()

        return emb


# Data loaders
train_set = params.train_loader()
valid_set = params.valid_loader()

# Xvector Model
modules = [params.model, params.output_linear]
first_x, first_y = next(iter(train_set))

# Object initialization for training xvector model
xvect_brain = XvectorBrain(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)

# Recover checkpoints
params.checkpointer.recover_if_possible()

# Train the Xvector model
xvect_brain.fit(
    params.epoch_counter, train_set=train_set, valid_set=valid_set,
)
print("Xvector model training completed!")

# Truncate model and keep till layer emb a
model_a = Sequential(*xvect_brain.modules[0].layers[0:17],)
print("Model has been truncated!")

# Instantiate extractor obj
ext_brain = Extractor(model=model_a)

# Extract xvectors from a validation sample
valid_x, valid_y = next(iter(valid_set))
print(
    "Extracting Xvector from a sample validation batch using truncated model!"
)
xvectors = ext_brain.extract(valid_x)
print("Extracted Xvector.Shape: ", xvectors.shape)
