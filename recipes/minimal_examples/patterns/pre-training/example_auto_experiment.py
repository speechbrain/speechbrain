#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average

experiment_dir = os.path.dirname(os.path.realpath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
save_folder = os.path.realpath(os.path.join(experiment_dir, "checkpoints"))
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(
        fin, {"data_folder": data_folder, "save_folder": save_folder}
    )


class AutoBrain(sb.core.Brain):
    def compute_forward(self, x, init_params=False):
        id, wavs, lens = x
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        encoded = params.linear1(feats, init_params)
        encoded = params.activation(encoded)
        decoded = params.linear2(encoded, init_params)

        return decoded

    def compute_objectives(self, predictions, targets):
        id, wavs, lens = targets
        feats = params.compute_features(wavs, init_params=False)
        feats = params.mean_var_norm(feats, lens)
        return params.compute_cost(predictions, feats, lens)

    def fit_batch(self, batch):
        inputs = batch[0]
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, inputs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch, stage="test"):
        inputs = batch[0]
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, inputs)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        params.checkpointer.save_and_keep_only(
            meta={"loss": summarize_average(valid_stats["loss"])},
            ckpt_predicate=lambda c: "loss" in c.meta,
            importance_keys=[lambda c: -c.meta["loss"]],
        )
        print("Completed epoch %d" % epoch)
        print("Train loss: %.3f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.3f" % summarize_average(valid_stats["loss"]))


train_set = params.train_loader()
first_x = next(iter(train_set))
auto_brain = AutoBrain(
    modules=[params.linear1, params.linear2],
    optimizer=params.optimizer,
    first_inputs=first_x,
)
auto_brain.fit(range(params.N_epochs), train_set, params.valid_loader())

# Save the pretrained parameters:
try:
    params.checkpointer.save_checkpoint(name="CKPT_final")
except FileExistsError:
    pass  # The pretrained model already existed, let's just pass.


################################################################################
## After this, a new experiment file could start, using the pretrained encoder.
## Since these files are concatenated, some lines here are redundant.
################################################################################

import os  # noqa:E402
import speechbrain as sb  # noqa:E402
from speechbrain.nnet.containers import Sequential  # noqa:E402

experiment_dir = os.path.dirname(os.path.realpath(__file__))
params_file = os.path.join(experiment_dir, "pretrained-params.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))

pretrained_checkpoint = os.path.realpath(
    os.path.join(experiment_dir, "checkpoints/CKPT_final")
)
with open(params_file) as fin:
    pt_params = sb.yaml.load_extended_yaml(
        fin,
        {
            "data_folder": data_folder,
            "pretrained_checkpoint": pretrained_checkpoint,
        },
    )


def feature_extractor(x, lens, init_params=False):
    x = pt_params.compute_features(x, init_params)
    x = pt_params.mean_var_norm(x, lens)
    return x


pretrained_encoder = Sequential(pt_params.linear1, pt_params.activation)

train_set = pt_params.train_loader()
ids, first_x, lens = next(iter(train_set))[0]
# Initialize, which also loads the pretrained parameters:
first_x = feature_extractor(first_x, lens, init_params=True)
pretrained_encoder(first_x, init_params=True)  # This is now a pretrained model.

################################################################################
## Now let's demonstrate that the model is really the same, using the models
## from both sections of this file
################################################################################
import torch  # noqa:E402

original_encoder = Sequential(params.linear1, params.activation)


# Use a function so that this gets called by pytest
def test_loss():
    for (inputs,) in train_set:
        id, wavs, lens = inputs
        features = feature_extractor(wavs, lens)
        original_out = original_encoder(features)
        pt_out = pretrained_encoder(features)
        assert torch.allclose(original_out, pt_out)
    # Delete the checkpoints here, so the automatic testing does not leave
    # things behind
    import shutil

    shutil.rmtree(save_folder)
