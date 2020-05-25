#!/usr/bin/python
import os
import sys
import torch
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average
from speechbrain.processing.features import spectral_magnitude

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from dns_prepare import DNSPreparer  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
if "seed" in overrides:
    torch.manual_seed(overrides["seed"])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
)


if params.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    train_logger = TensorboardLogger(params.tensorboard_logs)


class AutoBrain(sb.core.Brain):
    def compute_forward(self, x, train_mode=True, init_params=False):
        ids, wavs, lens = x
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        feats = params.compute_features(wavs)
        feats = spectral_magnitude(feats)
        feats = params.mean_var_norm(feats, lens)

        encoded = params.conv1(feats, init_params)
        encoded = params.activation(encoded)

        encoded = params.conv2(encoded, init_params)
        encoded = params.activation(encoded)

        encoded = params.conv3(encoded, init_params)
        encoded = params.activation(encoded)

        encoded = params.conv4(encoded, init_params)
        encoded = params.activation(encoded)

        encoded = params.conv5(encoded, init_params)
        encoded = params.activation(encoded)

        encoded = params.conv6(encoded, init_params)
        encoded = params.activation(encoded)

        encoded = params.conv7(encoded, init_params)
        encoded = params.activation(encoded)

        decoded = params.linear(encoded, init_params)

        return decoded

    def compute_objectives(self, predictions, targets, train_mode=True):
        ids, wavs, lens = targets
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        feats = params.compute_features(wavs)
        feats = spectral_magnitude(feats)
        # feats = params.mean_var_norm(feats, lens)
        return params.compute_cost(predictions, feats, lens)

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer(self.modules)
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, targets, train_mode=False)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        if params.use_tensorboard:
            train_logger.log_stats({"Epoch": epoch}, train_stats, valid_stats)
        print("Completed epoch %d" % epoch)
        print("Train loss: %.3f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.3f" % summarize_average(valid_stats["loss"]))


prepare = DNSPreparer(
    data_folder=params.data_folder, save_folder=params.csv_folder,
)
prepare()
train_set = params.train_loader()
valid_set = params.valid_loader()
first_x, first_y = next(zip(*train_set))

train_set = params.train_loader()
first_x = next(iter(train_set[0]))
auto_brain = AutoBrain(
    modules=[
        params.conv1,
        params.conv2,
        params.conv3,
        params.conv4,
        params.conv5,
        params.conv6,
        params.conv7,
        params.linear,
    ],
    optimizer=params.optimizer,
    first_inputs=[first_x],
)
auto_brain.fit(range(params.N_epochs), train_set, valid_set)
# test_stats = auto_brain.evaluate(params.test_loader())
# print("Test loss: %.3f" % summarize_average(test_stats["loss"]))


# If training is successful, reconstruction loss is less than 0.2
# def test_loss():
#     assert summarize_average(test_stats["loss"]) < 0.2
