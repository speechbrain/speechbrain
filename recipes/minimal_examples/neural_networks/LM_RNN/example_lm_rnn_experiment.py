#!/usr/bin/env python3
import os
import speechbrain as sb

from lm_brain import LMBrain

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

train_set = hyperparams.train_loader()
valid_set = hyperparams.valid_loader()
first_y = next(iter(train_set))

lm_brain = LMBrain(
    modules=hyperparams.modules,
    optimizers={"model": hyperparams.optimizer},
    device="cpu",
    first_inputs=first_y,
)

lm_brain.fit(hyperparams.epoch_counter, train_set, valid_set)

test_stats = lm_brain.evaluate(hyperparams.test_loader())


# Integration test: check that the model overfits the training data
def test_error():
    assert lm_brain.train_loss < 0.15
