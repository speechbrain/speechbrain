#!/usr/bin/python
import os
import speechbrain as sb
from auto_brain import AutoBrain

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

if hyperparams.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    train_logger = TensorboardLogger(hyperparams.tensorboard_logs)
    hyperparams.modules["train_logger"] = train_logger

train_set = hyperparams.train_loader()
first_x = next(iter(train_set))
auto_brain = AutoBrain(
    modules=hyperparams.modules,
    optimizers={("linear1", "linear2"): hyperparams.optimizer},
    first_inputs=first_x,
)
auto_brain.fit(
    range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
)
test_stats = auto_brain.evaluate(hyperparams.test_loader())


# Integration test: make sure we are overfitting training data
def test_loss():
    assert auto_brain.train_loss < 0.08
