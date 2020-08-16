#!/usr/bin/python
import os
import speechbrain as sb
from asr_brain import ASR_Brain

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

train_set = hyperparams.train_loader()
first_x, first_y = next(iter(train_set))
asr_brain = ASR_Brain(
    modules=hyperparams.modules,
    optimizers={("linear1", "linear2"): hyperparams.optimizer},
    device="cpu",
    first_inputs=[first_x],
)
asr_brain.fit(
    range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
)
test_stats = asr_brain.evaluate(hyperparams.test_loader())


# Define an integration test of overfitting on the train data
def test_error():
    assert asr_brain.train_loss < 0.2
