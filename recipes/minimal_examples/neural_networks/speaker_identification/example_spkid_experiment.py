#!/usr/bin/python
import os
import speechbrain as sb
from spk_id_brain import SpkIdBrain
from speechbrain.utils.train_logger import summarize_average

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

train_set = hyperparams.train_loader()
first_x, first_y = next(iter(train_set))
spk_id_brain = SpkIdBrain(
    modules=hyperparams.modules,
    optimizers={("linear1", "linear2"): hyperparams.optimizer},
    device="cpu",
    first_inputs=[first_x],
)
spk_id_brain.fit(
    range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
)
test_stats = spk_id_brain.evaluate(hyperparams.test_loader())
print("Test error: %.2f" % summarize_average(test_stats["error"]))


# Integration test: ensure we overfit the training data
def test_error():
    assert spk_id_brain.avg_train_loss < 0.2
