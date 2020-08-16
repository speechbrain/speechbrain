#!/usr/bin/python
import os
import speechbrain as sb
from gan_brain import EnhanceGanBrain

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


train_set = hyperparams.train_loader()
first_x = next(iter(train_set))
auto_brain = EnhanceGanBrain(
    modules=hyperparams.modules,
    optimizers={
        "generator": hyperparams.g_optimizer,
        "discriminator": hyperparams.d_optimizer,
    },
    device="cpu",
    first_inputs=first_x,
)
auto_brain.fit(
    range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
)
test_stats = auto_brain.evaluate(hyperparams.test_loader())


# Integration test: use eval loss cuz test loss is GAN loss.
def test_loss():
    assert test_stats < 0.002
