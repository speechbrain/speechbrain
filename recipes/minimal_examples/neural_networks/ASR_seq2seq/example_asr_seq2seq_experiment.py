#!/usr/bin/python
import os
import speechbrain as sb
from seq2seq_brain import seq2seqBrain
from speechbrain.utils.train_logger import summarize_error_rate

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

train_set = hyperparams.train_loader()
first_x, first_y = next(iter(train_set))
seq2seq_brain = seq2seqBrain(
    modules=hyperparams.modules,
    optimizers={("enc", "emb", "dec", "lin"): hyperparams.optimizer},
    device="cpu",
    first_inputs=[first_x, first_y],
)
seq2seq_brain.fit(
    range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
)
test_stats = seq2seq_brain.evaluate(hyperparams.test_loader())
print("Test PER: %.2f" % summarize_error_rate(test_stats["PER"]))


def test_error():
    assert seq2seq_brain.avg_train_loss < 1.0
