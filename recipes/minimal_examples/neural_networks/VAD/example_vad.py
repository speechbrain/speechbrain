#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.train_logger import summarize_average
from speechbrain.utils.train_logger import summarize_error_rate
import torch
from speechbrain.utils.classification_metrics import BinaryMetrics
from speechbrain.data_io.data_io import DataLoaderFactory
import numpy as np

experiment_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")
data_folder = "../../../../../samples/speech_labelling"
data_folder = os.path.abspath(experiment_dir + data_folder)
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


class VADBrain(sb.core.Brain):
    def on_training_start(self):
        self.metrics = BinaryMetrics()

    def compute_forward(self, x, train_mode=True, init_params=False, stage=None):
        id, wavs, lens = x
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)
        x = params.rnn(feats, init_params=init_params)
        outputs = params.lin(x, init_params)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage=True):
        predictions, lens = predictions

        targets = targets[1].to(predictions.device)
        predictions = predictions[:, :targets.shape[-1], 0]
        loss = params.compute_cost(torch.nn.BCEWithLogitsLoss(reduction="none"), predictions, targets, lens)

        self.metrics.update(torch.sigmoid(predictions), targets)
        # compute DER
        stats = {"loss": loss} # dummy for now
        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.4f" % summarize_average(train_stats["loss"]))
        print("Precision: %.2f" % self.metrics.get_precision())
        print("Recall: %.2f" % self.metrics.get_recall())


def parsing_func(params, string):
    boundaries = string.split(" ")
    # we group by two
    # 0.01 is 10 ms hop size ... IS THERE AN EASY WAY to pass to mfcc the step size ?
    boundaries =  [int(float(x)/0.01) for x in boundaries]
    boundaries = list(zip(boundaries[::2], boundaries[1::2]))

    gt = torch.zeros(int(np.ceil(params.example_length*(1/0.01))))

    for indxs in boundaries:
        start, stop = indxs
        gt[start:stop] = 1

    return gt

train_set = DataLoaderFactory(params.csv_train, params.N_batch, ["wav", "speech"],
                              replacements={"$data_folder": params.data_folder},
                              label_parsing_func = lambda x: parsing_func(params, x))

first_x, first_y = next(iter(train_set()))
vad_brain = VADBrain(
    modules=[params.rnn, params.lin],
    optimizer=params.optimizer,
    first_inputs=[first_x],
)
vad_brain.fit(range(params.N_epochs), train_set(), train_set())
test_stats = vad_brain.evaluate(train_set())
#print("Test DER: %.2f" % summarize_error_rate(test_stats["DER"]))


def test_error():
    assert summarize_average(test_stats["loss"]) < 0.01
