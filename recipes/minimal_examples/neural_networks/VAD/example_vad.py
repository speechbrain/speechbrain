#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.train_logger import summarize_average
from speechbrain.utils.train_logger import summarize_error_rate
import torch

experiment_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")
data_folder = "../../../../../samples/speech_labelling"
data_folder = os.path.abspath(experiment_dir + data_folder)
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


class VADBrain(sb.core.Brain):
    def compute_forward(self, x, train_mode=True, init_params=False, stage=None):
        id, wavs, lens = x
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)
        x = params.rnn(feats, init_params=init_params)
        x = params.lin(x, init_params)
        outputs = x

        return outputs, lens

    def create_targets_mat(self, predictions, targets):
        # create targets here
        gt = torch.zeros_like(predictions).to(predictions.device)

        for b in range(len(targets[1])): #kinda hacky
            tmp = [int(float(x)/0.01) for x in targets[1][b]]
            tmp = list(zip(tmp[::2], tmp[1::2]))
            for indxs in tmp:
                start, stop = indxs
                gt[b][start:stop] = 1

        return gt

    def compute_objectives(self, predictions, targets, stage=True):
        predictions, lens = predictions

        targets = self.create_targets_mat(predictions, targets)

        loss = params.compute_cost(torch.nn.BCEWithLogitsLoss(reduction="none"), predictions, targets, lens)


        # compute DER
        stats = {"DER": loss} # dummy for now
        return loss, stats



    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        #print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        #print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        #print("Valid DER: %.2f" % summarize_error_rate(valid_stats["DER"])) # detection error rate


train_set = params.train_loader()
first_x, first_y = next(iter(train_set))
vad_brain = VADBrain(
    modules=[params.rnn, params.lin],
    optimizer=params.optimizer,
    first_inputs=[first_x],
)
vad_brain.fit(range(params.N_epochs), train_set, params.valid_loader())
test_stats = vad_brain.evaluate(params.test_loader())
#print("Test DER: %.2f" % summarize_error_rate(test_stats["DER"]))


# For such a small dataset, the PER can be unpredictable.
# Instead, check that at the end of training, the error is acceptable.
def test_error():
    assert summarize_average(test_stats["DER"]) < 1.0
