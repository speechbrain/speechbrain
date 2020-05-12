#!/usr/bin/python
import os
import torch
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average

current_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(current_dir, "params.yaml")
data_folder = "../../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.abspath(current_dir + data_folder)
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


class SpkIdBrain(sb.core.Brain):
    def forward(self, x, init_params=False):
        id, wavs, lens = x
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        x = params.linear1(feats, init_params)
        x = params.activation(x)
        x = params.linear2(x, init_params)
        x = torch.mean(x, dim=1, keepdim=True)
        outputs = params.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, train=True):
        predictions, lens = predictions
        uttid, spkid, _ = targets
        loss = params.compute_cost(predictions, spkid, lens)

        if not train:
            stats = {"error": params.compute_error(predictions, spkid, lens)}
            return loss, stats

        return loss

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid error: %.2f" % summarize_average(valid_stats["error"]))


train_set = params.train_loader()
spk_id_brain = SpkIdBrain(
    modules=[params.linear1, params.linear2],
    optimizer=params.optimizer,
    first_input=next(iter(train_set[0])),
)
spk_id_brain.fit(range(params.N_epochs), train_set, params.valid_loader())
test_stats = spk_id_brain.evaluate(params.test_loader())
print("Test error: %.2f" % summarize_average(test_stats["error"]))

assert summarize_average(test_stats["error"]) == 0.0
