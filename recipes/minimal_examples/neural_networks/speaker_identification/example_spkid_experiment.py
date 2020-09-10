#!/usr/bin/python
import os
import torch
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


class SpkIdBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x
        feats = hyperparams.compute_features(wavs, init_params)
        feats = hyperparams.mean_var_norm(feats, lens)

        x = hyperparams.linear1(feats, init_params)
        x = hyperparams.activation(x)
        x = hyperparams.linear2(x, init_params)
        x = torch.mean(x, dim=1, keepdim=True)
        outputs = hyperparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        uttid, spkid, _ = targets
        loss = hyperparams.compute_cost(predictions, spkid, lens)

        stats = {}

        if stage != "train":
            stats["error"] = hyperparams.compute_error(predictions, spkid, lens)

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid error: %.2f" % summarize_average(valid_stats["error"]))


train_set = hyperparams.train_loader()
first_x, first_y = next(iter(train_set))
spk_id_brain = SpkIdBrain(
    modules=[hyperparams.linear1, hyperparams.linear2],
    optimizer=hyperparams.optimizer,
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
