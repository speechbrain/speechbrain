#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.utils.train_logger import summarize_average
from speechbrain.utils.train_logger import summarize_error_rate
from speechbrain.data_io.utils import (
    CategoricalEncoder,
    replace_entries,
    dataset_sanity_check,
)
from speechbrain.data_io.datasets import ASRDataset
from torch.utils.data import DataLoader
from speechbrain.data_io.dataloader import pad_examples

from ruamel import yaml


experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams_2.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


class CTCBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x
        wavs = wavs[0][:, 0]
        lens = lens[0]
        feats = hyperparams.compute_features(wavs, init_params)
        feats = hyperparams.mean_var_norm(feats, lens)
        x = hyperparams.model(feats, init_params=init_params)
        x = hyperparams.lin(x, init_params)
        outputs = hyperparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        ids, phns, phn_lens = targets
        ids = ids[0]
        phns = phns[0]
        phn_lens = phn_lens[0]
        loss = hyperparams.compute_cost(predictions, phns, lens, phn_lens)

        stats = {}
        if stage != "train":
            seq = ctc_greedy_decode(predictions, lens, blank_id=-1)
            phns = undo_padding(phns, phn_lens)
            stats["PER"] = wer_details_for_batch(ids, phns, seq)

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid PER: %.2f" % summarize_error_rate(valid_stats["PER"]))


with open(os.path.join(data_folder, "dev.yaml"), "r",) as f:
    devset = yaml.safe_load(f)


encoder = CategoricalEncoder(devset, "phones")
replacements_dict = {
    "files": {"DATASET_ROOT": data_folder},
    "alignment_file": {"ALIGNMENT_ROOT": data_folder},
}

devset = replace_entries(devset, replacements_dict)
dataset_sanity_check([devset, devset])  # sanity check for dev
dataset = ASRDataset(devset, "phones", encoder)


train_set = DataLoader(dataset, 2, shuffle=True, collate_fn=pad_examples)


first_x, first_y = next(iter(train_set))
ctc_brain = CTCBrain(
    modules=[hyperparams.model, hyperparams.lin],
    optimizer=hyperparams.optimizer,
    first_inputs=[first_x],
)
ctc_brain.fit(range(hyperparams.N_epochs), train_set, train_set)
test_stats = ctc_brain.evaluate(train_set)
print("Test PER: %.2f" % summarize_error_rate(test_stats["PER"]))


# Integration test: check that the model overfits the training data
def test_error():
    assert ctc_brain.avg_train_loss < 3.0
