#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.utils.train_logger import summarize_average
from speechbrain.utils.train_logger import summarize_error_rate
from ruamel import yaml
from speechbrain.data_io.dataloader import SaveableDataLoader, collate_pad
from speechbrain.data_io.encoders import CategoricalEncoder
from speechbrain.data_io.utils import replace_entries
from speechbrain.data_io.data_io import read_wav
from speechbrain.data_io.datasets import ASRDataset
import torch

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


class CTCBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        wavs, lens = x["waveforms"]
        # lens now supports multidimensional padding
        # IMO this is a must have feature but we need to change also losses accordingly.
        # TEMPORARY PATCH:
        lens = torch.Tensor([x[-1] for x in lens]).to(wavs.device)

        # also wavs is batch, channels, frames but feature extraction does not support
        # multi channel data
        # TEMPORARY FIX:
        assert wavs.size(1) == 1, "multichannel not supported"
        wavs = wavs.squeeze(1)

        feats = hyperparams.compute_features(wavs, init_params)
        feats = hyperparams.mean_var_norm(feats, lens)
        x = hyperparams.model(feats, init_params=init_params)
        x = hyperparams.lin(x, init_params)
        outputs = hyperparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage="train"):
        predictions, lens = predictions
        phns, phn_lens = batch["phones"]
        ids = batch["id"]
        # SAME UGLY PATCHES AS BEFORE:
        phn_lens = torch.Tensor([x[-1] for x in phn_lens]).to(
            predictions.device
        )
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


# STEP 1 load the yaml files
with open(os.path.join(data_folder, "train.yaml"), "r",) as f:
    trainset = yaml.safe_load(f)
with open(os.path.join(data_folder, "dev.yaml"), "r",) as f:
    devset = yaml.safe_load(f)
with open(os.path.join(data_folder, "test.yaml"), "r",) as f:
    testset = yaml.safe_load(f)

# STEP  2 build the label dictionary
encoder = CategoricalEncoder([trainset, devset, testset], "phones")
encoder.update("<blank>")

# STEP 3 replace entries (optional)
replacements_dict = {
    "files": {"DATASET_ROOT": data_folder},
    "alignment_file": {"ALIGNMENTS_ROOT": data_folder},
}

replace_entries(trainset, replacements_dict)  # in place substitution
replace_entries(devset, replacements_dict)
replace_entries(testset, replacements_dict)

# STEP 4 init the datasets

trainset = ASRDataset(
    trainset,
    ["waveforms", "phones"],
    {"waveforms": read_wav, "phones": encoder.encode},
)
devset = ASRDataset(
    devset,
    ["waveforms", "phones"],
    {"waveforms": read_wav, "phones": encoder.encode},
)
testset = ASRDataset(
    testset,
    ["waveforms", "phones"],
    {"waveforms": read_wav, "phones": encoder.encode},
)


# STEP 5 define the dataloaders

train_loader = SaveableDataLoader(
    trainset, batch_size=hyperparams.N_batch, collate_fn=collate_pad
)
valid_loader = SaveableDataLoader(
    trainset, batch_size=hyperparams.N_batch, collate_fn=collate_pad
)
test_loader = SaveableDataLoader(
    trainset, batch_size=hyperparams.N_batch, collate_fn=collate_pad
)


first_batch = next(iter(train_loader))
ctc_brain = CTCBrain(
    modules=[hyperparams.model, hyperparams.lin],
    optimizer=hyperparams.optimizer,
    first_inputs=first_batch,
)
ctc_brain.fit(range(hyperparams.N_epochs), train_loader, valid_loader)
test_stats = ctc_brain.evaluate(test_loader)
print("Test PER: %.2f" % summarize_error_rate(test_stats["PER"]))


# Integration test: check that the model overfits the training data
def test_error():
    print(ctc_brain.avg_train_loss)
    assert ctc_brain.avg_train_loss < 3.0
