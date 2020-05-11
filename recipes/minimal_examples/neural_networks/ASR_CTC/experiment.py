#!/usr/bin/python
import os
import torch
import speechbrain as sb
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.edit_distance import wer_summary
from speechbrain.utils.edit_distance import wer_details_for_batch

current_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(current_dir, "params.yaml")
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin)

sb.core.create_experiment_directory(
    experiment_directory=params.output_folder, params_to_save=params_file,
)


def summarize_per(stats):
    per_summary = wer_summary(stats)
    return per_summary["WER"]


class CTCBrain(sb.core.Brain):
    def forward(self, x, init_params=False):
        id, wavs, lens = x
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        x = params.rnn(feats, init_params)
        x = params.lin(x, init_params)
        outputs = params.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, train=True):
        predictions, lens = predictions
        ids, phns, phn_lens = targets
        loss = params.compute_cost(predictions, phns, [lens, phn_lens])

        if not train:
            seq = ctc_greedy_decode(predictions, lens, blank_id=-1)
            phns = undo_padding(phns, phn_lens)
            stats = {"PER": wer_details_for_batch(ids, phns, seq)}
            return loss, stats

        return loss

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % torch.Tensor(train_stats["loss"]).mean())
        print("Valid loss: %.2f" % torch.Tensor(valid_stats["loss"]).mean())
        print("Valid PER: %.2f" % summarize_per(valid_stats["PER"]))


train_set = params.train_loader()
ctc_brain = CTCBrain(
    modules=[params.rnn, params.lin],
    optimizer=params.optimizer,
    first_input=next(iter(train_set[0])),
)
ctc_brain.fit(range(params.N_epochs), train_set, params.valid_loader())
test_stats = ctc_brain.evaluate(params.test_loader())
print("Test PER: %.2f" % summarize_per(test_stats["PER"]))
