#!/usr/bin/python
import os
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

    def summarize(self, stats, write=False):
        summary = {"loss": float(sum(s["loss"] for s in stats) / len(stats))}

        if "PER" in stats[0]:
            per_stats = [item for stat in stats for item in stat["PER"]]
            per_summary = wer_summary(per_stats)
            summary["PER"] = per_summary["WER"]

        return summary


ctc_brain = CTCBrain([params.rnn, params.lin], params.optimizer)
ctc_brain.fit(params.train_loader(), params.valid_loader(), params.N_epochs)
ctc_brain.evaluate(params.train_loader())
