#!/usr/bin/env python3
import sys
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.core import Experiment, Brain

with open("recipes/ASR_CTC/TIMIT/VGG2_BLSTM_MLP/params.yaml") as fi:
    sb = Experiment(yaml_stream=fi, commandline_args=sys.argv[1:],)


class ASR(Brain):
    def forward(self, x, init_params=False):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(sb.device), wav_lens.to(sb.device)
        feats = sb.compute_features(wavs, init_params=init_params)
        feats = sb.normalize(feats, wav_lens)
        return sb.model(feats, init_params=init_params), wav_lens

    def compute_objectives(self, predictions, targets, train=True):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(sb.device), phn_lens.to(sb.device)
        loss = sb.compute_cost(pout, phns, [pout_lens, phn_lens])

        if not train:
            ind2lab = sb.train_loader.label_dict["phn"]["index2lab"]
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            sequence = convert_index_to_lab(sequence, ind2lab)
            phns = undo_padding(phns, phn_lens)
            phns = convert_index_to_lab(phns, ind2lab)
            stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats = {"per": stats}
            return loss, stats

        return loss

    def summarize(self, stats, write=False):

        # Accumulate
        accumulator = {"loss": 0.0}
        if "per" in stats[0]:
            accumulator["per"] = []
        for stat in stats:
            for stat_type in stat:
                accumulator[stat_type] += stat[stat_type]

        # Normalize
        summary = {"loss": float(accumulator["loss"] / len(stats))}
        if "per" in accumulator:
            per_summary = edit_distance.wer_summary(accumulator["per"])
            summary["per"] = per_summary["WER"]

            # Write test data to file
            if write:
                with open(sb.wer_file, "w") as fo:
                    wer_io.print_wer_summary(per_summary, fo)
                    wer_io.print_alignments(accumulator["per"], fo)

        return summary


# prepare the data
sb.prepare_timit()

# initialize brain and learn
asr_brain = ASR(
    modules=[sb.model],
    optimizer=sb.optimizer,
    scheduler=sb.lr_annealing,
    saver=sb.saver,
)
asr_brain.fit(
    train_set=sb.train_loader(),
    valid_set=sb.valid_loader(),
    number_of_epochs=sb.number_of_epochs,
    min_keys=["per"],
)

# load best model, evaluate that:
asr_brain.evaluate(sb.test_loader(), min_key="per")
