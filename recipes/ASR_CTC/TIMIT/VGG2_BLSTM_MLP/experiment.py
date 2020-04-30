#!/usr/bin/env python3
import sys
import torch
import speechbrain as sb
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding

# Load hyperparameters file with command-line overrides
overrides = sb.core.parse_arguments(sys.argv[1:])
if "seed" in overrides:
    torch.manual_seed(overrides["seed"])
params_file = "recipes/ASR_CTC/TIMIT/VGG2_BLSTM_MLP/params.yaml"
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
    log_config="logging.yaml",
)


# Define training procedure
class ASR(sb.core.Brain):
    def forward(self, x, init_params=False):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)
        return params.model(feats, init_params), wav_lens

    def compute_objectives(self, predictions, targets, train=True):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)
        loss = params.compute_cost(pout, phns, [pout_lens, phn_lens])

        if not train:
            ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            sequence = convert_index_to_lab(sequence, ind2lab)
            phns = undo_padding(phns, phn_lens)
            phns = convert_index_to_lab(phns, ind2lab)
            stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats = {"PER": stats}
            return loss, stats

        return loss

    def summarize(self, stats, write=False):
        accumulator = {"loss": 0.0}
        if "PER" in stats[0]:
            accumulator["PER"] = []
        for stat in stats:
            for stat_type in stat:
                accumulator[stat_type] += stat[stat_type]

        summary = {"loss": float(accumulator["loss"] / len(stats))}
        if "PER" in accumulator:
            per_summary = edit_distance.wer_summary(accumulator["PER"])
            summary["PER"] = per_summary["WER"]
            if write:
                with open(params.wer_file, "w") as fo:
                    wer_io.print_wer_summary(per_summary, fo)
                    wer_io.print_alignments(accumulator["PER"], fo)

        return summary


saver = sb.utils.checkpoints.Checkpointer(
    checkpoints_dir=params.save_folder,
    recoverables={
        "model": params.model,
        "optimizer": params.optimizer,
        "scheduler": params.lr_annealing,
        "normalizer": params.normalize,
    },
)
asr_brain = ASR(
    modules=[params.model],
    optimizer=params.optimizer,
    scheduler=params.lr_annealing,
    saver=saver,
)

# Experiment
params.prepare_timit()

asr_brain.fit(
    train_set=params.train_loader(),
    valid_set=params.valid_loader(),
    number_of_epochs=params.number_of_epochs,
    min_keys=["PER"],
)

asr_brain.evaluate(params.test_loader(), min_key="PER")
