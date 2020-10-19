#!/usr/bin/env python3
import os
import sys
import speechbrain as sb
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_error_rate

# This hack needed to import data preparation script from ../..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from timit_prepare import prepare_timit  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    hyperparams_to_save=params_file,
    overrides=overrides,
)


# Define training procedure
class ASR(sb.core.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_type = params.init_training_type
        print("Starting training type:", self.training_type)

    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        if hasattr(params, "normalize"):
            feats = params.normalize(feats, wav_lens)
        out = params.model(feats, init_params)
        out = params.output(out, init_params)
        out = out - out.mean(1).unsqueeze(1)
        pout = params.log_softmax(out)
        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage="train"):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets[0]
        _, ends, end_lens = targets[1]
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)

        phns_orig = undo_padding(phns, phn_lens)
        phns = params.aligner.expand_phns_by_states_per_phoneme(phns, phn_lens)

        if self.training_type == "forward":
            forward_scores = params.aligner(
                pout, pout_lens, phns, phn_lens, "forward"
            )
            loss = -forward_scores

        elif self.training_type == "ctc":
            loss = params.compute_cost_ctc(pout, phns, pout_lens, phn_lens)
        elif self.training_type == "viterbi":
            prev_alignments = params.aligner.get_prev_alignments(
                ids, pout, pout_lens, phns, phn_lens
            )
            prev_alignments = prev_alignments.to(params.device)
            loss = params.compute_cost_nll(pout, prev_alignments)

        viterbi_scores, alignments = params.aligner(
            pout, pout_lens, phns, phn_lens, "viterbi"
        )

        if self.training_type in ["viterbi", "forward"]:
            params.aligner.store_alignments(ids, alignments)

        stats = {}
        acc = params.aligner.calc_accuracy(alignments, ends, phns_orig)
        stats["accuracy"] = acc

        if stage != "train":
            ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
            sequence = ctc_greedy_decode(
                pout, pout_lens, blank_id=params.blank_index
            )

            # convert sequence back to 1 state per phoneme style
            sequence = [params.aligner.collapse_alignments(x) for x in sequence]

            sequence = convert_index_to_lab(sequence, ind2lab)

            phns = convert_index_to_lab(phns_orig, ind2lab)
            per_stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats["PER"] = per_stats
        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        per = summarize_error_rate(valid_stats["PER"])
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, per)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        params.checkpointer.save_and_keep_only(
            meta={"PER": per},
            importance_keys=[ckpt_recency, lambda c: -c.meta["PER"]],
        )

        if self.training_type == "viterbi":
            # recompute alignments over the full training set
            self.evaluate(train_set)

        if hasattr(params, "switch_training_type"):
            if not hasattr(params, "switch_training_epoch"):
                raise ValueError(
                    "Please specify `switch_training_epoch` in `params`"
                )
            if epoch + 1 == params.switch_training_epoch:
                self.training_type = params.switch_training_type
                print("Switching to training type", self.training_type)

    def fit_batch(self, batch):
        """
        Modify slightly from original version as batch returns ends as well.
        Only first 2 lines are modified
        """
        inputs, phns, ends = batch
        targets = phns, ends
        predictions = self.compute_forward(inputs)
        loss, stats = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def evaluate_batch(self, batch, stage="test"):
        """
        Modify slightly from original version as batch returns ends as well.
        Only first 2 lines are modified.
        """
        inputs, phns, ends = batch
        targets = phns, ends
        out = self.compute_forward(inputs, stage=stage)
        loss, stats = self.compute_objectives(out, targets, stage=stage)
        stats["loss"] = loss.detach()
        return stats


# Prepare data
prepare_timit(
    data_folder=params.data_folder,
    splits=["train", "dev", "test"],
    save_folder=params.data_folder,
    phn_set=str(params.ground_truth_phn_set),
)
train_set = params.train_loader()
valid_set = params.valid_loader()
first_x, first_phns, first_ends = next(iter(train_set))

# Modules are passed to optimizer and have train/eval called on them
modules = [params.model, params.output]
if hasattr(params, "augmentation"):
    modules.append(params.augmentation)

# Create brain object for training
asr_brain = ASR(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
params.checkpointer.recover_if_possible(lambda c: -c.meta["PER"])
test_stats = asr_brain.evaluate(params.test_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
per_summary = edit_distance.wer_summary(test_stats["PER"])
with open(params.wer_file, "w") as fo:
    wer_io.print_wer_summary(per_summary, fo)
    wer_io.print_alignments(test_stats["PER"], fo)
