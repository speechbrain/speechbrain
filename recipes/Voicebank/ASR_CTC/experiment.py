#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.train_logger import summarize_error_rate

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from voicebank_prepare import prepare_voicebank  # noqa E402

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
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        if hasattr(params, "augmentation") and stage == "train":
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)
        out = params.model(feats, init_params)
        out = params.output(out, init_params)
        pout = params.log_softmax(out)
        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage="train"):
        pout, pout_lens = predictions
        ids, chars, char_lens = targets
        chars, char_lens = chars.to(params.device), char_lens.to(params.device)
        loss = params.compute_cost(pout, chars, pout_lens, char_lens)

        stats = {}
        if stage != "train":
            ind2lab = params.train_loader.label_dict["char"]["index2lab"]
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            sequence = convert_index_to_lab(sequence, ind2lab)
            chars = undo_padding(chars, char_lens)
            chars = convert_index_to_lab(chars, ind2lab)
            cer_stats = edit_distance.wer_details_for_batch(
                ids, chars, sequence, compute_alignments=True
            )
            stats["CER"] = cer_stats

        return loss, stats

    def fit_batch(self, batch):
        if len(batch) == 3:
            (ids, clean, lens), (_, noisy, _), (_, chars, char_lens) = batch
            joint_batch = torch.cat((clean, noisy))
            inputs = (ids + ids, joint_batch, torch.cat((lens, lens)))
            joint_targets = torch.cat((chars, chars))
            targets = (ids, joint_targets, torch.cat((char_lens, char_lens)))
        else:
            inputs, targets = batch
        predictions = self.compute_forward(inputs)
        loss, stats = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        cer = summarize_error_rate(valid_stats["CER"])
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, cer)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        params.checkpointer.save_and_keep_only(
            meta={"CER": cer}, min_keys=["CER"],
        )


# Prepare data
prepare_voicebank(
    data_folder=params.data_folder, save_folder=params.data_folder,
)
train_set = params.train_loader()
valid_set = params.valid_loader()
batch = next(iter(train_set))
if len(batch) == 3:
    (ids, clean, lens), (_, noisy, lens), first_y = batch
    first_x = (ids + ids, torch.cat((clean, noisy)), torch.cat((lens, lens)))
else:
    first_x, first_y = batch

# Modules are passed to optimizer and have train/eval called on them
modules = [params.model, params.output]
if hasattr(params, "augmentation"):
    modules.append(params.augmentation)

# Create brain object for training
asr_brain = ASR(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)
if hasattr(params, "pretrained"):
    params.model.load_state_dict(torch.load(params.pretrained))

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
params.checkpointer.recover_if_possible(min_key="CER")
test_stats = asr_brain.evaluate(params.test_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
cer_summary = edit_distance.wer_summary(test_stats["CER"])
with open(params.wer_file, "w") as fo:
    wer_io.print_wer_summary(cer_summary, fo)
    wer_io.print_alignments(test_stats["CER"], fo)
