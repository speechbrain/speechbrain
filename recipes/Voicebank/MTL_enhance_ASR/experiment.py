#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_average
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.losses import mse_loss

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
    params_to_save=params_file,
    overrides=overrides,
)


# Define training procedure
class MTL_Brain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):

        ids, wavs, wav_lens = x

        # Compute magnitude features
        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        mag = spectral_magnitude(params.compute_stft(wavs), power=0.5)
        mag = torch.log1p(mag)

        # Compute enhancement outputs
        mask = params.enhance_model(mag, init_params)
        enhanced_mag = mask * mag

        # If necessary, compute asr model outputs
        out = None
        if params.enhance_loss["mimic"] > 0 or params.asr_loss["ctc"] > 0:

            # Convert to ASR's expected inputs: fbanks
            feats = torch.expm1(enhanced_mag) ** 2
            feats = params.filterbank(feats)

            # Compute ASR outputs
            out = params.asr_model(feats, init_params)

            # Only need to compute output layer if we're using CTC loss
            if params.asr_loss["ctc"] > 0:
                out = params.output(out, init_params)
                out = params.log_softmax(out)

        return enhanced_mag, out

    def compute_objectives(self, predictions, clean, chars, stage="train"):
        enhanced_mag, asr_out = predictions
        ids, clean_wav, clean_lens = clean
        ids, chars, char_lens = chars

        clean_wav = clean_wav.to(params.device)
        clean_lens = clean_lens.to(params.device)
        chars = chars.to(params.device)
        char_lens = char_lens.to(params.device)

        stats = {}
        # Compute all necessary losses
        if params.enhance_loss["mse"] > 0.0:
            clean_mag = params.compute_stft(clean_wav)
            clean_mag = spectral_magnitude(clean_mag, power=0.5)
            clean_mag = torch.log1p(clean_mag)
            stats["mse"] = params.enhance_loss["mse"] * mse_loss(
                enhanced_mag, clean_mag, clean_lens
            )

        if params.enhance_loss["mimic"] > 0.0:
            feats = params.compute_stft(clean_wav)
            feats = spectral_magnitude(feats, power=1)
            feats = params.filterbank(feats)
            clean_outputs = params.asr_model(feats)

            # NOTE: This is not correct if both "mimic" and "CTC" are used
            stats["mimic"] = params.enhance_loss["mimic"] * mse_loss(
                asr_out, clean_outputs, clean_lens
            )

        if params.asr_loss["ctc"] > 0.0:
            stats["ctc"] = params.asr_loss["ctc"] * params.ctc_loss(
                asr_out, chars, [clean_lens, char_lens]
            )

        loss = sum(stats.get(k, 0.0) for k in ["mse", "mimic", "ctc"])

        return loss, stats

    def fit_batch(self, batch):
        clean, noisy, chars = batch
        predictions = self.compute_forward(noisy)
        loss, stats = self.compute_objectives(predictions, clean, chars)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def evaluate_batch(self, batch, stage="test"):
        clean, noisy, chars = batch
        predictions = self.compute_forward(noisy, stage=stage)
        loss, stats = self.compute_objectives(
            predictions, clean, chars, stage=stage
        )
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        params.train_logger.log_stats(
            {"epoch": epoch}, train_stats, valid_stats
        )
        params.checkpointer.save_and_keep_only(
            meta={"loss": summarize_average(valid_stats["loss"])},
            importance_keys=[ckpt_recency, lambda c: -c.meta["loss"]],
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
modules = []
if (
    params.enhance_loss["mse"] > 0.0
    or params.enhance_loss["mimic"] > 0.0
    or params.enhance_loss["ctc"] > 0.0
):
    modules.append(params.enhance_model)

if params.asr_loss["ctc"] > 0.0:
    modules.extend([params.asr_model, params.output])

if hasattr(params, "augmentation"):
    modules.append(params.augmentation)

# Create brain object for training
mtl_brain = MTL_Brain(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)

# Load pretrained models
if hasattr(params, "enhance_pretrain"):
    params.enhance_model.load_state_dict(torch.load(params.enhance_pretrain))
if hasattr(params, "clean_asr_model"):
    params.asr_model.load_state_dict(torch.load(params.clean_asr_model))

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
mtl_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
params.checkpointer.recover_if_possible(lambda c: -c.meta["CER"])
test_stats = mtl_brain.evaluate(params.test_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
cer_summary = edit_distance.wer_summary(test_stats["CER"])
with open(params.wer_file, "w") as fo:
    wer_io.print_wer_summary(cer_summary, fo)
    wer_io.print_alignments(test_stats["CER"], fo)
