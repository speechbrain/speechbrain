#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.transducer import decode_batch
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_error_rate

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from timit_prepare import TIMITPreparer  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
if "seed" in overrides:
    torch.manual_seed(overrides["seed"])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
)


# Define training procedure
class TransducerBrain(sb.core.Brain):
    def compute_forward(self, x, y, train_mode=True, init_params=False):
        ids, wavs, lens = x
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        if train_mode:
            _, targets, _ = y
            targets = targets.to(params.device)

        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, lens)
        # Transcription Network: input-output dependency
        TN_output = params.encoder_model(feats, init_params=init_params)
        TN_output = params.encoder_output(TN_output, init_params)
        if train_mode:
            # Generate input seq for PN
            blank_vect = (
                torch.ones(
                    (targets.size(0), 1),
                    device=params.device,
                    dtype=torch.int64,
                )
                * params.blank_id
            )
            prediction_seq = torch.cat((blank_vect, targets.long()), dim=1)
            # Prediction Network: output-output dependency
            PN_output = params.decoder_embedding(prediction_seq, init_params)
            PN_output, _ = params.decoder_model(
                PN_output, init_params=init_params
            )
            PN_output = params.decoder_output(PN_output, init_params)
            # Joint the networks
            joint = params.Tjoint(
                TN_output.unsqueeze(2), PN_output.unsqueeze(1)
            )
            # Output network
            outputs = params.output(joint, init_params)
        else:
            outputs = decode_batch(
                TN_output,
                [
                    params.decoder_embedding,
                    params.decoder_model,
                    params.decoder_output,
                ],
                params.Tjoint,
                [params.output, params.log_softmax],
                params.blank_id,
            )

        outputs = params.log_softmax(outputs)
        return outputs, lens

    def compute_objectives(self, predictions, targets, train_mode=True):
        predictions, lens = predictions
        ids, phns, phn_lens = targets
        if not train_mode:
            # transducer tensor
            pout = predictions.squeeze(2)
            predictions = predictions.expand(-1, -1, phns.shape[1] + 1, -1)

        loss = params.compute_cost(
            predictions,
            phns,
            [lens.to(predictions.device), phn_lens.to(predictions.device)],
        )

        if not train_mode:
            ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
            sequence = ctc_greedy_decode(pout, lens, blank_id=-1)
            sequence = convert_index_to_lab(sequence, ind2lab)
            phns = undo_padding(phns, phn_lens)
            phns = convert_index_to_lab(phns, ind2lab)
            stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats = {"PER": stats}
            return loss, stats

        return loss

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        per = summarize_error_rate(valid_stats["PER"])
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, per)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        params.checkpointer.save_and_keep_only(
            meta={"PER": per},
            importance_keys=[ckpt_recency, lambda c: -c.meta["PER"]],
        )

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets)
        loss = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer(self.modules)
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch):
        inputs, targets = batch
        out = self.compute_forward(inputs, None, train_mode=False)
        loss, stats = self.compute_objectives(out, targets, train_mode=False)
        stats["loss"] = loss.detach()
        return stats


# Prepare data
# prepare = TIMITPreparer(
#     data_folder=params.data_folder,
#     splits=["train", "dev", "test"],
#     save_folder=params.data_folder,
# )
# prepare()
train_set = params.train_loader()
valid_set = params.valid_loader()
first_x, first_y = next(zip(*train_set))

modules = [
    params.encoder_model,
    params.encoder_output,
    params.decoder_model,
    params.decoder_output,
    params.output,
]
if hasattr(params, "augmentation"):
    modules.append(params.augmentation)

asr_brain = TransducerBrain(
    modules=modules,
    optimizer=params.optimizer,
    first_inputs=[first_x, first_y],
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
