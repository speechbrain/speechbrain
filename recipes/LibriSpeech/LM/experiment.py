#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb

from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token
from speechbrain.data_io.data_io import merge_csvs
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_average

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from librispeech_prepare import prepare_librispeech  # noqa E402

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

modules = torch.nn.ModuleList([params.model])
checkpointer = sb.utils.checkpoints.Checkpointer(
    checkpoints_dir=params.save_folder,
    recoverables={
        "model": modules,
        "optimizer": params.optimizer,
        "scheduler": params.lr_annealing,
        "counter": params.epoch_counter,
    },
)


# Define training procedure
class LM(sb.core.Brain):
    def compute_forward(self, y, stage="train", init_params=False):
        ids, chars, char_lens = y

        chars, char_lens = chars.to(params.device), char_lens.to(params.device)

        y_in = prepend_bos_token(chars, bos_index=params.bos_index)
        logits = params.model(y_in, init_params=init_params)
        pout = params.log_softmax(logits)
        return pout

    def compute_objectives(self, predictions, targets, stage="train"):
        pout = predictions
        ids, chars, char_lens = targets
        chars, char_lens = chars.to(params.device), char_lens.to(params.device)

        abs_length = torch.round(char_lens * chars.shape[1])

        # Append eos token at the end of the label sequences
        chars_with_eos = append_eos_token(
            chars, length=abs_length, eos_index=params.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / chars_with_eos.shape[1]
        loss = params.compute_cost(pout, chars_with_eos, length=rel_length)

        return loss, {}

    def fit_batch(self, batch):
        inputs = batch[0]
        predictions = self.compute_forward(inputs)
        loss, stats = self.compute_objectives(predictions, inputs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def evaluate_batch(self, batch, stage="valid"):
        inputs = batch[0]
        predictions = self.compute_forward(inputs, stage=stage)
        loss, stats = self.compute_objectives(predictions, inputs, stage=stage)
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        val_loss = summarize_average(valid_stats["loss"])
        old_lr, new_lr = params.lr_annealing(
            [params.optimizer], epoch, val_loss
        )
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        checkpointer.save_and_keep_only(
            meta={"loss": val_loss},
            importance_keys=[ckpt_recency, lambda c: -c.meta["loss"]],
        )


# Prepare data
prepare_librispeech(
    data_folder=params.data_folder,
    splits=[
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
    ],
    save_folder=params.data_folder,
)
merge_csvs(
    data_folder=params.data_folder,
    csv_lst=[
        "train-clean-100.csv",
        "train-clean-360.csv",
        "train-other-500.csv",
    ],
    merged_csv="train-960.csv",
)

# Using train-clean-100 to generate the same label_dict as ASR model
# Temporary solution
_ = params.label_loader()
train_set = params.train_loader()
valid_set = params.valid_loader()
first_y = next(iter(train_set))

lm_brain = LM(
    modules=modules, optimizer=params.optimizer, first_inputs=first_y,
)
if params.multigpu:
    params.model = torch.nn.DataParallel(params.model)
# Load latest checkpoint to resume training
checkpointer.recover_if_possible()
lm_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(lambda c: -c.meta["loss"])
test_stats = lm_brain.evaluate(params.valid_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)
