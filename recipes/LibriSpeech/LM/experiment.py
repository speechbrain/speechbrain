#!/usr/bin/env python3
import os
import sys
import torch

import speechbrain as sb
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_average
from speechbrain.utils.data_utils import download_file

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from librispeech_prepare import prepare_librispeech  # noqa E402
from librispeech_lm_prepare import prepare_lm_corpus  # noqa E402

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

steps = 0


# Define training procedure
class LM(sb.core.Brain):
    def compute_forward(self, y, stage="train", init_params=False):
        ids, chars, char_lens = y
        index2lab = params.label_loader.label_dict["char"]["index2lab"]
        bpe, _ = params.bpe_tokenizer(
            chars, char_lens, index2lab, task="encode", init_params=init_params
        )
        bpe = bpe.to(params.device)

        y_in = prepend_bos_token(bpe, bos_index=params.bos_index)
        logits = params.model(y_in, init_params=init_params)
        pout = params.log_softmax(logits)
        return pout

    def compute_objectives(self, predictions, targets, stage="train"):
        pout = predictions
        ids, chars, char_lens = targets
        index2lab = params.label_loader.label_dict["char"]["index2lab"]
        bpe, bpe_lens = params.bpe_tokenizer(
            chars, char_lens, index2lab, task="encode"
        )
        bpe, bpe_lens = bpe.to(params.device), bpe_lens.to(params.device)

        abs_length = torch.round(bpe_lens * bpe.shape[1])

        # Append eos token at the end of the label sequences
        bpe_with_eos = append_eos_token(
            bpe, length=abs_length, eos_index=params.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / bpe_with_eos.shape[1]
        loss = params.compute_cost(pout, bpe_with_eos, length=rel_length)

        return loss, {}

    def fit_batch(self, batch):
        inputs = batch[0]
        predictions = self.compute_forward(inputs)
        loss, stats = self.compute_objectives(predictions, inputs)
        (loss / params.accu_steps).backward()
        global steps
        steps += 1
        if steps % params.accu_steps == 0:
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

    def load_tokenizer(self):
        save_model_path = params.save_folder + "/tok_unigram.model"
        save_vocab_path = params.save_folder + "/tok_unigram.vocab"

        if hasattr(params, "tok_mdl_file"):
            download_file(
                params.tok_mdl_file, save_model_path, replace_existing=True
            )
            params.bpe_tokenizer.sp.load(save_model_path)
        if hasattr(params, "tok_voc_file"):
            download_file(
                params.tok_voc_file, save_vocab_path, replace_existing=True
            )


# Prepare data
prepare_librispeech(
    data_folder=params.data_folder,
    splits=params.train_splits + [params.dev_split],
    merge_lst=params.train_splits,
    merge_name=params.csv_label,
    save_folder=params.data_folder,
)

prepare_lm_corpus(
    data_folder=params.data_folder,
    save_folder=params.data_folder,
    filename=params.filename,
)

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

lm_brain.load_tokenizer()
lm_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(lambda c: -c.meta["loss"])
test_stats = lm_brain.evaluate(params.valid_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)
