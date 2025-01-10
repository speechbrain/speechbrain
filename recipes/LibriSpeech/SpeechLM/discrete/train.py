#!/usr/bin/env/python3
"""
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py hparams/ssl.yaml \
--data_folder $SLURM_TMPDIR/LibriSpeech/ --tokens_folder $SCRATCH/results/hubert25hzl11_test/librispeech/  --num_workers 4 \
--num_codebooks 1  --eval_precision=bf16 --batch_size=16 --block_size=2048 --grad_accumulation_factor=8 \
--max_grad_norm=1.0 --optimizer_step_limit 10_000 --number_of_epochs=500 --tqdm_colored_bar \
--output_folder $SCRATCH/results/speech_lm/hubert25hzl11/ --codebook_size=500 --skip_prep=True


HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py hparams/ssl.yaml \
--data_folder $SLURM_TMPDIR/LibriSpeech/ --tokens_folder $SCRATCH/results/ST/librispeech/  --num_workers 4 \
--num_codebooks 1  --eval_precision=bf16 --batch_size=16 --block_size=2048 --grad_accumulation_factor=8 \
--max_grad_norm=1.0 --optimizer_step_limit 10_000 --number_of_epochs=500 --tqdm_colored_bar \
--output_folder $SCRATCH/results/speech_lm/ST_test/ --codebook_size=1024 --skip_prep=True


HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py hparams/ssl.yaml \
--data_folder $SLURM_TMPDIR/LibriSpeech/ --tokens_folder /scratch/adelmou/results/hubert25hzl11_last/save/librispeech/  --num_workers 4 \
--num_codebooks 1  --eval_precision=bf16 --batch_size=16 --block_size=2048 --grad_accumulation_factor=8 \
--max_grad_norm=1.0 --optimizer_step_limit 10_000 --number_of_epochs=500 --tqdm_colored_bar \
--output_folder $SCRATCH/results/speech_lm/hubert25hzl11_collapsed/ --codebook_size=500 --skip_prep=True \
--collapse_repeated_tokens=True

TODOS:
- wandb integration here + file logging
- fix bug of empty autograd 
- refactor codebook pattern name and make sure to better understand what is going on
- some batch may have padding tokens, we should remove them.
"""

import os
import sys
import torch
import torchaudio
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from speechbrain.lobes.models.huggingface_transformers.discrete_speechlm import (
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
    InterleavedCodebookPattern,
)
from torch.nn import functional as F
import torch.nn as nn 
logger = logging.getLogger(__name__)


# Define training procedure
class GSLM(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        inputs, _ = batch.tokens
        # exit()
        # (B, T, C) -> (B, C, T)
        inputs = inputs.permute(0, 2, 1)
        labels = inputs[:, :, 1:]
        inputs = inputs[:, :, :-1]
        
        block_size = self.hparams.config.block_size
        assert inputs.size(-1) <= block_size, 'input length >= block_size'
 
        patterned_inp = self.hparams.codebook_pattern.apply_delay_pattern(inputs)
        # print(patterned_inp.shape)
        # print((patterned_inp == self.hparams.pad_token).sum())
        # print(self.hparams.pad_token)
        # exit()
        logits = self.modules.model(patterned_inp)
        # print(logits.shape)
        undelayed_logits, undelayed_logits_mask = self.hparams.codebook_pattern.undelay_logits(logits)
        # print(undelayed_logits.shape)
        # exit()
        return undelayed_logits, undelayed_logits_mask, labels, 

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        # unpack batch
        logits, logits_mask, labels, = predictions
        logits = logits.float()
        # here we should use the length of the sequence
        lm_loss = F.cross_entropy(
            logits[logits_mask],
            labels[logits_mask],
            ignore_index=hparams["pad_token"], 
            reduction='mean',
        )
        return lm_loss


    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            steps = self.step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": old_lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_loss, "epoch": epoch},
                min_keys=["loss"],
            )
        
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 1. Define tokens pipeline:
    tokens_loader = hparams["tokens_loader"]
    num_codebooks = hparams["num_codebooks"]
    
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("tokens")
    def tokens_pipeline(id):
        # (T, C)
        tokens = tokens_loader.tokens_by_uttid(id, num_codebooks=num_codebooks)
        # concat eos_token to the end of the sequence
        eos_token = torch.full((1, tokens.size(-1)), hparams['eos_token'], dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([tokens, eos_token], dim=0)
        # TODO: collapse repeated tokens if hparams['collapse_repeated_tokens'] is True
        # Remove consecutive repeated tokens if enabled
        if hparams.get('collapse_repeated_tokens', False) and num_codebooks == 1:
            # this block led a reduction from 277 batches to 213 batches
            # with semantic tokenizer
            # Get indices where values change
            changes = (tokens[1:] != tokens[:-1]).any(dim=-1)
            # Add True at start to always keep first token
            mask = torch.cat([torch.tensor([True], device=tokens.device), changes])
            # Use mask to keep only tokens that differ from previous
            tokens = tokens[mask]
        return tokens
        
    sb.dataio.dataset.add_dynamic_item(datasets, tokens_pipeline)

    # 2. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "tokens"],
    )

    train_data = sb.dataio.dataset.PackedDatasetWrapper(
        train_data, 
        # we add 1 to the block size as we are going to use the same input
        # for both, input and target. Since the model can take up to block_size tokens,
        # we need to add 1 to maximise efficiency.
        hparams['block_size'] + 1, 
        token_key="tokens", 
        # pad_token_id=hparams["pad_token"]
    )
    valid_data = sb.dataio.dataset.PackedDatasetWrapper(
        valid_data, 
        hparams['block_size'] + 1, 
        token_key="tokens", 
        # pad_token_id=hparams["pad_token"]
    )
    return train_data, valid_data, test_datasets


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, = dataio_prepare(
        hparams
    )

    # Trainer initialization
    gslm_brain = GSLM(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["opt_class"],
    )

    print(f"Total number of tokens per opt step: {hparams['block_size'] * hparams['batch_size'] * hparams['grad_accumulation_factor']}")

    # ~ takes a minute or so to compile
    gslm_brain.modules.model = torch.compile(gslm_brain.modules.model)

    # Training
    gslm_brain.fit(
        gslm_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # report the NLL on the test datasets (i.e. how likely those inputs are to be generated by the model)
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        gslm_brain.evaluate(
            test_datasets[k],
            min_key="loss",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )