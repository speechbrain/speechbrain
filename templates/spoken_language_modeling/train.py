#!/usr/bin/env/python3
"""
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py hparams/ssl.yaml \
--data_folder $SLURM_TMPDIR/LibriSpeech/ --tokens_folder $SCRATCH/results/dac/librispeech/  --num_workers 4 \
--num_codebooks 8  --eval_precision=bf16 --batch_size=16 --block_size=2048 --grad_accumulation_factor=8 \
--max_grad_norm=1.0 --optimizer_step_limit 10_000 --number_of_epochs=500 --tqdm_colored_bar \
--output_folder $SCRATCH/results/speech_lm/DAC_overfit_dev_clean/ --codebook_size=500 --skip_prep=True \
--experiment_name=DAC_overfit_dev_clean

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py hparams/ssl.yaml --data_folder $SLURM_TMPDIR/LibriSpeech/ --tokens_folder $SCRATCH/results/dac/librispeech/  --num_workers 4 --num_codebooks 12  --eval_precision=bf16 --batch_size=16 --block_size=2048 --grad_accumulation_factor=8 --max_grad_norm=1.0 --optimizer_step_limit 10_000 --number_of_epochs=500 --tqdm_colored_bar --output_folder $SCRATCH/results/speech_lm/DAC_overfit_dev_clean/ --codebook_size=1024 --skip_prep=True --experiment_name=DAC_overfit_dev_clean


HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py hparams/ssl.yaml \
--data_folder $SLURM_TMPDIR/LibriSpeech/ --tokens_folder $SCRATCH/results/ST/librispeech/  --num_workers 4 \
--num_codebooks 8  --eval_precision=bf16 --batch_size=32 --block_size=2048 --grad_accumulation_factor=8 \
--max_grad_norm=1.0 --optimizer_step_limit 10_000 --number_of_epochs=500 --tqdm_colored_bar \
--output_folder $SCRATCH/results/speech_lm/ST_test/ --codebook_size=1024 --skip_prep=True \
--experiment_name=ST_overfit_dev_clean


HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py hparams/ssl.yaml \
--data_folder $SLURM_TMPDIR/LibriSpeech/ --tokens_folder /scratch/adelmou/results/hubert25hzl11_last/save/librispeech/  --num_workers 4 \
--num_codebooks 1  --eval_precision=bf16 --batch_size=16 --block_size=2048 --grad_accumulation_factor=1 \
--max_grad_norm=1.0 --optimizer_step_limit 10_000 --number_of_epochs=500 --tqdm_colored_bar \
--output_folder $SCRATCH/results/speech_lm/hubert25hzl11_collapsed_no_overfit_back_to_work/ --codebook_size=500 --skip_prep=True \
--collapse_repeated_tokens=True --experiment_name=hubert25hzl11_collapsed_no_overfit

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py hparams/ssl.yaml \
--data_folder $SLURM_TMPDIR/LibriSpeech/ --tokens_folder /scratch/adelmou/results/hubert25hzl11_last/save/librispeech/  --num_workers 4 \
--num_codebooks 1  --eval_precision=bf16 --batch_size=32 --block_size=2048 --grad_accumulation_factor=1 \
--max_grad_norm=1.0 --optimizer_step_limit 10_000 --number_of_epochs=500 --tqdm_colored_bar \
--output_folder $SCRATCH/results/speech_lm/hubert25hzl11_collapsed_overfit_v2/ --codebook_size=500 --skip_prep=True \
--collapse_repeated_tokens=True --experiment_name=hubert25hzl11_collapsed_overfit_v2


TODOS:
- calculate the MFU and report it
- refactor codebook pattern name and make sure to better understand what is going on
- add author + header
- this shouldn't be here. I think it should be a template.
- add the end of one epoch, I think we should reshuffle the batches.
- the scheduler should be based on opt steps not epochs
"""

import logging
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml
from torch.nn import functional as F

import speechbrain as sb
from torch.distributed import destroy_process_group

logger = logging.getLogger(__name__)


# Define training procedure
class SpeechLM(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        inputs, _ = batch.tokens
        inputs = inputs.to(torch.long)
        inputs = inputs.permute(0, 2, 1)
        labels = inputs[:, :, 1:]
        inputs = inputs[:, :, :-1]
        
        block_size = self.hparams.config.block_size
        assert inputs.size(-1) <= block_size, "input length > block_size"

        delayed_tokens = self.hparams.codebook_pattern.apply_delay_pattern(
            inputs
        )
        logits = self.modules.model(delayed_tokens)
        undelayed_logits, undelayed_logits_mask = (
            self.hparams.codebook_pattern.undelay_logits(logits)
        )

        return (
            undelayed_logits,
            undelayed_logits_mask,
            labels,
        )

    def compute_objectives(self, predictions, batch, stage):
        # unpack batch
        (
            logits,
            logits_mask,
            labels,
        ) = predictions
        lm_loss = F.cross_entropy(
            logits[logits_mask].float(),  # cast logits to fp32
            labels[logits_mask],
            ignore_index=hparams["pad_token"],
            reduction="mean",
        )
        return lm_loss

    def on_fit_batch_start(self, batch, should_step):
        """Called before ``fit_batch()``.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        outputs : list or dictionary of torch.Tensors
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        if should_step:
            self.hparams.lr_annealing(self.optimizer)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after ``fit_batch()``.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        outputs : list or dictionary of torch.Tensors
            Returned value of compute_forward().
        loss : torch.Tensor
            Returned value of compute_objectives().
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        if (
            hasattr(self.hparams, "wandb_logger")
            and (self.optimizer_step + 1) % self.hparams.save_interval == 0
            and should_step
        ):
            wb_loss = loss.detach().cpu().item()
            lr = self.optimizer.param_groups[0]['lr']
            stats_meta = {"train_loss": wb_loss, "steps": self.optimizer_step, "lr": lr}
            self.hparams.wandb_logger.log_stats(
                stats_meta=stats_meta,
            )
        if (self.optimizer_step + 1) % self.hparams.save_interval == 0 and should_step:
            print(f"Saving checkpoint at step {self.optimizer_step}")
            ckpt_loss = loss.detach().cpu().item()
            self.checkpointer.save_and_keep_only(
                meta={
                    "loss": ckpt_loss,
                },
                name=f"iter_{self.optimizer_step}",
                num_to_keep=5,
                min_keys=["loss"],
                end_of_epoch=False,
            )

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.VALID:
            epoch_stats = {
                "epoch": epoch,
                "lr": old_lr,
                "steps": self.optimizer_step,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if hasattr(self.hparams, "wandb_logger"):
                self.hparams.wandb_logger.log_stats(
                    stats_meta=epoch_stats,
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
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
        csv_path=hparams["valid_csv"],
    )
    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file,
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
        eos_token = torch.full(
            (1, tokens.size(-1)),
            hparams["eos_token"],
            dtype=tokens.dtype,
            device=tokens.device,
        )
        tokens = torch.cat([tokens, eos_token], dim=0)
        return tokens

    sb.dataio.dataset.add_dynamic_item(datasets, tokens_pipeline)

    # 2. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "tokens"],
    )
    print(f"Train data length: {len(train_data)}")
    print(f"Valid data length: {len(valid_data)}")
    train_data = sb.dataio.dataset.PackedDatasetWrapper(
        train_data,
        # we add 1 to the block size as we are going to use the same input
        # for both, input and target. Since the model can take up to block_size tokens,
        # we need to add 1 to maximise efficiency.
        hparams["block_size"] + 1,
        token_key="tokens",
        # num_training_steps=hparams["num_training_steps"] * hparams["batch_size"],
    )
    # valid_data = sb.dataio.dataset.PackedDatasetWrapper(
    #     valid_data,
    #     hparams["block_size"] + 1,
    #     token_key="tokens",
    # )
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

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
    ) = dataio_prepare(hparams)

    # Trainer initialization
    speechlm_brain = SpeechLM(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["opt_class"],
    )

    print(
        f"Total number of tokens per opt step: {hparams['block_size'] * hparams['batch_size'] * hparams['grad_accumulation_factor']}"
    )

    # Watch model with wandb if wandb logger is set
    if hasattr(speechlm_brain, "wandb_logger"):
        speechlm_brain.wandb_logger.wandb.watch(
            speechlm_brain.modules.model, log="all"
        )
    # ~ takes a minute or so to compile
    speechlm_brain.modules.model = torch.compile(speechlm_brain.modules.model)

    # Training
    speechlm_brain.fit(
        speechlm_brain.hparams.epoch_counter,
        train_data,
        # valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        # valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # # report the NLL on the test datasets (i.e. how likely those inputs are to be generated by the model)
    # for k in test_datasets.keys():  # keys are test_clean, test_other etc
    #     speechlm_brain.evaluate(
    #         test_datasets[k],
    #         min_key="loss",
    #         test_loader_kwargs=hparams["test_dataloader_opts"],
    #     )

    destroy_process_group()