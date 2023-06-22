#!/usr/bin/env python3

import sys
import json
import itertools
import torch
import copy
import pathlib as pl
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import scalarize
import torch
import torchaudio
import os
import numpy as np
import random

from utils.embedding import EmbeddingManager
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

#torch.autograd.set_detect_anomaly(True)


# Define training procedure
class S2U(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig  # audio
        tokens_bos, _ = batch.unit_bos  # speech unit _bos

        # replace pad_index by 0 in the signal
        wavs[wavs == self.hparams.pad_index] = 0.0

        # compute features
        feats = self.modules.wav2vec2(wavs, wav_lens)

        # dimensionality reduction
        src = self.modules.enc(feats)

        # Patch to allow DP or DDP training
        if isinstance(self.modules.transformer, DataParallel):
            dec_out = self.modules.transformer.module.forward_decoder_only(
                src, tokens_bos, pad_idx=self.hparams.pad_index
            )
        elif isinstance(self.modules.transformer, DistributedDataParallel):
            dec_out = self.modules.transformer.module.forward_decoder_only(
                src, tokens_bos, pad_idx=self.hparams.pad_index
            )
        else:
            dec_out = self.modules.transformer.forward_decoder_only(
                src, tokens_bos, pad_idx=self.hparams.pad_index
            ) 

        # logits and softmax
        pred = self.modules.seq_lin(dec_out)
        p_seq = self.hparams.log_softmax(pred)
        
        hyps = None
        if stage == sb.Stage.VALID:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                hyps, _ = self.hparams.valid_search(src.detach(), wav_lens)

        return p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
            """Computes the loss given predictions and targets."""
            (p_seq, wav_lens, hyps) = predictions
            tokens_eos, tokens_eos_lens = batch.unit_eos
            ids = batch.id
            
            # st loss
            loss = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens)

            if stage != sb.Stage.TRAIN:
                # compute the accuracy of the one-step-forward prediction
                self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
                
                current_epoch = self.hparams.epoch_counter.current
                if current_epoch % self.hparams.valid_search_interval == 0:
                    self.wer_metric_st_greedy.append(ids, hyps, tokens_eos, target_len=tokens_eos_lens)

            return loss

    def init_optimizers(self):
        # Initializes the wav2vec2 optimizer if the model is not wav2vec2_frozen
        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
        self.model_optimizer = self.hparams.opt_class(
            self.hparams.model.parameters()
        )

    def zero_grad(self, set_to_none=False):
        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer.zero_grad(set_to_none)
        self.model_optimizer.zero_grad(set_to_none)

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0
        if self.bfloat16_mix_prec:
            with torch.autocast(
                device_type=torch.device(self.device).type,
                dtype=torch.bfloat16,
            ):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

        with self.no_sync(not should_step):
            (loss / self.grad_accumulation_factor).backward()
        if should_step:
            if self.check_gradients(loss):
                if not self.hparams.wav2vec2_frozen:  # if wav2vec2 is not frozen
                    self.wav2vec_optimizer.step()
                self.model_optimizer.step()

            if not self.hparams.wav2vec2_frozen:
                self.wav2vec_optimizer.zero_grad()
            self.zero_grad()
            self.optimizer_step += 1

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        if should_step:
            # anneal model lr every update
            self.hparams.noam_annealing(self.model_optimizer)

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric_st_greedy = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss

        else:  # valid or test
            stage_stats = {"loss": stage_loss}
            stage_stats["accuracy_st"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                stage_stats["wer_st_greedy"] = self.wer_metric_st_greedy.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            current_epoch = self.hparams.epoch_counter.current
            lr = self.hparams.noam_annealing.current_lr
            lr_wav2vec = 0.0 # self.wav2vec_optimizer.param_groups[-1]["lr"]

            if not self.hparams.wav2vec2_frozen:
                (
                    lr_wav2vec,
                    new_lr_wav2vec,
                ) = self.hparams.wav2vec_annealing(stage_stats["accuracy_st"])
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": current_epoch, "lr": lr, "lr_wav2vec": lr_wav2vec},
                train_stats={"loss": self.train_stats},
                valid_stats=stage_stats,
            )

            if self.hparams.use_wandb:
                self.wandb_logger.log_stats(
                stats_meta={"epoch": current_epoch, "lr": lr},
                train_stats={"loss": self.train_stats},
                valid_stats=stage_stats,
                )

            # create checkpoing
            meta = {"epoch": current_epoch, "loss": stage_stats["loss"], "accuracy_st": stage_stats["accuracy_st"]}
            name = "checkpoint_epoch" + str(current_epoch)

            self.checkpointer.save_and_keep_only(
                meta=meta, name=name, num_to_keep=10, max_keys=["accuracy_st"]
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if self.hparams.use_wandb:
                self.wandb_logger.log_stats(
                    {"Epoch loaded": self.hparams.epoch_counter.current},
                    test_stats=stage_stats,
                )

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    
    audio_folder = pl.Path(hparams["audio_folder"])
    
    units_loader = EmbeddingManager(hparams["units_folder"])
    
    # Define audio pipeline. In this case, we simply read the audio contained
    # in the variable src_audio with the custom reader.
    @sb.utils.data_pipeline.takes("src_audio")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torch.FloatTensor(sig)
        return sig

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("unit_bos", "unit_eos")
    def unit_pipeline(clip_id):
        #clip_id = clip_id.split('_')[-1]
        unit = units_loader.get_embedding_by_clip(clip_id)
        unit = torch.LongTensor(unit)
        unit = torch.unique_consecutive(unit)
        unit_bos = torch.cat((torch.LongTensor([hparams["bos_index"]]), unit))
        yield unit_bos
        unit_eos = torch.cat((unit, torch.LongTensor([hparams["eos_index"]])))
        yield unit_eos
    
    datasets = {}
    for split in hparams["splits"]:
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{split}_json"],
            dynamic_items=[audio_pipeline, unit_pipeline],
            output_keys=[
                "id",
                "sig",
                "duration",
                "unit_bos",
                "unit_eos"
            ],
        )

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="duration"
        )
        datasets["valid"] = datasets["valid"].filtered_sorted(
            sort_key="duration"
        )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False
        
    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="duration", reverse=True
        )
        datasets["valid"] = datasets["valid"].filtered_sorted(
            sort_key="duration", reverse=True
        )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False
        
    elif hparams["sorting"] == "random":
        # use smaller dataset to debug the model
        # hparams["train_dataloader_opts"]["shuffle"] = True
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
        
    # Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            datasets["train"],
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
            max_batch_ex=dynamic_hparams["max_batch_ex"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            datasets["valid"],
            dynamic_hparams["max_batch_len_val"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )
    
    return datasets, (train_batch_sampler, valid_batch_sampler)


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        
    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets, samplers = dataio_prepare(hparams)
    (train_bsampler, valid_bsampler) = samplers
    
    s2u_brain = S2U(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if hparams["use_wandb"]:
        from utils.logger import WandBLogger
        s2u_brain.wandb_logger = WandBLogger(**hparams["logger_opts"])
        
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    s2u_brain.fit(
        s2u_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )