#!/usr/bin/env python3
"""Recipe for pretraining wav2vec2 (https://arxiv.org/abs/2006.11477).
See config file for model definition.
See the readme of the recipe for advices on the pretraining that may appear
a bit challenging depending on your available resources.

To run this recipe call python train_sb_wav2vec2.py hparams/wav2vec2_base.yaml --find_unused_parameters --max_grad_norm 0.0

Authors
    * Rudolf Braun 2022
    * Guillermo Cámbara 2022
    * Titouan Parcollet 2022
"""

import logging
import sys
import time
from functools import partial

import speechbrain as sb
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from hyperpyyaml import load_hyperpyyaml

from speechbrain import Stage
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.lobes.models.wav2vec import w2v_mask_collate_fn
from speechbrain.lobes.models.wav2vec import sample_negatives

logger = logging.getLogger(__name__)


class W2V2Brain(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Computes forward pass through wav2vec model and returns encoded and
        target embeddings as well as other metrics of interest.
        """
        wavs, wav_lens, mask = batch
        wavs, wav_lens, mask = (
            wavs.to(self.device),
            wav_lens.to(self.device),
            mask.to(self.device),
        )
        batch_size = wavs.size(0)

        # Mormalisation already done in dataloader
        # 1. Go through features extractor
        latents = self.modules.latent_extractor(wavs, normalize_signal=False)

        # 2. Go through latent (Transformer).
        results = self.modules.latent_encoder(
            latents, mask=mask, wav_lens=wav_lens,
        )

        embeddings = results["embeddings"]

        # 3. Mask some of the latent and projection
        embeddings = embeddings[mask]
        embeddings = self.modules.feat_proj(embeddings)
        results["embeddings"] = embeddings.view(
            batch_size, -1, embeddings.size(1)
        )

        latents = latents[mask].view(batch_size, -1, latents.size(2))

        # 4. Apply the quantiser as well
        targets, meta = self.modules.target_quantiser(latents)
        results.update(meta)
        results["targets"] = targets
        return results

    def compute_objectives(self, forward_outputs, batch, stage):
        """Samples negatives, computes contrastive loss and accuracy.
        """

        embeddings = forward_outputs["embeddings"]
        targets = forward_outputs["targets"]

        negs = sample_negatives(targets, self.hparams.num_negatives)

        loss, accuracy = self.hparams.loss(embeddings, targets, negs)

        # This is only used for logging purpose
        if stage != sb.Stage.TRAIN:
            self.acc_metric.append(accuracy)

        objectives = {
            "loss": loss,
            "accuracy": accuracy,
            "num_masked": forward_outputs["num_masked"],
            "ratio_masked": forward_outputs["ratio_masked"],
        }
        if (
            "diversity_loss" in forward_outputs
        ):  # only quantised model has these
            objectives.update(
                {
                    "diversity_loss": forward_outputs["diversity_loss"],
                    "prob_perplex": forward_outputs["prob_perplex"],
                    "code_perplex": forward_outputs["code_perplex"],
                    "num_vars": forward_outputs["num_vars"],
                    "temp": forward_outputs["temp"],
                }
            )

        # Compute the loss given the original equation from the paper
        loss = objectives["loss"]
        if self.hparams.diversity_loss_weight == 0.0:
            objectives["backprop_loss"] = loss
        else:
            objectives["backprop_loss"] = (
                loss
                + objectives["diversity_loss"]
                * self.hparams.diversity_loss_weight
                * objectives["num_masked"]
            )
        return objectives

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with self.no_sync(not should_step):
                with torch.cuda.amp.autocast():
                    outputs = self.compute_forward(batch, Stage.TRAIN)
                    objectives = self.compute_objectives(
                        outputs, batch, Stage.TRAIN
                    )

                self.scaler.scale(
                    objectives["backprop_loss"] / self.grad_accumulation_factor
                ).backward()

                objectives["total_loss"] = objectives["backprop_loss"].detach()
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    if self.check_gradients(objectives["backprop_loss"]):
                        self.scaler.step(self.optimizer)
                    self.optimizer.zero_grad()
                    self.optimizer_step += 1
                    self.scaler.update()
        else:
            with self.no_sync(not should_step):
                outputs = self.compute_forward(batch, Stage.TRAIN)
                objectives = self.compute_objectives(
                    outputs, batch, Stage.TRAIN
                )

                (
                    objectives["backprop_loss"] / self.grad_accumulation_factor
                ).backward()
                objectives["total_loss"] = objectives["backprop_loss"].detach()

                if should_step:
                    if self.check_gradients(objectives["backprop_loss"]):
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.optimizer_step += 1

        if should_step:
            self.on_fit_batch_end(objectives)

        return objectives["backprop_loss"].detach()

    def on_fit_batch_end(self, objectives):
        """ Called after fit_batch(), updates learning rate and does per-step logging. """
        if isinstance(self.modules.target_quantiser, DistributedDataParallel):
            w2v_model = self.modules.target_quantiser.module
        else:
            w2v_model = self.modules.target_quantiser

        w2v_model.quantiser.update_temp(self.optimizer_step)

        self.hparams.lr_scheduler(self.optimizer, self.optimizer_step)

        # Perform step-wise logging
        if (
            hasattr(self.hparams, "log_interval")
            and self.optimizer_step % self.hparams.log_interval == 0
        ):

            # Create a dictionary and fill it with everything we
            # want to log such as contrastive loss, diversity loss,
            # learning rate etc.
            log_dct = {
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in objectives.items()
            }
            current_lr = self.optimizer.param_groups[0]["lr"]
            log_dct["steps"] = self.optimizer_step
            log_dct["lr"] = current_lr
            log_dct["avg_loss"] = self.avg_train_loss

            if hasattr(self, "time_last_log"):
                run_time_since_last_log = time.time() - self.time_last_log
                log_dct["run_time"] = run_time_since_last_log
            self.time_last_log = time.time()

            if sb.utils.distributed.if_main_process():
                self.hparams.train_steps_logger.log_stats(stats_meta=log_dct,)

    def evaluate_batch(self, batch, stage):
        """ Returns accuracy on contrastive objective. """
        out = self.compute_forward(batch, stage=stage)
        objectives = self.compute_objectives(out, batch, stage=stage)
        return objectives["backprop_loss"].detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = []

    def on_stage_end(self, stage, stage_loss, epoch=None):

        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID:
            print(self.acc_metric)
            stage_stats["accuracy"] = sum(self.acc_metric) / len(
                self.acc_metric
            )

            self.hparams.train_stage_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "steps": self.optimizer_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                end_of_epoch=True,
                num_to_keep=5,
                meta={"valid_loss": stage_loss},
            )


def dataio_prepare(hparams):
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    # We remove longer and shorter files from the train.
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_shorter_than"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]

    def get_output_lengths(input_lengths):
        """ Function to get the output length of the feature extractor this is
            necessery to compute the masks of wav2vec2.
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        for kernel_size, stride in zip(
            hparams["latentextractor_kernels"],
            hparams["latentextractor_strides"],
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths.to(torch.long)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        assert sig.dim() == 1, sig.dim()

        # Audio normalization
        with torch.no_grad():
            sig = F.layer_norm(sig, sig.shape)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # We create the DynamicBatch Sampler
    train_sampler = DynamicBatchSampler(
        train_data,
        hparams["seconds_per_batch"],
        num_buckets=hparams["train_num_buckets"],
        length_func=lambda x: x["duration"],
        batch_ordering="random",
        shuffle=True,
    )

    # We define the custom collation function that is necessary for w2v2 to
    # generate masks.
    w2v_mask_collate_fn_partial = partial(
        w2v_mask_collate_fn,
        get_out_len_fn=get_output_lengths,
        mask_prob=hparams["mask_prob"],
        mask_length=hparams["mask_length"],
    )

    train_loader_kwargs = {
        "batch_sampler": train_sampler,
        "collate_fn": w2v_mask_collate_fn_partial,
        "num_workers": hparams["train_dataloader_options"]["num_workers"],
        "pin_memory": True,
    }

    valid_loader = SaveableDataLoader(
        valid_data,
        collate_fn=w2v_mask_collate_fn_partial,
        num_workers=hparams["test_dataloader_options"]["num_workers"],
        batch_size=hparams["test_dataloader_options"]["batch_size"],
        pin_memory=True,
    )

    return train_data, valid_loader, train_loader_kwargs


def main():
    logger.setLevel(logging.INFO)
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams.update(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from librispeech_prepare import prepare_librispeech

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

    # Part that matters starts here.
    train_dataset, valid_loader, train_loader_kwargs = dataio_prepare(hparams)

    brain = W2V2Brain(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    brain.fit(
        brain.hparams.epoch_counter,
        train_dataset,
        valid_loader,
        train_loader_kwargs=train_loader_kwargs,
        progressbar=False,
    )


if __name__ == "__main__":
    main()
