#!/usr/bin/env python3
"""Recipe for pretraining wav2vec2. See config file for model definition.

To run this recipe call python train.py hparams/train_wav2vec.yaml --find_unused_parameters --max_grad_norm 0.0

Authors
    * Rudolf Braun 2022
    * Guillermo Cámbara 2022
"""

import os
import os.path as osp
import logging
import sys
import time
import math
from functools import partial

import speechbrain as sb
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from speechbrain.utils.data_utils import batch_pad_right
from hyperpyyaml import load_hyperpyyaml

from speechbrain import Stage
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import (
    DynamicBatchSampler,
    DistributedSamplerWrapper,
)
from speechbrain.lobes.models.wav2vec import compute_mask


PYTHON_VERSION_MAJOR = 3
PYTHON_VERSION_MINOR = 7


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
        B = wavs.size(0)
        # normalisation already done in dataloader
        latents = self.modules.latent_extractor(wavs, normalize_signal=False)

        disable_halfprec = (
            self.hparams.disable_halfprec
            if hasattr(self.hparams, "disable_halfprec")
            else False
        )
        results = self.modules.latent_encoder(
            latents,
            mask=mask,
            wav_lens=wav_lens,
            disable_halfprec=disable_halfprec,
        )

        embeddings = results["embeddings"]

        embeddings = embeddings[mask]
        embeddings = self.modules.feat_proj(embeddings)
        results["embeddings"] = embeddings.view(B, -1, embeddings.size(1))

        latents = latents[mask].view(B, -1, latents.size(2))
        targets, meta = self.modules.target_quantiser(latents)
        results.update(meta)
        results["targets"] = targets
        return results

    def compute_objectives(self, forward_outputs, batch, stage):
        """Samples negatives, computes contrastive loss and accuracy.
        """

        embeddings = forward_outputs["embeddings"]
        targets = forward_outputs["targets"]
        negs = sample_negatives(targets, 100)

        loss, accuracy = self.hparams.loss(embeddings, targets, negs)

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
                loss = objectives["loss"]
                if self.hparams.diversity_loss_weight == 0.0:
                    backprop_loss = loss
                else:
                    backprop_loss = (
                        loss
                        + objectives["diversity_loss"]
                        * self.hparams.diversity_loss_weight
                        * objectives["num_masked"]
                    )
                self.scaler.scale(
                    backprop_loss / self.grad_accumulation_factor
                ).backward()

                objectives["total_loss"] = backprop_loss.detach()
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    if self.check_gradients(loss):
                        self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer_step += 1
                    self.optimizer.zero_grad()
        else:
            with self.no_sync(not should_step):
                outputs = self.compute_forward(batch, Stage.TRAIN)
                objectives = self.compute_objectives(
                    outputs, batch, Stage.TRAIN
                )
                loss = objectives["loss"]
                if self.hparams.diversity_loss_weight == 0.0:
                    backprop_loss = loss
                else:
                    backprop_loss = (
                        loss
                        + objectives["diversity_loss"]
                        * self.hparams.diversity_loss_weight
                        * objectives["num_masked"]
                    )
                (backprop_loss / self.grad_accumulation_factor).backward()
                objectives["total_loss"] = backprop_loss.detach()
                if should_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.optimizer_step += 1

        if should_step:
            self.on_fit_batch_end(objectives)

        return objectives["loss"].detach()

    def on_fit_batch_end(self, objectives):
        """ Called after fit_batch(), updates learning rate and does logging. """
        if isinstance(self.modules.target_quantiser, DistributedDataParallel):
            w2v_model = self.modules.target_quantiser.module
        else:
            w2v_model = self.modules.target_quantiser
        w2v_model.quantiser.update_temp(self.optimizer_step)

        self.hparams.lr_scheduler(self.optimizer, self.optimizer_step)

        if (
            hasattr(self.hparams, "log_interval")
            and self.optimizer_step % self.hparams.log_interval == 0
        ):

            log_dct = {
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in objectives.items()
            }
            current_lr = self.optimizer.param_groups[0]["lr"]
            log_dct["lr"] = current_lr
            log_dct["avg_loss"] = self.avg_train_loss

            if hasattr(self, "time_last_log"):
                run_time_since_last_log = time.time() - self.time_last_log
                log_dct["stats/run_time"] = run_time_since_last_log
            self.time_last_log = time.time()

            log_str = (
                f"Optimizer step: {self.optimizer_step} - Objectives: {log_dct}"
            )
            logger.info(log_str)

            if self.hparams.use_wandb:
                run_on_main(
                    wandb.log,
                    kwargs={"data": log_dct, "step": self.optimizer_step,},
                )

    def evaluate_batch(self, batch, stage):
        """ Returns accuracy on contrastive objective. """
        out = self.compute_forward(batch, stage=stage)
        objectives = self.compute_objectives(out, batch, stage=stage)
        return objectives["accuracy"].cpu()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.VALID:
            logger.info(
                f"Update: {self.optimizer_step} - valid_accuracy: {stage_loss:.3f}"
            )
            if self.hparams.use_wandb:
                wandb.log(
                    {"valid_acuracy": stage_loss,}, step=self.optimizer_step,
                )
            self.checkpointer.save_and_keep_only(
                end_of_epoch=True,
                num_to_keep=2,
                meta={"valid_acc": stage_loss},
                verbosity=logging.DEBUG,
            )

    def update_average(self, loss, avg_loss):
        if avg_loss == 0.0:
            avg_loss = loss.item()
        else:
            avg_loss = 0.99 * avg_loss + 0.01 * loss.item()
        return avg_loss


def sample_negatives(y, num_neg):
    """ Samples negatives from target tensor y.
    Arguments
    ---------
    y : torch.Tensor
        Tensor of shape (B, T, C)
    num_neg : int
        Number of negatives to sample.
    Returns
    -------
    negs : torch.Tensor
        Negatives in shape (N, B, T, C)
    """
    B, T, C = y.shape
    high = T - 1
    with torch.no_grad():
        targets = torch.arange(T).unsqueeze(-1).expand(-1, num_neg).flatten()
        neg_indcs = torch.randint(low=0, high=high, size=(B, T * num_neg))
        # negative should not be target and to make distribution uniform shift all >
        neg_indcs[neg_indcs >= targets] += 1

    neg_indcs = neg_indcs + torch.arange(B).unsqueeze(1) * high
    y = y.view(-1, C)
    negs = y[neg_indcs.view(-1)]
    negs = negs.view(B, T, num_neg, C).permute(2, 0, 1, 3)  # to N, B, T, C
    return negs


def dataio_prepare(hparams):
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    # we sort training data to speed up training and get better results.
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_shorter_than"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data]

    def get_output_lengths(input_lengths):
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
        with torch.no_grad():
            sig = F.layer_norm(sig, sig.shape)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    w2v_mask_collate_fn_partial = partial(
        w2v_mask_collate_fn,
        get_out_len_fn=get_output_lengths,
        mask_prob=hparams["mask_prob"],
        mask_length=hparams["mask_length"],
        max_dur=hparams["max_sample_dur"],
    )

    train_sampler = DynamicBatchSampler(
        train_data,
        hparams["seconds_per_batch"],
        num_buckets=hparams["train_num_buckets"],
        length_func=lambda x: x["duration"],
        batch_ordering="random",
    )
    if torch.distributed.is_initialized():
        train_sampler = DistributedSamplerWrapper(train_sampler)

    train_loader = SaveableDataLoader(
        train_data,
        batch_sampler=train_sampler,
        collate_fn=w2v_mask_collate_fn_partial,
        num_workers=6,
        pin_memory=True,
    )
    valid_loader = SaveableDataLoader(
        valid_data,
        collate_fn=w2v_mask_collate_fn_partial,
        num_workers=hparams["test_dataloader_options"]["num_workers"],
        batch_size=hparams["test_dataloader_options"]["batch_size"],
        pin_memory=True,
    )

    return train_loader, valid_loader


def w2v_mask_collate_fn(
    samples_lst, get_out_len_fn, mask_prob, mask_length, max_dur=16.0
):
    """ This creates a batch from a list of samples and also creates
    the boolean mask that will be used to mask the inputs of the latent
    encoder. To create the mask we need to know the output shape after the
    latent extractor, therefore the argument `get_out_len_fn`.
    One could also create masks per sample (when loading the audio file) and
    then collate them but at that time one doesn't know the length of the
    shortest sample in the batch (which determines the number of masked frames)
    so it's better this way.
    This function also has the feature to split samples longer than `max_dur` into
    smaller chunks, which helps reduce the memory usage of attention based systems.

    Arguments
    ---------
    samples_lst : list
        List of samples returned by the audio_pipeline.
    get_out_len_fn : function
        Function that calculates length of sample after it passes through feature extractor.
    mask_prob : float
        Approximate percentage of frames to mask.
    mask_length : int
        Number of contiguous frames that will be masked.
    max_dur : float
        Samples longer than this will be split so they're shorter.
    Returns
    -------
    wavs_padded : torch.Tensor, shape (B, T)
        Audio arrays with right-sided padding.
    wav_lens : torch.Tensor, shape (B,)
        For each sample the percentage of the array that is not padding.
    mask : torch.Tensor, shape (B, T)
        Boolean mask to mask frames.
    """
    wav_lst, latent_length_lst = [], []
    # If the sample is longer than `max_dur` create several shorter samples from it.
    # Using max of all samples because if we considered the splitting for each sample separately
    # then it could happen that some samples would be just below `max_dur` and others just above,
    # and splitting those above would result in an unnecessarily large amount of padding.
    max_sample_dur = max(
        sample["sig"].size(0) / 16000 for sample in samples_lst
    )
    will_break = max_sample_dur > max_dur
    splits = math.ceil(max_sample_dur / max_dur)
    for sample in samples_lst:
        sig = sample["sig"]
        if will_break:
            for i in range(splits):
                sample_len_split = math.ceil(sample["sig"].size(0) / splits)
                wav_lst.append(
                    sample["sig"][
                        i * sample_len_split : (i + 1) * sample_len_split
                    ]
                )
                latent_length = get_out_len_fn(
                    torch.as_tensor(wav_lst[-1].size(0))
                ).item()
                latent_length_lst.append(latent_length)
        else:
            wav_lst.append(sig)
            latent_length = get_out_len_fn(torch.as_tensor(sig.size(-1)))
            latent_length_lst.append(latent_length.item())
    bs = len(wav_lst)
    wavs_padded, wav_lens = batch_pad_right(wav_lst)

    batch_time_len = max(latent_length_lst)
    mask = compute_mask(
        (bs, batch_time_len,), latent_length_lst, mask_prob, mask_length
    )
    return (
        torch.as_tensor(wavs_padded),
        torch.as_tensor(wav_lens),
        torch.as_tensor(mask, dtype=torch.bool),
    )


def _setup_wandb(hparams):
    global wandb
    import wandb

    id_file = osp.join(hparams["output_folder"], "id")
    if not hparams["distributed_launch"] or (
        hparams["distributed_launch"] and int(os.environ["RANK"]) == 0
    ):
        # This should run when not using distributed training OR
        # when using and this is the rank 0 node of distributed training.
        dct = {
            k: hparams.get(k)
            for k in hparams.keys()
            if hparams[k].__class__.__name__ in ("str", "int", "float", "bool")
        }  # collects hparams to log
        if os.path.exists(id_file):
            id, name = open(id_file).read().split()
            logger.info(
                f"Loading ID {id} and name {name} for wandb from existing file {id_file}"
            )
            run = wandb.init(
                project="wav2vec", resume="allow", config=dct, id=id, name=name,
            )
        else:
            run = wandb.init(project="wav2vec", resume="allow", config=dct)
            id = run.id
            name = run.name
            logger.info(
                f"Creating new ID {id} and name {name} for wandb, putting in file {id_file}"
            )
            with open(id_file, "w") as fh:
                fh.write(f"{id} {name}")


def main():
    logger.setLevel(logging.INFO)
    print(sys.argv[1:])
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

    if hparams["use_wandb"]:
        _setup_wandb(hparams)

    # Part that matters starts here.
    train_loader, valid_loader = dataio_prepare(hparams)

    brain = W2V2Brain(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    brain.fit(
        brain.hparams.epoch_counter,
        train_loader,
        valid_loader,
        progressbar=False,
    )


if __name__ == "__main__":
    main()
