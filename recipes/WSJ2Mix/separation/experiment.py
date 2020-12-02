#!/usr/bin/python

"""
Recipe to train DP-RNN model on the WSJ0-2Mix dataset

Author:
    * Cem Subakan 2020
    * Mirko Bronzi 2020
"""

import logging
import os
import pprint
import shutil

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import speechbrain as sb
from recipes.minimal_examples.neural_networks.separation.example_conv_tasnet import (
    create_minimal_data,
)
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper


import speechbrain.nnet.schedulers as schedulers
import numpy as np
from tqdm import tqdm
import sys
import csv


logger = logging.getLogger(__name__)


def reset_layer_recursively(layer):
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()
    for child_layer in layer.modules():
        if layer != child_layer:
            reset_layer_recursively(child_layer)


def save_audio_results(
    params, model, test_loader, device, Ntosave=10, justNitems=False
):
    # this package is required for SDR computation
    from mir_eval.separation import bss_eval_sources

    # for some reason speechbrain save method causes clipping, so I am using this one
    from soundfile import write

    save_path = os.path.join(params["output_folder"], "audio_results")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fs = 8000
    all_sdrs = []
    all_sisnrs = []

    csv_columns = [
        "sdr",
        "si-snr",
    ]

    with open(save_path + "/results.csv", "w") as results_csv:
        writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
        writer.writeheader()

        with tqdm(test_loader.get_dataloader(), dynamic_ncols=True) as t:
            for i, batch in enumerate(t):

                inputs = batch[0][1].to(device)
                predictions = model.compute_forward(
                    inputs, stage="test"
                ).detach()
                targets = torch.stack(
                    [
                        batch[i][1].squeeze()
                        for i in range(1, model.modules.masknet.num_spks + 1)
                    ],
                    dim=0,
                )

                sdr, _, _, _ = bss_eval_sources(
                    targets.numpy(), predictions[0].t().cpu().numpy()
                )
                sisnr = get_si_snr_with_pitwrapper(
                    targets.t().unsqueeze(0).to(device), predictions
                )

                row = {"sdr": sdr.mean(), "si-snr": -sisnr.item()}

                all_sdrs.append(sdr.mean())
                all_sisnrs.append(-sisnr.item())
                if i < Ntosave:

                    for ns in range(model.modules.masknet.num_spks):
                        tosave = predictions[0, :, ns].cpu().numpy()
                        mx_val = tosave.max()
                        write(
                            save_path
                            + "/item{}_source{}hat.wav".format(i, ns + 1),
                            tosave / mx_val,
                            fs,
                        )

                        tosave = batch[ns + 1][1].cpu().squeeze().numpy()
                        mx_val = tosave.max()
                        write(
                            save_path
                            + "/item{}_source{}.wav".format(i, ns + 1),
                            tosave / mx_val,
                            fs,
                        )

                    tosave = inputs[0].data.cpu().numpy()
                    mx_val = tosave.max()
                    write(
                        save_path + "/item{}_mixture.wav".format(i),
                        tosave / mx_val,
                        fs,
                    )

                    writer.writerow(row)
                else:
                    if justNitems:
                        break

                t.set_postfix(average_sdr=np.array(all_sdrs).mean())
            row = {
                "sdr": np.array(all_sdrs).mean(),
                "si-snr": np.array(all_sisnrs).mean(),
            }
            writer.writerow(row)

    print("Mean SDR is {}".format(np.array(all_sdrs).mean()))


class SourceSeparationBrain(sb.core.Brain):
    def compute_objectives(self, predictions, targets):
        loss = get_si_snr_with_pitwrapper(targets, predictions)
        return loss

    def fit_batch(self, batch):

        inputs = batch[0][1].to(self.device)
        targets = torch.cat(
            [
                batch[i][1].unsqueeze(-1)
                for i in range(1, self.hparams.MaskNet.num_spks + 1)
            ],
            dim=-1,
        ).to(self.device)

        if self.hparams.limit_training_signal_len:
            randstart = torch.randint(
                0,
                1 + max(0, inputs.shape[1] - self.hparams.training_signal_len),
                (1,),
            ).item()
            targets = targets[
                :, randstart : randstart + self.hparams.training_signal_len, :
            ]
            inputs = inputs[
                :, randstart : randstart + self.hparams.training_signal_len
            ]

        if self.hparams.use_speedperturb:
            targets = targets.permute(0, 2, 1)
            targets = targets.reshape(-1, targets.shape[-1])
            wav_lens = torch.tensor(
                [targets.shape[-1]] * targets.shape[0], device=self.device
            )

            targets = self.hparams.speedperturb_block(targets, wav_lens)
            targets = targets.reshape(
                -1, self.hparams.MaskNet.num_spks, targets.shape[-1]
            )
            targets = targets.permute(0, 2, 1)

            inputs = targets.sum(-1)

        if self.hparams.use_waveformdrop:
            wav_lens = torch.tensor(
                [inputs.shape[-1]] * inputs.shape[0], device=self.device
            )
            inputs = self.hparams.waveformdrop_block(inputs, wav_lens)

        # Note: Other LR schedulers such as NoamScheduler can be used starting from this line

        if self.hparams.auto_mix_prec:
            with autocast():
                predictions = self.compute_forward(inputs)
                loss = self.compute_objectives(predictions, targets)

            if (
                loss < self.hparams.loss_upper_lim
            ):  # fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.inifinite_loss_found += 1
                logger.info(
                    "infinite loss! it happened {} times so far - skipping this batch".format(
                        self.inifinite_loss_found
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        else:
            predictions = self.compute_forward(inputs)
            loss = self.compute_objectives(predictions, targets)
            loss.backward()
            if self.hparams.clip_grad_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(), self.hparams.clip_grad_norm
                )
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage="test"):
        inputs = batch[0][1].to(self.device)
        targets = torch.cat(
            [
                batch[i][1].unsqueeze(-1)
                for i in range(1, self.hparams.MaskNet.num_spks + 1)
            ],
            dim=-1,
        ).to(self.device)

        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, targets)
        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch):

        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

            logger.info("Completed epoch %d" % epoch)
            logger.info("Train SI-SNR: %.3f" % -stage_loss)

        if stage == sb.Stage.VALID:

            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )

                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                next_lr = (
                    current_lr
                ) = self.hparams.optimizer.optim.param_groups[0]["lr"]

            logger.info("Valid SI-SNR: %.3f" % -stage_loss)
            logger.info(
                "Current LR {} New LR on next epoch {}".format(
                    current_lr, next_lr
                )
            )

            if self.root_process:
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch, "lr": current_lr},
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )
                self.checkpointer.save_and_keep_only(
                    meta={"SI-SNR": stage_stats["loss"]}, min_keys=["SI-SNR"],
                )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def compute_forward(self, mixture, stage="train"):
        """

        :param mixture: raw audio - dimension [batch_size, time]
        :param stage:
        :param init_params:
        :return:
        """

        mixture_w = self.hparams.Encoder(mixture)
        # [batch, channel, time / kernel stride]
        est_mask = self.hparams.MaskNet(mixture_w)

        out = [
            est_mask[i] * mixture_w
            for i in range(self.hparams.MaskNet.num_spks)
        ]
        est_source = torch.cat(
            [
                self.hparams.Decoder(out[i]).unsqueeze(-1)
                for i in range(self.hparams.MaskNet.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        # [B, T, Number of speaker=2]
        return est_source


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    hparams_file, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        params = sb.yaml.load_extended_yaml(fin, overrides)

    if params["minimal"]:
        repo_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../"
        params = create_minimal_data(repo_path, hparams_file)
        logger.info("setting epoch size to 1 - because --minimal")
        params["N_epochs"] = 1
    else:

        # override the data_path if we want to --
        # I want to do it speechbrain way not, but I am not sure how to do it
        # if args.data_path is not None:
        #     params.wsj0mixpath = args.data_path

        # this points to the folder to which we will save the wsj0-mix dataset
        data_save_dir = params["wsj0mixpath"]

        # if the dataset is not present, we create the dataset
        if not os.path.exists(data_save_dir):
            from recipes.WSJ2Mix.prepare_data import get_wsj_files

            raise NotImplementedError(
                "We have implemented a function for wsj0mix, \
                  but it is not confirmed to give the same dataset as the original one "
            )
            # this points to the folder which holds the wsj0 dataset folder
            wsj0path = params.wsj0path
            get_wsj_files(wsj0path, data_save_dir)

        if params["MaskNet"].num_spks == 2:
            from recipes.WSJ2Mix.prepare_data import create_wsj_csv

            create_wsj_csv(data_save_dir, params["save_folder"])
        elif params["MaskNet"].num_spks == 3:
            from recipes.WSJ2Mix.prepare_data import create_wsj_csv_3spks

            create_wsj_csv_3spks(data_save_dir, params["save_folder"])
        else:
            raise ValueError("We do not support this many speakers")

        tr_csv = os.path.realpath(
            os.path.join(params["save_folder"] + "/wsj_tr.csv")
        )
        cv_csv = os.path.realpath(
            os.path.join(params["save_folder"] + "/wsj_cv.csv")
        )
        tt_csv = os.path.realpath(
            os.path.join(params["save_folder"] + "/wsj_tt.csv")
        )

        with open(hparams_file) as fin:
            params = sb.yaml.load_extended_yaml(
                fin, {"tr_csv": tr_csv, "cv_csv": cv_csv, "tt_csv": tt_csv}
            )

        # copy the config file for book keeping
        shutil.copyfile(hparams_file, params["output_folder"] + "/config.txt")

    logger.info(pprint.PrettyPrinter(indent=4).pformat(params))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "will run on device {} / using mixed precision? {}".format(
            device, params["auto_mix_prec"]
        )
    )

    train_loader = params["train_loader"]()
    val_loader = params["val_loader"]()
    test_loader = params["test_loader"]()

    ssb = SourceSeparationBrain(
        modules={
            "encoder": params["Encoder"],
            "masknet": params["MaskNet"],
            "decoder": params["Decoder"],
        },
        opt_class=params["optimizer"],
        hparams=params,
        checkpointer=params["checkpointer"],
    )

    # re-initialize the parameters
    for module in ssb.modules.values():
        reset_layer_recursively(module)

    if params["test_only"]:
        # save_audio_results(params, ctn, test_loader, device, N=10)

        # get the score on the whole test set
        test_stats = ssb.evaluate(
            test_loader, min_key="SI-SNR", progressbar=True
        )
        save_audio_results(
            params, ssb, test_loader, device=device, justNitems=True
        )
    else:

        ssb.fit(
            epoch_counter=params["epoch_counter"],
            train_set=train_loader,
            valid_set=val_loader,
            progressbar=params["progressbar"],
            # early_stopping_with_patience=params["early_stopping_with_patience"],
        )
