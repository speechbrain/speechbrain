#!/usr/bin/python

"""
Recipe to train DP-RNN model on the WSJ0-2Mix dataset

Author:
    * Cem Subakan 2020
    * Mirko Bronzi 2020
"""

import argparse
import logging
import os
import pprint
import shutil

# from pathlib import PosixPath

# import itertools as it

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast  # , GradScaler

import speechbrain as sb
from recipes.minimal_examples.neural_networks.separation.example_conv_tasnet import (
    create_minimal_data,
)
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper

# from speechbrain.utils.checkpoints import ckpt_recency

# from speechbrain.utils.train_logger import summarize_average

# from speechbrain.data_io.data_io import write_wav_soundfile

import speechbrain.nnet.schedulers as schedulers

# from tqdm import tqdm
import numpy as np


logger = logging.getLogger(__name__)


def reset_layer_recursively(layer):
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()
    for child_layer in layer.modules():
        if layer != child_layer:
            reset_layer_recursively(child_layer)


class SourceSeparationBrain(sb.core.Brain):
    def compute_objectives(self, predictions, targets):
        if self.hparams.loss_fn == "sisnr":
            loss = get_si_snr_with_pitwrapper(targets, predictions)
            return loss
        else:
            raise ValueError("Not Correct Loss Function Type")

    def fit_batch(self, batch):
        # train_onthefly option enables data augmentation,
        # by creating random mixtures within the batch
        if self.hparams.train_onthefly:
            bs = batch[0][1].shape[0]
            perm = torch.randperm(bs)

            T = 24000
            Tmax = max((batch[0][1].shape[-1] - T) // 10, 1)
            Ts = torch.randint(0, Tmax, (1,))
            source1 = batch[1][1][perm, Ts : Ts + T].to(self.device)
            source2 = batch[2][1][:, Ts : Ts + T].to(self.device)

            ws = torch.ones(2).to(self.device)
            ws = ws / ws.sum()

            inputs = ws[0] * source1 + ws[1] * source2
            targets = torch.cat(
                [source1.unsqueeze(1), source2.unsqueeze(1)], dim=1
            )
        else:
            inputs = batch[0][1].to(self.device)
            targets = torch.cat(
                [
                    batch[i][1].unsqueeze(-1)
                    for i in range(1, self.hparams.MaskNet.num_spks + 1)
                ],
                dim=-1,
            ).to(self.device)

        if isinstance(
            self.hparams.MaskNet.dual_mdl[0].intra_mdl,
            sb.lobes.models.dual_path.DPTNetBlock,
        ) or isinstance(
            self.hparams.MaskNet.dual_mdl[0].intra_mdl,
            sb.lobes.models.dual_path.PTRNNBlock,
        ):
            randstart = np.random.randint(
                0, 1 + max(0, inputs.shape[1] - 32000)
            )
            targets = targets[:, randstart : randstart + 32000, :]

        if self.hparams.use_data_augmentation:
            targets = targets.permute(0, 2, 1)
            targets = targets.reshape(-1, targets.shape[-1])
            wav_lens = torch.tensor([targets.shape[-1]] * targets.shape[0]).to(
                self.device
            )

            targets = self.hparams.augmentation(targets, wav_lens)
            targets = targets.reshape(
                -1, self.hparams.MaskNet.num_spks, targets.shape[-1]
            )
            targets = targets.permute(0, 2, 1)

            if hasattr(self.hparams, "use_data_shuffling"):
                # only would work for 2 spks
                perm = torch.randperm(targets.size(0))
                targets = torch.stack(
                    [targets[perm, :, 0], targets[:, :, 1]], dim=2
                )

            inputs = targets.sum(-1)

        # TODO: consider if we need this part..
        # if isinstance(self.params.lr_scheduler, schedulers.NoamScheduler):
        #     old_lr, new_lr = self.params.lr_scheduler(
        #         [self.optimizer], None, None
        #     )
        #     print("oldlr ", old_lr, "newlr", new_lr)
        #     print(self.optimizer.optim.param_groups[0]["lr"])

        if self.hparams.auto_mix_prec:
            with autocast():
                predictions = self.compute_forward(inputs)
                loss = self.compute_objectives(predictions, targets)

            if loss < 999999:  # fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm
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

        predictions = self.compute_forward(inputs, stage="test")
        loss = self.compute_objectives(predictions, targets)
        return loss.detach()

    # Todo: I need to convert this on_stage_end
    # def on_epoch_end(self, epoch, train_stats, valid_stats):
    def on_stage_end(self, stage, stage_loss, epoch):

        # I need to figure out what function to use for summary averages
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

            logger.info("Completed epoch %d" % epoch)
            logger.info("Train SI-SNR: %.3f" % -stage_loss)

        if stage == sb.Stage.VALID:

            # av_valid_loss = summarize_average(valid_stats["loss"])
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

    def compute_forward(self, mixture, stage="train", init_params=False):
        """

        :param mixture: raw audio - dimension [batch_size, time]
        :param stage:
        :param init_params:
        :return:
        """

        mixture_w = self.hparams.Encoder(mixture, init_params=init_params)
        # [batch, channel, time / kernel stride]
        est_mask = self.hparams.MaskNet(mixture_w, init_params=init_params)

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


# TODO: complete the main / check the one above from the updated code.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file", required=True)
    parser.add_argument(
        "--data_path", help="the data path to load the dataset", required=False
    )
    parser.add_argument(
        "--minimal",
        help="will run a minimal example for debugging",
        action="store_true",
    )
    parser.add_argument(
        "--test_only",
        help="will only run testing, and not training",
        action="store_true",
    )
    parser.add_argument(
        "--use_multigpu",
        help="will use multigpu in training",
        action="store_true",
    )
    parser.add_argument(
        "--num_spks", help="number of speakers", type=int, default=2,
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.minimal:
        repo_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../"
        params = create_minimal_data(repo_path, args.config)
        logger.info("setting epoch size to 1 - because --minimal")
        params["N_epochs"] = 1
        # params = fix_params_for_orion(params)
    else:
        with open(args.config) as fin:
            params = sb.yaml.load_extended_yaml(fin)

        # override the data_path if we want to
        if args.data_path is not None:
            params.wsj0mixpath = args.data_path

        # this points to the folder to which we will save the wsj0-mix dataset
        data_save_dir = params.wsj0mixpath

        # if the dataset is not present, we create the dataset
        if not os.path.exists(data_save_dir):
            from recipes.WSJ2Mix.prepare_data import get_wsj_files

            # this points to the folder which holds the wsj0 dataset folder
            wsj0path = params.wsj0path
            get_wsj_files(wsj0path, data_save_dir)

        # load or create the csv files which enables us to get the speechbrain dataloaders
        # if not (
        #    os.path.exists(params.save_folder + "/wsj_tr.csv")
        #    and os.path.exists(params.save_folder + "/wsj_cv.csv")
        #    and os.path.exists(params.save_folder + "/wsj_tt.csv")
        # ):
        # we always recreate the csv files too keep track of the latest path

        if params.MaskNet.num_spks == 2:
            from recipes.WSJ2Mix.prepare_data import create_wsj_csv

            create_wsj_csv(data_save_dir, params.save_folder)
        elif params.MaskNet.num_spks == 3:
            from recipes.WSJ2Mix.prepare_data import create_wsj_csv_3spks

            create_wsj_csv_3spks(data_save_dir, params.save_folder)
        else:
            raise ValueError("We do not support this many speakers")

        tr_csv = os.path.realpath(
            os.path.join(params.save_folder + "/wsj_tr.csv")
        )
        cv_csv = os.path.realpath(
            os.path.join(params.save_folder + "/wsj_cv.csv")
        )
        tt_csv = os.path.realpath(
            os.path.join(params.save_folder + "/wsj_tt.csv")
        )

        with open(args.config) as fin:
            params = sb.yaml.load_extended_yaml(
                fin, {"tr_csv": tr_csv, "cv_csv": cv_csv, "tt_csv": tt_csv}
            )
        # params = fix_params_for_orion(params)
        # copy the config file for book keeping
        shutil.copyfile(args.config, params.output_folder + "/config.txt")

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
            "encoder": params["Encoder"],  # .to(device),
            "masknet": params["MaskNet"],  # .to(device),
            "decoder": params["Decoder"],  # .to(device),
        },
        opt_class=params["optimizer"],
        # first_inputs=[next(iter(train_loader))[0][1]],
        hparams=params,
        checkpointer=params["checkpointer"],
    )
    ssb.args = args

    for module in ssb.modules.values():
        reset_layer_recursively(module)

    if args.test_only:
        # save_audio_results(params, ctn, test_loader, device, N=10)

        # get the score on the whole test set
        test_stats = ssb.evaluate(test_loader)
        # logger.info(
        #    "Test SI-SNR: %.3f" % -summarize_average(test_stats["loss"])
        # )
    else:

        # mlflow.start_run()
        ssb.fit(
            epoch_counter=params["epoch_counter"],
            train_set=train_loader,
            valid_set=val_loader,
            progressbar=params["progressbar"],
            # early_stopping_with_patience=params["early_stopping_with_patience"],
        )

        # test_stats = ctn.evaluate(test_loader)
        # logger.info(
        #    "Test SI-SNR: %.3f" % -summarize_average(test_stats["loss"])
        # )

        # best_eval = min(ctn.eval_scores)
        # logger.info("Best result on validation: {}".format(-best_eval))

        # report_results(
        #    [dict(name="dev_metric", type="objective", value=float(best_eval),)]
        # )
