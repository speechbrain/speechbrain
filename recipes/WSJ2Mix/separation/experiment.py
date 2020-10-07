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
from pathlib import PosixPath
import itertools as it

import mlflow
import orion
import torch
import torch.nn.functional as F
from mlflow import log_metric
from orion.client import report_results
from torch.cuda.amp import GradScaler, autocast

import speechbrain as sb
from recipes.minimal_examples.neural_networks.separation.example_conv_tasnet import (
    create_minimal_data,
)
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_average
from speechbrain.data_io.data_io import write_wav_soundfile
import speechbrain.nnet.lr_schedulers as schedulers

logger = logging.getLogger(__name__)


def reset_layer_recursively(layer):
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()
    for child_layer in layer.modules():
        if layer != child_layer:
            reset_layer_recursively(child_layer)


def save_audio_results(params, model, test_loader, device, N=10):
    save_path = os.path.join(params.output_folder, "audio_results")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fs = 8000

    for i, batch in enumerate(it.islice(test_loader, 0, N, 1)):
        inputs = batch[0][1].to(device)
        predictions = model.compute_forward(inputs, stage="test").detach()
        write_wav_soundfile(
            predictions[0, :, 0] / predictions[0, :, 0].std(),
            save_path + "/item{}_source{}hat.wav".format(i, 1),
            fs,
        )
        write_wav_soundfile(
            predictions[0, :, 1] / predictions[0, :, 1].std(),
            save_path + "/item{}_source{}hat.wav".format(i, 2),
            fs,
        )
        write_wav_soundfile(
            batch[1][1], save_path + "/item{}_source{}.wav".format(i, 1), fs
        )
        write_wav_soundfile(
            batch[2][1], save_path + "/item{}_source{}.wav".format(i, 2), fs
        )
        write_wav_soundfile(
            inputs[0], save_path + "/item{}_mixture.wav".format(i), fs
        )


class SourceSeparationBrainSuperclass(sb.core.Brain):
    def __init__(self, params, device, **kwargs):
        self.params = params
        self.device = device
        super(SourceSeparationBrainSuperclass, self).__init__(**kwargs)
        self.eval_scores = []
        self.scaler = GradScaler()

    def compute_forward(self, mixture, stage="train", init_params=False):
        raise NotImplementedError("use a subclass")

    def compute_objectives(self, predictions, targets):
        if self.params.loss_fn == "sisnr":
            loss = get_si_snr_with_pitwrapper(targets, predictions)
            return loss
        else:
            raise ValueError("Not Correct Loss Function Type")

    def fit_batch(self, batch):
        # train_onthefly option enables data augmentation,
        # by creating random mixtures within the batch
        if self.params.train_onthefly:
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
                [batch[1][1].unsqueeze(-1), batch[2][1].unsqueeze(-1)], dim=-1
            ).to(self.device)

        if self.params.use_data_augmentation:
            targets = targets.permute(0, 2, 1)
            targets = targets.reshape(-1, targets.shape[-1])
            wav_lens = torch.tensor([targets.shape[-1]] * targets.shape[0]).to(
                self.device
            )

            targets = self.params.augmentation(targets, wav_lens)
            targets = targets.reshape(-1, 2, targets.shape[-1])
            targets = targets.permute(0, 2, 1)
            inputs = targets.sum(-1)

        if isinstance(self.params.lr_scheduler, schedulers.NoamScheduler):
            old_lr, new_lr = self.params.lr_scheduler(
                [self.optimizer], None, None
            )
            print("oldlr ", old_lr, "newlr", new_lr)
            print(self.optimizer.optim.param_groups[0]["lr"])

        if self.params.mixed_precision:
            with autocast():
                predictions = self.compute_forward(inputs)
                loss = self.compute_objectives(predictions, targets)
            self.scaler.scale(loss).backward()

            if self.params.clip_grad_norm >= 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(), self.params.clip_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predictions = self.compute_forward(inputs)
            loss = self.compute_objectives(predictions, targets)
            loss.backward()
            if self.params.clip_grad_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(), self.params.clip_grad_norm
                )
            self.optimizer.step()
        self.optimizer.zero_grad()

        return {"loss": loss.detach()}

    def evaluate_batch(self, batch, stage="test"):
        inputs = batch[0][1].to(self.device)
        targets = torch.cat(
            [batch[1][1].unsqueeze(-1), batch[2][1].unsqueeze(-1)], dim=-1
        ).to(self.device)

        predictions = self.compute_forward(inputs, stage="test")
        loss = self.compute_objectives(predictions, targets)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):

        av_valid_loss = summarize_average(valid_stats["loss"])
        if isinstance(self.params.lr_scheduler, schedulers.ReduceLROnPlateau):
            current_lr, next_lr = self.params.lr_scheduler(
                [self.params.optimizer], epoch, av_valid_loss
            )
        else:
            # if we do not use the reducelronplateau, we do not change the lr
            next_lr = current_lr = self.params.optimizer.optim.param_groups[0][
                "lr"
            ]

        epoch_stats = {"epoch": epoch, "lr": current_lr}
        self.params.train_logger.log_stats(
            epoch_stats, train_stats, valid_stats
        )

        av_train_loss = summarize_average(train_stats["loss"])
        log_metric("train_loss", av_train_loss, step=epoch)
        log_metric("valid_loss", av_valid_loss, step=epoch)
        log_metric("current_lr", current_lr, step=epoch)

        logger.info("Completed epoch %d" % epoch)
        logger.info(
            "Train SI-SNR: %.3f" % -summarize_average(train_stats["loss"])
        )
        eval_score = summarize_average(valid_stats["loss"])
        self.eval_scores.append(eval_score)
        logger.info("Valid SI-SNR: %.3f" % -eval_score)
        logger.info(
            "Current LR {} New LR on next epoch {}".format(current_lr, next_lr)
        )

        self.params.checkpointer.save_and_keep_only(
            meta={"av_loss": av_valid_loss},
            importance_keys=[ckpt_recency, lambda c: -c.meta["av_loss"]],
        )


class SourceSeparationBrain(SourceSeparationBrainSuperclass):
    def compute_forward(self, mixture, stage="train", init_params=False):
        """

        :param mixture: raw audio - dimension [batch_size, time]
        :param stage:
        :param init_params:
        :return:
        """

        mixture_w = self.params.Encoder(mixture, init_params=init_params)
        # [batch, channel, time / kernel stride]
        est_mask = self.params.MaskNet(mixture_w, init_params=init_params)

        out = [est_mask[i] * mixture_w for i in range(2)]
        est_source = torch.cat(
            [self.params.Decoder(out[i]).unsqueeze(-1) for i in range(2)],
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


def fix_params_for_orion(params):
    if orion.client.cli.IS_ORION_ON:
        save_folder = os.getenv("ORION_WORKING_DIR")
        logger.info(
            "running orion - changing the output folder to {}".format(
                save_folder
            )
        )
        params.output_folder = save_folder
        params.save_folder = os.path.join(save_folder, "models")
        params.checkpointer.checkpoints_dir = PosixPath(save_folder)
        params.train_log = os.path.join(save_folder, "train_log.txt")
        params.train_logger.save_file = params.train_log
        params.tensorboard_logs += os.path.sep + os.getenv("ORION_TRIAL_ID")
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file", required=True)
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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.minimal:
        repo_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../"
        params = create_minimal_data(repo_path, args.config)
        logger.info("setting epoch size to 1 - because --minimal")
        params.N_epochs = 1
        params = fix_params_for_orion(params)
    else:
        with open(args.config) as fin:
            params = sb.yaml.load_extended_yaml(fin)

        # this points to the folder to which we will save the wsj0-mix dataset
        data_save_dir = params.wsj0mixpath

        # if the dataset is not present, we create the dataset
        if not os.path.exists(data_save_dir):
            from recipes.WSJ2Mix.prepare_data import get_wsj_files

            # this points to the folder which holds the wsj0 dataset folder
            wsj0path = params.wsj0path
            get_wsj_files(wsj0path, data_save_dir)

        # load or create the csv files which enables us to get the speechbrain dataloaders
        if not (
            os.path.exists(params.save_folder + "/wsj_tr.csv")
            and os.path.exists(params.save_folder + "/wsj_cv.csv")
            and os.path.exists(params.save_folder + "/wsj_tt.csv")
        ):
            from recipes.WSJ2Mix.prepare_data import create_wsj_csv

            create_wsj_csv(data_save_dir, params.save_folder)

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
        params = fix_params_for_orion(params)
        # copy the config file for book keeping
        shutil.copyfile(args.config, params.output_folder + "/config.txt")

    logger.info(pprint.PrettyPrinter(indent=4).pformat(params))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "will run on device {} / using mixed precision? {}".format(
            device, params.mixed_precision
        )
    )

    train_loader = params.train_loader()
    val_loader = params.val_loader()
    test_loader = params.test_loader()

    ctn = SourceSeparationBrain(
        modules=[
            params.Encoder.to(device),
            params.MaskNet.to(device),
            params.Decoder.to(device),
        ],
        optimizer=params.optimizer,
        first_inputs=[next(iter(train_loader))[0][1].to(device)],
        params=params,
        device=device,
    )

    # reinitialize the parameters
    for module in ctn.modules:
        reset_layer_recursively(module)

    params.checkpointer.recover_if_possible(lambda c: -c.meta["av_loss"])

    if args.test_only:
        save_audio_results(params, ctn, test_loader, device, N=10)

        # get the score on the whole test set
        test_stats = ctn.evaluate(test_loader)
        logger.info(
            "Test SI-SNR: %.3f" % -summarize_average(test_stats["loss"])
        )
    else:
        mlflow.start_run()
        ctn.fit(
            range(params.N_epochs),
            train_set=train_loader,
            valid_set=val_loader,
            progressbar=params.progressbar,
            early_stopping_with_patience=params.early_stopping_with_patience,
        )
        mlflow.end_run()

        test_stats = ctn.evaluate(test_loader)
        logger.info(
            "Test SI-SNR: %.3f" % -summarize_average(test_stats["loss"])
        )

        best_eval = min(ctn.eval_scores)
        logger.info("Best result on validation: {}".format(-best_eval))

        report_results(
            [dict(name="dev_metric", type="objective", value=float(best_eval),)]
        )


if __name__ == "__main__":
    main()
