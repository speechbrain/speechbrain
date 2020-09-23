#!/usr/bin/python

"""
Recipe to train CONV-TASNET model on the WSJ0 dataset

Author:
    * Cem Subakan 2020
"""
import argparse
import logging
import os
import speechbrain as sb
from recipes.minimal_examples.neural_networks.separation.example_conv_tasnet import (
    create_minimal_data,
)
from speechbrain.utils.train_logger import summarize_average, TensorboardLogger
import torch
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper

import torch.nn.functional as F
import itertools as it


logger = logging.getLogger(__name__)


def split_overlapping_chunks(tensor, chunk_len=200, overlap_rate=0.5, dim=1):

    chunks = []
    for i in it.islice(
        range(tensor.shape[1]),
        0,
        tensor.shape[1],
        int(chunk_len * overlap_rate),
    ):
        chunk = tensor[:, i : i + chunk_len, :]

        orig_len = chunk.shape[1]
        if orig_len < chunk_len:
            pad = (0, 0, 0, chunk_len - orig_len, 0, 0)
            chunk = F.pad(chunk, pad, "constant", 0)
            assert (
                chunk[:, orig_len:, :].sum() == 0
            ), "zero padding is not proper"
        assert chunk.shape[1] == chunk_len, "a chunk does not have bptt length"
        chunks.append(chunk)

    return chunks


class CTNBrain(sb.core.Brain):
    def __init__(self, params, device, **kwargs):
        self.param = params
        self.device = device
        super(CTNBrain, self).__init__(**kwargs)

    def compute_forward(self, mixture, stage="train", init_params=False):
        if hasattr(self.param, "env_corrupt"):
            if stage == "train":
                wav_lens = torch.tensor(
                    [mixture.shape[-1]] * mixture.shape[0]
                ).to(self.device)
                mixture = self.param.augmentation(
                    mixture, wav_lens, init_params
                )

        mixture_w = self.param.Encoder(mixture, init_params=init_params)

        est_mask = self.param.MaskNet(mixture_w, init_params=init_params)

        out = [est_mask[i] * mixture_w for i in range(2)]
        est_source = torch.cat(
            [self.param.Decoder(out[i]).unsqueeze(-1) for i in range(2)], dim=-1
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(1)
        T_conv = est_source.size(1)
        if T_origin > T_conv:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_conv))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source

    def compute_objectives(self, predictions, targets):
        if self.param.loss_fn == "sisnr":
            loss = get_si_snr_with_pitwrapper(targets, predictions)
            return loss
        else:
            raise ValueError("Not Correct Loss Function Type")

    def fit_batch(self, batch):
        # train_onthefly option enables data augmentation, by creating random mixtures within the batch
        if self.param.train_onthefly:
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

        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, targets)

        loss.backward()
        if self.param.clip_grad_norm >= 0:
            torch.nn.utils.clip_grad_norm_(
                self.modules.parameters(), self.param.clip_grad_norm
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

        av_loss = summarize_average(valid_stats["loss"])
        current_lr, next_lr = self.param.lr_scheduler(
            [self.param.optimizer], epoch, av_loss
        )
        # if params.use_tensorboard:
        train_logger = TensorboardLogger(self.param.tensorboard_logs)
        train_logger.log_stats({"Epoch": epoch}, train_stats, valid_stats)
        logger.info("Completed epoch %d" % epoch)
        logger.info(
            "Train SI-SNR: %.3f" % -summarize_average(train_stats["loss"])
        )
        logger.info(
            "Valid SI-SNR: %.3f" % -summarize_average(valid_stats["loss"])
        )
        logger.info(
            "Current LR {} New LR on next epoch {}".format(current_lr, next_lr)
        )

        self.param.checkpointer.save_and_keep_only(
            meta={"av_loss": av_loss},
            importance_keys=[ckpt_recency, lambda c: -c.meta["av_loss"]],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file", required=True)
    parser.add_argument(
        "--minimal",
        help="will run a minimal example for debugging",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    experiment_dir = os.path.dirname(os.path.realpath(__file__))

    # os.path.dirname(os.path.realpath(__file__)) + "/../../../s.path.dirname(os.path.realpath(__file__)) + "/../../../
    if args.minimal:
        params = create_minimal_data(
            os.path.dirname(os.path.realpath(__file__)) + "/../../../",
            args.config,
        )
    else:
        params_file = os.path.join(experiment_dir, args.config)
        with open(params_file) as fin:
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
                os.path.join(experiment_dir, params.save_folder + "/wsj_tr.csv")
            )
            cv_csv = os.path.realpath(
                os.path.join(experiment_dir, params.save_folder + "/wsj_cv.csv")
            )
            tt_csv = os.path.realpath(
                os.path.join(experiment_dir, params.save_folder + "/wsj_tt.csv")
            )

            with open(params_file) as fin:
                params = sb.yaml.load_extended_yaml(
                    fin, {"tr_csv": tr_csv, "cv_csv": cv_csv, "tt_csv": tt_csv}
                )
            # logger.info(params)  # if needed this line can be uncommented for logging

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = params.train_loader()
    val_loader = params.val_loader()
    test_loader = params.test_loader()

    ctn = CTNBrain(
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

    ctn.fit(
        range(params.N_epochs),
        train_set=train_loader,
        valid_set=val_loader,
        progressbar=params.progressbar,
    )

    test_stats = ctn.evaluate(test_loader)
    logger.info("Test SI-SNR: %.3f" % -summarize_average(test_stats["loss"]))


if __name__ == "__main__":
    main()
