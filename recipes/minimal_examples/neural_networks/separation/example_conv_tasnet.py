#!/usr/bin/python
"""
A minimal example on the conv-tasnet model

Author
    * Cem Subakan 2020
"""

import os
import torch
import speechbrain as sb
import torch.nn.functional as F
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper


class CTN_Brain(sb.Brain):
    def compute_forward(self, mixture, stage):

        mixture_w = self.modules.encoder(mixture)
        est_mask = self.modules.mask_net(mixture_w)
        est_source = self.modules.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(1)
        T_conv = est_source.size(1)
        if T_origin > T_conv:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_conv))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source

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
                [batch[1][1].unsqueeze(-1), batch[2][1].unsqueeze(-1)], dim=-1
            ).to(self.device)

        predictions = self.compute_forward(inputs, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, targets)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        inputs = batch[0][1].to(self.device)
        targets = torch.cat(
            [batch[1][1].unsqueeze(-1), batch[2][1].unsqueeze(-1)], dim=-1
        ).to(self.device)

        predictions = self.compute_forward(inputs, stage)
        loss = self.compute_objectives(predictions, targets)
        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        elif stage == sb.Stage.VALID:
            if self.hparams.use_tensorboard:
                self.hparams.train_logger.log_stats(
                    {"Epoch": epoch},
                    {"loss": -self.train_loss},
                    {"loss": -stage_loss},
                )
            print("Completed epoch %d" % epoch)
            print("Train SI-SNR: %.3f" % -self.train_loss)
            print("Valid SI-SNR: %.3f" % -stage_loss)
        elif stage == sb.Stage.TEST:
            print("Test SI-SNR: %.3f" % -stage_loss)


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    params_file = os.path.join(experiment_dir, "hyperparams.yaml")
    sourcesep_samples_dir = os.path.realpath(
        os.path.join(
            experiment_dir,
            "..",
            "..",
            "..",
            "..",
            "samples",
            "audio_samples",
            "sourcesep_samples",
        )
    )

    with open(params_file) as fin:
        hparams = sb.yaml.load_extended_yaml(
            fin, {"data_folder": sourcesep_samples_dir},
        )

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["train_logger"] = TensorboardLogger(hparams["tensorboard_logs"])

    train_loader = hparams["train_loader"]()
    val_loader = hparams["val_loader"]()
    test_loader = hparams["test_loader"]()

    ctn = CTN_Brain(hparams["modules"], hparams["opt_class"], hparams)

    ctn.fit(
        ctn.hparams.epoch_counter,
        train_set=train_loader,
        valid_set=val_loader,
        progressbar=hparams["progressbar"],
    )

    ctn.evaluate(test_loader)

    # Integration test: check that the model overfits the training data
    assert -ctn.train_loss > 10.0


if __name__ == "__main__":
    main()


def test_error():
    main()
