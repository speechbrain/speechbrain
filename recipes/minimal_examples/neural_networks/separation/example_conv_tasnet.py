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


def create_minimal_data(repository_folder, config_file_path):
    tr_csv = os.path.realpath(
        os.path.join(
            repository_folder,
            "samples/audio_samples/sourcesep_samples/minimal_example_convtasnet_tr.csv",
        )
    )
    cv_csv = os.path.realpath(
        os.path.join(
            repository_folder,
            "samples/audio_samples/sourcesep_samples/minimal_example_convtasnet_cv.csv",
        )
    )
    tt_csv = os.path.realpath(
        os.path.join(
            repository_folder,
            "samples/audio_samples/sourcesep_samples/minimal_example_convtasnet_tt.csv",
        )
    )

    data_folder = "samples/audio_samples/sourcesep_samples"
    data_folder = os.path.realpath(os.path.join(repository_folder, data_folder))

    with open(config_file_path) as fin:
        params = sb.yaml.load_extended_yaml(
            fin,
            {
                "data_folder": data_folder,
                "tr_csv": tr_csv,
                "cv_csv": cv_csv,
                "tt_csv": tt_csv,
            },
        )
    return params


class CTN_Brain(sb.Brain):
    def compute_forward(self, mixture, stage):

        mix_w = self.hparams.encoder(mixture)
        est_mask = self.hparams.mask_net(mix_w)
        mix_w = torch.stack([mix_w] * 2)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [self.hparams.decoder(sep_h[i]).unsqueeze(-1) for i in range(2)],
            dim=-1,
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
        if self.hparams.loss_fn == "sisnr":
            loss = get_si_snr_with_pitwrapper(targets, predictions)
            return loss
        else:
            raise ValueError("Not Correct Loss Function Type")

    def fit_batch(self, batch):
        # train_onthefly option enables data augmentation,
        # by creating random mixtures within the batch
        if self.hparams.train_onthefly:
            bs = batch.sig.data.shape[0]
            perm = torch.randperm(bs)

            T = 24000
            Tmax = max((batch.mix_sig.data.shape[-1] - T) // 10, 1)
            Ts = torch.randint(0, Tmax, (1,))
            source1 = batch.source1.data[perm, Ts : Ts + T].to(self.device)
            source2 = batch.source2.data[:, Ts : Ts + T].to(self.device)

            ws = torch.ones(2).to(self.device)
            ws = ws / ws.sum()

            inputs = ws[0] * source1 + ws[1] * source2
            targets = torch.cat(
                [source1.unsqueeze(1), source2.unsqueeze(1)], dim=1
            )
        else:
            inputs = batch.mix_sig.data.to(self.device)
            targets = torch.cat(
                [
                    batch.source1.data.unsqueeze(-1),
                    batch.source2.data.unsqueeze(-1),
                ],
                dim=-1,
            ).to(self.device)

        predictions = self.compute_forward(inputs, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, targets)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        inputs = batch.mix_sig.data.to(self.device)
        targets = torch.cat(
            [
                batch.source1.data.unsqueeze(-1),
                batch.source2.data.unsqueeze(-1),
            ],
            dim=-1,
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

    ctn = CTN_Brain(hparams["modules"], hparams["opt_class"], hparams)

    print(hparams["train_data"].data.items())

    ctn.fit(
        ctn.hparams.epoch_counter,
        train_set=hparams["train_data"],
        valid_set=hparams["valid_data"],
        progressbar=hparams["progressbar"],
    )

    ctn.evaluate(hparams["valid_data"])

    # Integration test: check that the model overfits the training data
    assert -ctn.train_loss > 5.0


if __name__ == "__main__":
    main()


def test_error():
    main()
