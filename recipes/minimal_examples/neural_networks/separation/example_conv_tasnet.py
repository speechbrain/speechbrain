#!/usr/bin/python

"""
A minimal example on the conv-tasnet model

Author
    * Cem Subakan 2020
"""

import os
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average
import torch
from speechbrain.utils.checkpoints import ckpt_recency
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


class CTN_Brain(sb.core.Brain):
    def compute_forward(self, mixture, stage="train", init_params=False):

        if hasattr(params, "env_corrupt"):
            if stage == "train":
                wav_lens = torch.tensor(
                    [mixture.shape[-1]] * mixture.shape[0]
                ).to(device)
                mixture = params.augmentation(mixture, wav_lens, init_params)

        mixture_w = params.Encoder(mixture, init_params)
        est_mask = params.MaskNet(mixture_w, init_params)
        est_source = params.Decoder(mixture_w, est_mask, init_params)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(1)
        T_conv = est_source.size(1)
        if T_origin > T_conv:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_conv))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source

    def compute_objectives(self, predictions, targets):
        if params.loss_fn == "sisnr":
            loss = get_si_snr_with_pitwrapper(targets, predictions)
            return loss
        else:
            raise ValueError("Not Correct Loss Function Type")

    def fit_batch(self, batch):
        # train_onthefly option enables data augmentation, by creating random mixtures within the batch
        if params.train_onthefly:
            bs = batch[0][1].shape[0]
            perm = torch.randperm(bs)

            T = 24000
            Tmax = max((batch[0][1].shape[-1] - T) // 10, 1)
            Ts = torch.randint(0, Tmax, (1,))
            source1 = batch[1][1][perm, Ts : Ts + T].to(device)
            source2 = batch[2][1][:, Ts : Ts + T].to(device)

            ws = torch.ones(2).to(device)
            ws = ws / ws.sum()

            inputs = ws[0] * source1 + ws[1] * source2
            targets = torch.cat(
                [source1.unsqueeze(1), source2.unsqueeze(1)], dim=1
            )
        else:
            inputs = batch[0][1].to(device)
            targets = torch.cat(
                [batch[1][1].unsqueeze(-1), batch[2][1].unsqueeze(-1)], dim=-1
            ).to(device)

        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, targets)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch, stage="test"):
        inputs = batch[0][1].to(device)
        targets = torch.cat(
            [batch[1][1].unsqueeze(-1), batch[2][1].unsqueeze(-1)], dim=-1
        ).to(device)

        predictions = self.compute_forward(inputs, stage="test")
        loss = self.compute_objectives(predictions, targets)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):

        av_loss = summarize_average(valid_stats["loss"])
        if params.use_tensorboard:
            train_logger.log_stats({"Epoch": epoch}, train_stats, valid_stats)
        print("Completed epoch %d" % epoch)
        print("Train SI-SNR: %.3f" % -summarize_average(train_stats["loss"]))
        print("Valid SI-SNR: %.3f" % -summarize_average(valid_stats["loss"]))

        params.checkpointer.save_and_keep_only(
            meta={"av_loss": av_loss},
            importance_keys=[ckpt_recency, lambda c: -c.meta["av_loss"]],
        )


if __name__ == "__main__":

    repository_folder = (
        os.path.dirname(os.path.realpath(__file__)) + "/../../../../"
    )
    params = create_minimal_data(
        repository_folder,
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "params.yaml"
        ),
    )

    if params.use_tensorboard:
        from speechbrain.utils.train_logger import TensorboardLogger

        train_logger = TensorboardLogger(params.tensorboard_logs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = params.train_loader()
    val_loader = params.val_loader()
    test_loader = params.test_loader()

    ctn = CTN_Brain(
        modules=[
            params.Encoder.to(device),
            params.MaskNet.to(device),
            params.Decoder.to(device),
        ],
        optimizer=params.optimizer,
        first_inputs=[next(iter(train_loader))[0][1].to(device)],
    )

    ctn.fit(
        range(params.N_epochs),
        train_set=train_loader,
        valid_set=val_loader,
        progressbar=params.progressbar,
    )

    test_stats = ctn.evaluate(test_loader)
    print("Test SI-SNR: %.3f" % -summarize_average(test_stats["loss"]))


# Integration test: check that the model overfits the training data
def test_error():
    assert -ctn.avg_train_loss > 10.0
