#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average
import torch
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper

import torch.nn.functional as F
import csv

experiment_dir = os.path.dirname(os.path.realpath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")

with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin)

# this points to the folder which holds the wsj0-mix dataset folder
datapath = params.datapath


# load or create the csv files for the data
if not (
    os.path.exists("wsj_tr.csv")
    and os.path.exists("wsj_cv.csv")
    and os.path.exists("wsj_tt.csv")
):
    for set_type in ["tr", "cv", "tt"]:
        mix_path = (
            datapath + "wsj0-mix/2speakers/wav8k/min/" + set_type + "/mix/"
        )
        s1_path = datapath + "wsj0-mix/2speakers/wav8k/min/" + set_type + "/s1/"
        s2_path = datapath + "wsj0-mix/2speakers/wav8k/min/" + set_type + "/s2/"

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
        ]

        with open("wsj_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                }
                writer.writerow(row)


tr_csv = os.path.realpath(os.path.join(experiment_dir, "wsj_tr.csv"))
cv_csv = os.path.realpath(os.path.join(experiment_dir, "wsj_cv.csv"))
tt_csv = os.path.realpath(os.path.join(experiment_dir, "wsj_tt.csv"))

with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(
        fin, {"tr_csv": tr_csv, "cv_csv": cv_csv, "tt_csv": tt_csv}
    )
# print(params)  # if needed this line can be uncommented for logging


if params.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    train_logger = TensorboardLogger(params.tensorboard_logs)


device = "cuda" if torch.cuda.is_available() else "cpu"


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

# params.checkpointer.recover_if_possible(lambda c: -c.meta["av_loss"])

# with torch.autograd.detect_anomaly():
ctn.fit(
    range(params.N_epochs),
    train_set=train_loader,
    valid_set=val_loader,
    progressbar=params.progressbar,
)

test_stats = ctn.evaluate(test_loader)
print("Test loss: %.3f" % summarize_average(test_stats["loss"]))
