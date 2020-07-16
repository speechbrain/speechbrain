#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average
import torch
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.lobes.models.conv_tasnet import cal_loss

experiment_dir = os.path.dirname(os.path.realpath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")

csv_dir = os.path.realpath(os.path.join(experiment_dir, "minimal_example.csv"))

data_folder = "../../../../samples/audio_samples/sourcesep_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))

with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(
        fin, {"data_folder": data_folder, "csv_dir": csv_dir}
    )
print(params)

if params.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    train_logger = TensorboardLogger(params.tensorboard_logs)

device = "cuda" if torch.cuda.is_available() else "cpu"


class CTN_Brain(sb.core.Brain):
    def compute_forward(self, mixture, init_params=False):
        est_sources = params.conv_tasnet(mixture)
        return est_sources

    def compute_objectives(self, predictions, targets):
        if params.loss_fn == "sisnr":
            lengths = torch.tensor(
                [predictions.shape[-1]] * predictions.shape[0]
            ).to(device)
            loss = cal_loss(targets, predictions, lengths)[0]
            return loss
        else:
            raise ValueError("Not Correct Loss Function Type")

    def fit_batch(self, batch):
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
                [batch[1][1].unsqueeze(1), batch[2][1].unsqueeze(1)], dim=1
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
            [batch[1][1].unsqueeze(1), batch[2][1].unsqueeze(1)], dim=1
        ).to(device)

        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, targets)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):

        av_loss = summarize_average(valid_stats["loss"])
        if params.use_tensorboard:
            train_logger.log_stats({"Epoch": epoch}, train_stats, valid_stats)
        print("Completed epoch %d" % epoch)
        print("Train loss: %.3f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.3f" % summarize_average(valid_stats["loss"]))

        params.checkpointer.save_and_keep_only(
            meta={"av_loss": av_loss},
            importance_keys=[ckpt_recency, lambda c: -c.meta["av_loss"]],
        )


train_loader = params.train_loader()
val_loader = params.val_loader()
test_loader = params.test_loader()

ctn = CTN_Brain(
    modules=[params.conv_tasnet.to(device)],
    optimizer=params.optimizer,
    first_inputs=[next(iter(test_loader))[0][1].to(device)],
)

params.checkpointer.recover_if_possible(lambda c: -c.meta["av_loss"])

# with torch.autograd.detect_anomaly():
ctn.fit(
    range(params.N_epochs),
    train_set=train_loader,
    valid_set=val_loader,
    progressbar=params.progressbar,
)

test_stats = ctn.evaluate(test_loader)
print("Test loss: %.3f" % summarize_average(test_stats["loss"]))
