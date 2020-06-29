#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average
import torch

experiment_dir = os.path.dirname(os.path.realpath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")
path1 = "../../../../samples/audio_samples/WSJ_mini_partitioned"
data_folder1 = os.path.realpath(os.path.join(experiment_dir, path1))
path2 = "../../../../samples/audio_samples/ESC50_mini_partitioned"
data_folder2 = os.path.realpath(os.path.join(experiment_dir, path2))

with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(
        fin, {"data_folder1": data_folder1, "data_folder2": data_folder2}
    )

if params.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    train_logger = TensorboardLogger(params.tensorboard_logs)


device = "cuda" if torch.cuda.is_available() else "cpu"


class CTN_Brain(sb.core.Brain):
    def compute_forward(self, mixture, init_params=False):
        est_sources = params.conv_tasnet(mixture)
        return est_sources

    def compute_objectives(self, predictions, targets):
        return params.sisdr_loss(predictions, targets)

    def fit_batch(self, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch, stage="test"):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, targets)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        if params.use_tensorboard:
            train_logger.log_stats({"Epoch": epoch}, train_stats, valid_stats)
        print("Completed epoch %d" % epoch)
        print("Train loss: %.3f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.3f" % summarize_average(valid_stats["loss"]))


train_dataset = params.tr_dataset
train_loader = train_dataset.get_dataloader()

val_dataset = params.val_dataset
val_loader = val_dataset.get_dataloader()

ctn = CTN_Brain(
    modules=[params.conv_tasnet.to(device)],
    optimizer=params.optimizer,
    first_inputs=[next(iter(train_loader))[0].to(device)],
)

ctn.fit(range(params.N_epochs), train_set=train_loader, valid_set=val_loader)
