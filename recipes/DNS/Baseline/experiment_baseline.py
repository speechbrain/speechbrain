#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
import torchaudio
from speechbrain.utils.train_logger import summarize_average
from speechbrain.processing.features import spectral_magnitude

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from dns_prepare import prepare_dns  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

if params.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    train_logger = TensorboardLogger(params.tensorboard_logs)

# Create the folder to save enhanced files
if not os.path.exists(params.enhanced_folder):
    os.mkdir(params.enhanced_folder)


class SEBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, lens = x
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        # print(torch.mean(wavs[0]))
        if stage == "train":
            wavs = params.add_noise(wavs, lens)

        feats = params.compute_stft(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        output = params.model(feats, init_params)
        # print(torch.mean(wavs[0]), torch.isnan(feats[0]))
        # self.write_wavs(torch.expm1(output), x, "01")

        return output

    def compute_objectives(self, predictions, targets, stage="train"):
        ids, wavs, lens = targets
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        feats = params.compute_stft(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        loss = params.compute_cost(predictions, feats, lens)

        return loss, {}

    def fit_batch(self, batch):
        inputs = batch[0]
        predictions = self.compute_forward(inputs)
        loss, stats = self.compute_objectives(predictions, inputs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()

        return stats

    def evaluate_batch(self, batch, stage="valid"):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, stage=stage)
        epoch = params.epoch_counter.current

        # Create the folder to save enhanced files
        if not os.path.exists(os.path.join(params.enhanced_folder, str(epoch))):
            os.mkdir(os.path.join(params.enhanced_folder, str(epoch)))

        # Write batch enhanced files to directory
        self.write_wavs(torch.expm1(predictions), targets, epoch)

        loss, stats = self.compute_objectives(predictions, targets, stage=stage)
        stats["loss"] = loss.detach()

        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        if params.use_tensorboard:
            train_logger.log_stats({"Epoch": epoch}, train_stats, valid_stats)

        params.checkpointer.save_checkpoint()

        print("Completed epoch %d" % epoch)
        print("Train loss: %.3f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.3f" % summarize_average(valid_stats["loss"]))

    def write_wavs(self, predictions, inputs, epoch):
        ids, wavs, lens = inputs
        predictions = predictions.cpu()

        feats = params.compute_stft(wavs)
        phase = torch.atan2(feats[:, :, :, 1], feats[:, :, :, 0])
        complex_predictions = torch.mul(
            torch.unsqueeze(predictions, -1),
            torch.cat(
                (
                    torch.unsqueeze(torch.cos(phase), -1),
                    torch.unsqueeze(torch.sin(phase), -1),
                ),
                -1,
            ),
        )
        pred_wavs = params.compute_istft(complex_predictions)

        for name, pred_wav in zip(ids, pred_wavs):
            enhance_path = os.path.join(
                params.enhanced_folder, str(epoch), name
            )
            torchaudio.save(enhance_path + ".wav", pred_wav, 16000)


prepare_dns(
    data_folder=params.data_folder,
    save_folder=params.data_folder,
    seg_size=10.0,
)

train_set = params.train_loader()
valid_set = params.valid_loader()
first_x = next(iter(train_set))

se_brain = SEBrain(
    modules=[params.model], optimizer=params.optimizer, first_inputs=first_x,
)

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
# with torch.autograd.detect_anomaly():
se_brain.fit(params.epoch_counter, train_set, valid_set)
