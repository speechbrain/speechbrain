#!/usr/bin/env python3
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.loss.stoi_loss import stoi_loss

torchaudio.set_audio_backend("sox_io")

try:
    from pesq import pesq
except ImportError:
    print("Please install PESQ from https://pypi.org/project/pesq/")


class SEBrain(sb.core.Brain):
    def compute_forward(self, x, stage):
        ids, wavs, lens = x
        wavs, lens = wavs.to(self.device), lens.to(self.device)

        feats = self.hparams.compute_STFT(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        predict_spec = self.hparams.model(feats)

        # Also return predicted wav
        if stage != sb.Stage.TRAIN:
            predict_wav = self.resynthesize(torch.expm1(predict_spec), wavs)
        else:
            predict_wav = None

        return predict_spec, predict_wav

    def compute_objectives(self, predictions, cleans, stage):
        predict_spec, predict_wav = predictions
        ids, clean_wav, lens = cleans
        clean_wav, lens = clean_wav.to(self.device), lens.to(self.device)

        feats = self.hparams.compute_STFT(clean_wav)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        loss = self.hparams.compute_cost(predict_spec, feats, lens)

        self.loss_metric.append(
            ids, predict_spec, feats, lens, reduction="batch"
        )

        if stage != sb.Stage.TRAIN:
            # Evaluate speech quality/intelligibility
            self.stoi_metric.append(
                ids, predict_wav, clean_wav, lens, reduction="batch"
            )
            self.pesq_metric.append(
                ids, predict=predict_wav, target=clean_wav, lengths=lens
            )

        # Write wavs to file
        if stage == sb.Stage.TEST:
            lens = lens * clean_wav.shape[1]
            for name, wav, length in zip(ids, predict_wav, lens):
                enhance_path = os.path.join(self.hparams.enhanced_folder, name)
                if not enhance_path.endswith(".wav"):
                    enhance_path = enhance_path + ".wav"
                torchaudio.save(
                    enhance_path,
                    torch.unsqueeze(wav[: int(length)].cpu(), 0),
                    16000,
                )

        return loss

    def fit_batch(self, batch):
        cleans = batch[0]
        ids, clean_wavs, lens = cleans

        # Dynamically mix noises
        noisy_wavs = self.hparams.add_noise(clean_wavs, lens)

        predictions = self.compute_forward(
            [ids, noisy_wavs, lens], sb.Stage.TRAIN
        )
        loss = self.compute_objectives(predictions, cleans, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch=None):
        self.loss_metric = MetricStats(metric=self.hparams.compute_cost)
        self.stoi_metric = MetricStats(metric=stoi_loss)

        # Define function taking (prediction, target) for parallel eval
        def pesq_eval(pred_wav, target_wav):
            return pesq(
                fs=16000,
                ref=target_wav.numpy(),
                deg=pred_wav.numpy(),
                mode="wb",
            )

        if stage != sb.Stage.TRAIN:
            self.pesq_metric = MetricStats(metric=pesq_eval, n_jobs=30)

    def on_stage_end(self, stage, stage_loss, epoch=None):

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {"loss": self.loss_metric.scores}
        else:
            stats = {
                "loss": stage_loss,
                "pesq": self.pesq_metric.summarize("average"),
                "stoi": -self.stoi_metric.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(4.5 - stats["pesq"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            if self.hparams.use_tensorboard:
                valid_stats = {
                    "loss": self.loss_metric.scores,
                    "stoi": self.stoi_metric.scores,
                    "pesq": self.pesq_metric.scores,
                }
                self.hparams.tensorboard_train_logger.log_stats(
                    {"Epoch": epoch, "lr": old_lr},
                    self.train_stats,
                    valid_stats,
                )

            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["pesq"])

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def resynthesize(self, predictions, noisy_wav):
        # Extract noisy phase
        feats = self.hparams.compute_STFT(noisy_wav)
        phase = torch.atan2(feats[..., 1], feats[..., 0])
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

        # Get the predicted waveform
        pred_wavs = self.hparams.compute_ISTFT(complex_predictions)

        # Normalize the waveform
        abs_max, _ = torch.max(torch.abs(pred_wavs), dim=1, keepdim=True)
        pred_wavs = pred_wavs / abs_max * 0.99

        padding = (0, noisy_wav.shape[1] - pred_wavs.shape[1])
        return torch.nn.functional.pad(pred_wavs, padding)


# Recipe begins!
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["hparams"]["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )

    # Create the folder to save enhanced files (+ support for DDP)
    try:
        # all writing command must be done with the main_process
        if sb.if_main_process():
            if not os.path.isdir(hparams["enhanced_folder"]):
                os.makedirs(hparams["enhanced_folder"])
    finally:
        # wait for main_process if ddp is used
        sb.ddp_barrier()


    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Load latest checkpoint to resume training
    se_brain.fit(
        se_brain.hparams.epoch_counter,
        train_set=hparams["train_loader"](),
        valid_set=hparams["valid_loader"](),
    )

    # Load best checkpoint for evaluation
    test_stats = se_brain.evaluate(hparams["test_loader"](), max_key="pesq")
