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

EPS = 1e-8


class SEBrain(sb.core.Brain):
    def compute_forward(self, x, stage):
        ids, wavs, lens = x
        wavs, lens = wavs.to(self.device), lens.to(self.device)

        feats = self.hparams.compute_STFT(wavs)  # [N, T, F, 2]
        output = self.hparams.model(feats)

        # Extract magnitude
        noisy_mag = spectral_magnitude(feats, power=0.5)
        output_mag = spectral_magnitude(output, power=0.5)

        # Extract phase
        noisy_phase = torch.atan2(feats[..., 1] + EPS, feats[..., 0] + EPS)
        output_phase = torch.atan2(output[..., 1] + EPS, output[..., 0] + EPS)

        # enhanced = |X||M| * e^(X_phase + M_phase)
        predict_spec = torch.mul(
            torch.unsqueeze(
                noisy_mag * self.hparams.mask_activation(output_mag), -1
            ),
            torch.cat(
                (
                    torch.unsqueeze(torch.cos(noisy_phase + output_phase), -1),
                    torch.unsqueeze(torch.sin(noisy_phase + output_phase), -1),
                ),
                -1,
            ),
        )

        predict_wav = self.hparams.compute_ISTFT(predict_spec)
        padding = (0, wavs.shape[1] - predict_wav.shape[1])
        predict_wav = torch.nn.functional.pad(predict_wav, padding)

        if stage != sb.Stage.TRAIN:
            # Normalize the waveform
            abs_max, _ = torch.max(torch.abs(predict_wav), dim=1, keepdim=True)
            predict_wav = predict_wav / abs_max * 0.99

        return predict_wav

    def compute_sisnr(self, est_target, target, lens, reduction="batch"):
        assert target.size() == est_target.size()

        # Step 1. Zero-mean norm
        mean_source = torch.mean(target, dim=1, keepdim=True)
        mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
        target = target - mean_source
        est_target = est_target - mean_estimate

        # Step 2. Pair-wise SI-SNR.
        # [batch, 1]
        dot = torch.sum(est_target * target, dim=1, keepdim=True)
        # [batch, 1]
        s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + EPS
        # [batch, time]
        scaled_target = dot * target / s_target_energy

        e_noise = scaled_target - est_target
        # [batch]
        losses = torch.sum(scaled_target ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + EPS
        )
        # take log
        losses = 10 * torch.log10(losses + EPS)

        if reduction == "batch":
            return -losses
        else:
            return -torch.mean(losses)

    def compute_objectives(self, predictions, cleans, stage):
        ids, clean_wav, lens = cleans
        clean_wav, lens = clean_wav.to(self.device), lens.to(self.device)

        loss = self.compute_sisnr(
            predictions, clean_wav, lens, reduction="mean"
        )

        self.loss_metric.append(ids, predictions, clean_wav, lens)

        if stage != sb.Stage.TRAIN:
            # Evaluate speech quality/intelligibility
            self.stoi_metric.append(
                ids, predictions, clean_wav, lens, reduction="batch"
            )
            self.pesq_metric.append(
                ids, predict=predictions, target=clean_wav, lengths=lens
            )

        # Write wavs to file
        if stage == sb.Stage.TEST and self.root_process:
            lens = lens * clean_wav.shape[1]
            for name, wav, length in zip(ids, predictions, lens):
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
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch=None):
        self.loss_metric = MetricStats(metric=self.compute_sisnr)
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

            # In distributed setting, only want to save model/stats once
            if self.root_process:
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
                self.checkpointer.save_and_keep_only(
                    meta=stats, max_keys=["pesq"]
                )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


# Recipe begins!
if __name__ == "__main__":

    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from dns_prepare import prepare_dns  # noqa E402

    # Load hyperparameters file with command-line overrides
    hparams_file, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

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

    # Create the folder to save enhanced files
    if not os.path.exists(hparams["enhanced_folder"]):
        os.mkdir(hparams["enhanced_folder"])

    # Prepare data
    prepare_dns(
        data_folder=hparams["data_folder"],
        save_folder=hparams["data_folder"],
        valid_folder=os.path.join(hparams["data_folder"], "valid"),
        seg_size=10.0,
    )

    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
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
