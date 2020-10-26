#!/usr/bin/python
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from pesq import pesq
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.loss.stoi_loss import stoi_loss


# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    def compute_forward(self, x, stage):
        ids, wavs, lens = x
        wavs, lens = wavs.to(self.device), lens.to(self.device)
        feats = self.hparams.compute_STFT(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        # mask with "signal approximation (SA)"
        mask = self.hparams.model(feats)
        mask = torch.squeeze(mask, 2)
        predict_spec = torch.mul(mask, feats)

        # Also return predicted wav
        predict_wav = self.resynthesize(torch.expm1(predict_spec), x)

        return predict_spec, predict_wav

    def compute_objectives(self, predictions, targets, stage):
        predict_spec, predict_wav = predictions
        ids, target_wav, lens = targets
        target_wav, lens = target_wav.to(self.device), lens.to(self.device)

        if getattr(self.hparams, "waveform_target", False):
            loss = self.hparams.compute_cost(predict_wav, target_wav, lens)
            self.loss_metric.append(
                ids, predict_wav, target_wav, lens, reduction="batch"
            )
        else:
            targets = self.hparams.compute_STFT(target_wav)
            targets = spectral_magnitude(targets, power=0.5)
            targets = torch.log1p(targets)
            loss = self.hparams.compute_cost(predict_spec, targets, lens)
            self.loss_metric.append(
                ids, predict_spec, targets, lens, reduction="batch"
            )

        if stage != sb.Stage.TRAIN:

            # Evaluate speech quality/intelligibility
            self.stoi_metric.append(
                ids, predict_wav, target_wav, lens, reduction="batch"
            )
            self.pesq_metric.append(
                ids, predict=predict_wav, target=target_wav, lengths=lens
            )

            # Write wavs to file
            if stage == sb.Stage.TEST:
                lens = lens * target_wav.shape[1]
                for name, pred_wav, length in zip(ids, predict_wav, lens):
                    name += ".wav"
                    enhance_path = os.path.join(
                        self.hparams.enhanced_folder, name
                    )
                    torchaudio.save(
                        enhance_path, predict_wav[: int(length)].cpu(), 16000
                    )

        return loss

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
            if self.hparams.use_tensorboard:
                valid_stats = {
                    "loss": self.loss_metric.scores,
                    "stoi": self.stoi_metric.scores,
                    "pesq": self.pesq_metric.scores,
                }
                self.hparams.tensorboard_train_logger.log_stats(
                    {"Epoch": epoch}, self.train_stats, valid_stats
                )
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["pesq"])

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def resynthesize(self, predictions, inputs):
        ids, wavs, lens = inputs
        lens = lens * wavs.shape[1]
        wavs = wavs.to(self.device)

        # Extract noisy phase
        feats = self.hparams.compute_STFT(wavs)
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

        # Get the predicted waveform
        pred_wavs = self.hparams.compute_ISTFT(complex_predictions)

        # Normalize the waveform
        abs_max, _ = torch.max(torch.abs(pred_wavs), dim=1, keepdim=True)
        pred_wavs = pred_wavs / abs_max * 0.99
        padding = (0, wavs.shape[1] - pred_wavs.shape[1])
        return torch.nn.functional.pad(pred_wavs, padding)


# Recipe begins!
if __name__ == "__main__":

    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from voicebank_prepare import prepare_voicebank  # noqa E402

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
    prepare_voicebank(
        data_folder=hparams["data_folder"], save_folder=hparams["data_folder"],
    )

    se_brain = SEBrain(
        modules=hparams["modules"],
        hparams=hparams["hparams"],
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
        device=hparams["device"],
    )

    # Load latest checkpoint to resume training
    se_brain.fit(
        hparams["epoch_counter"],
        train_set=hparams["train_loader"](),
        valid_set=hparams["valid_loader"](),
    )

    # Load best checkpoint for evaluation
    test_stats = se_brain.evaluate(hparams["test_loader"](), max_key="pesq")
