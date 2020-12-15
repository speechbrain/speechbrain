#!/usr/bin/python
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from pesq import pesq
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.nnet.loss.stoi_loss import stoi_loss


# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    def compute_forward(self, x, stage):
        ids, wavs, lens = x
        wavs, lens = wavs.to(self.device), lens.to(self.device)
        wavs = torch.unsqueeze(wavs, -1)
        predict_wav = self.hparams.model(wavs)[:, :, 0]

        return predict_wav

    def compute_objectives(self, predict_wav, targets, stage):
        ids, target_wav, lens = targets
        target_wav, lens = target_wav.to(self.device), lens.to(self.device)

        loss = self.hparams.compute_cost(predict_wav, target_wav, lens)
        self.loss_metric.append(
            ids, predict_wav, target_wav, lens, reduction="batch"
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
                    pred_wav = pred_wav / torch.max(torch.abs(pred_wav)) * 0.99
                    torchaudio.save(
                        enhance_path,
                        torch.unsqueeze(pred_wav[: int(length)].cpu(), 0),
                        16000,
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

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )

    # Create the folder to save enhanced files
    if not os.path.exists(hparams["enhanced_folder"]):
        os.mkdir(hparams["enhanced_folder"])

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
