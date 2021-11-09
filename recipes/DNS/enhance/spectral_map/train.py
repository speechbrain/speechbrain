#!/usr/bin/env/python3
"""Recipe for training a speech enhancement system with the DNS dataset.
The system is based on spectral masking.

To run this recipe, do the following:
> python train.py hparams/{hyperparam_file}.yaml

Authors
 * Chien-Feng Liao 2020
"""
import os
import sys
import torch
import logging
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.torch_audio_backend import get_torchaudio_backend

torchaudio_backend = get_torchaudio_backend()
torchaudio.set_audio_backend(torchaudio_backend)
logger = logging.getLogger(__name__)

try:
    from pesq import pesq
except ImportError:
    print("Please install PESQ from https://pypi.org/project/pesq/")


class SEBrain(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the enhanced output."""
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.noisy_sig

        feats = self.hparams.compute_STFT(noisy_wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        predict_spec = self.hparams.model(feats)

        # Also return predicted wav
        if stage != sb.Stage.TRAIN:
            predict_wav = self.hparams.resynth(
                torch.expm1(predict_spec), noisy_wavs
            )
        else:
            predict_wav = None

        return predict_spec, predict_wav

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs"""
        predict_spec, predict_wav = predictions
        ids = batch.id
        clean_wav, lens = batch.clean_sig

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
                batch.id, predict=predict_wav, target=clean_wav, lengths=lens
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
                    self.hparams.Sample_rate,
                )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch"""
        self.loss_metric = MetricStats(metric=self.hparams.compute_cost)
        self.stoi_metric = MetricStats(metric=stoi_loss)

        # Define function taking (prediction, target) for parallel eval
        def pesq_eval(pred_wav, target_wav):
            """Computes the PESQ evaluation metric"""
            return pesq(
                fs=16000,
                ref=target_wav.cpu().numpy(),
                deg=pred_wav.cpu().numpy(),
                mode="wb",
            )

        if stage != sb.Stage.TRAIN:
            self.pesq_metric = MetricStats(
                metric=pesq_eval, n_jobs=1, batch_eval=False
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
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


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    # Define audio pipelines
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("clean_sig", "noisy_sig")
    def train_pipeline(wav):
        clean_sig = sb.dataio.dataio.read_audio(wav)
        noisy_sig = hparams["add_noise"](
            clean_sig.unsqueeze(0), torch.Tensor([1])
        )
        return clean_sig, noisy_sig.squeeze(0)

    @sb.utils.data_pipeline.takes("wav", "target")
    @sb.utils.data_pipeline.provides("clean_sig", "noisy_sig")
    def eval_pipeline(wav, target):
        noisy_sig = sb.dataio.dataio.read_audio(wav)
        clean_sig = sb.dataio.dataio.read_audio(target)
        return clean_sig, noisy_sig

    # Define datasets
    train_set = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": hparams["data_folder"]},
        dynamic_items=[train_pipeline],
        output_keys=["id", "clean_sig", "noisy_sig"],
    )

    valid_set = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": hparams["data_folder"]},
        dynamic_items=[eval_pipeline],
        output_keys=["id", "clean_sig", "noisy_sig"],
    )

    test_set = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"data_root": hparams["data_folder"]},
        dynamic_items=[eval_pipeline],
        output_keys=["id", "clean_sig", "noisy_sig"],
    )

    return train_set, valid_set, test_set


# Recipe begins!
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Data preparation
    from dns_prepare import prepare_dns  # noq

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(
        prepare_dns,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "valid_folder": hparams["valid_folder"],
            "seg_size": 10.0,
            "skip_prep": hparams["skip_prep"],
        },
    )

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["hparams"]["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )

    # Create the folder to save enhanced files (+ support for DDP)
    try:
        # all writing command must be done with the main_process
        if sb.utils.distributed.if_main_process():
            if not os.path.isdir(hparams["enhanced_folder"]):
                os.makedirs(hparams["enhanced_folder"])
    finally:
        # wait for main_process if ddp is used
        sb.utils.distributed.ddp_barrier()

    # Create dataset objects
    train_set, valid_set, test_set = dataio_prep(hparams)

    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Load latest checkpoint to resume training
    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=train_set,
        valid_set=valid_set,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load best checkpoint for evaluation
    test_stats = se_brain.evaluate(test_set, max_key="pesq")
