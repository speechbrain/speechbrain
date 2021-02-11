#!/usr/bin/env/python3
"""
Recipe for training a speech enhancement system with the Voicebank dataset.

To run this recipe, do the following:
> python train.py hparams/{hyperparam_file}.yaml

Authors
 * Szu-Wei Fu 2020
"""

import os
import sys
import shutil
import torch
import torchaudio
import speechbrain as sb
from pesq import pesq
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.distributed import run_on_main


class MetricGanBrain(sb.Brain):
    def compute_feats(self, wavs):
        """Feature computation pipeline"""
        feats = self.hparams.compute_STFT(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)
        return feats

    def compute_forward(self, batch, stage):
        "Given an input batch computes the enhanced signal"
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.noisy_sig
        noisy_spec = self.compute_feats(noisy_wavs)

        # mask with "signal approximation (SA)"
        mask = self.modules.generator(noisy_spec)
        mask = torch.maximum(
            mask,
            self.hparams.min_mask * torch.ones(mask.shape, device=self.device),
        )
        mask = torch.squeeze(mask, 2)
        predict_spec = torch.mul(mask, noisy_spec)

        # Also return predicted wav
        predict_wav = self.hparams.resynth(
            torch.expm1(predict_spec), noisy_wavs
        )

        return predict_wav, noisy_spec, noisy_wavs

    def compute_objectives(
        self,
        predictions,
        batch,
        stage,
        optim_name="",
        epoch=0,
        Replay_buffer=[],
        current_list=[],
        mode="clean",
    ):
        "Given the network predictions and targets computed the total loss"
        predict_wav, noisy_spec, noisy_wavs = predictions
        predict_spec = self.compute_feats(predict_wav)

        clean_wavs, lens = batch.clean_sig
        clean_spec = self.compute_feats(clean_wavs)
        batch_size = clean_wavs.size(0)

        mse_cost = self.hparams.compute_cost(predict_spec, clean_spec, lens)
        mse_weight = 0
        adv_cost = 0
        # One is real, zero is fake
        if optim_name == "generator":
            target_score = torch.ones(batch_size, 1, device=self.device)
            estimated_enhanced_score = self.modules.discriminator(
                torch.cat(
                    [
                        torch.unsqueeze(predict_spec, 1),
                        torch.unsqueeze(clean_spec, 1),
                    ],
                    1,
                )
            )

            adv_cost = self.hparams.compute_cost(
                estimated_enhanced_score, target_score
            )
            self.metrics["G"].append(adv_cost.detach())
            self.mse_metric.append(
                batch.id, predict_spec, clean_spec, lens, reduction="batch"
            )
        elif optim_name == "discriminator":
            if (
                mode == "clean"
            ):  # D Learns to estimate the scores of clean speech
                estimated_clean_score = self.modules.discriminator(
                    torch.cat(
                        [
                            torch.unsqueeze(clean_spec, 1),
                            torch.unsqueeze(clean_spec, 1),
                        ],
                        1,
                    )
                )
                true_score = torch.ones(batch_size, 1, device=self.device)

                adv_cost = self.hparams.compute_cost(
                    estimated_clean_score, true_score
                )

                for name in batch.id:
                    current_list.append(
                        "1," + self.hparams.train_clean_folder + name + ".wav"
                    )
            elif (
                mode == "enhanced"
            ):  # D Learns to estimate the scores of enhanced speech
                if self.hparams.TargetMetric == "pesq":
                    self.target_metric.append(
                        batch.id,
                        predict=predict_wav.detach(),
                        target=clean_wavs,
                        lengths=lens,
                    )
                    true_score = torch.tensor(
                        [[s] for s in self.target_metric.scores],
                        device=self.device,
                    )
                elif self.hparams.TargetMetric == "stoi":
                    self.target_metric.append(
                        batch.id,
                        predict_wav,
                        clean_wavs,
                        lens,
                        reduction="batch",
                    )
                    true_score = torch.tensor(
                        [[-s] for s in self.target_metric.scores],
                        device=self.device,
                    )
                estimated_enhanced_score = self.modules.discriminator(
                    torch.cat(
                        [
                            torch.unsqueeze(predict_spec, 1),
                            torch.unsqueeze(clean_spec, 1),
                        ],
                        1,
                    )
                )

                adv_cost = self.hparams.compute_cost(
                    estimated_enhanced_score, true_score
                )

                # Write wavs to files, for historical discriminator training
                lens = lens * clean_wavs.shape[1]
                i = 0
                for name, pred_wav, length in zip(batch.id, predict_wav, lens):
                    name += "@" + str(epoch) + ".wav"
                    enhance_path = os.path.join(
                        self.hparams.MetricGAN_folder, name
                    )
                    Replay_buffer.append(
                        str(true_score[i][0].cpu().numpy()) + "," + enhance_path
                    )
                    current_list.append(
                        str(true_score[i][0].cpu().numpy()) + "," + enhance_path
                    )
                    i += 1
                    torchaudio.save(
                        enhance_path,
                        torch.unsqueeze(pred_wav[: int(length)].cpu(), 0),
                        16000,
                    )
                self.target_metric.clear()
            elif (
                mode == "noisy"
            ):  # D Learns to estimate the scores of noisy speech
                if self.hparams.TargetMetric == "pesq":
                    self.target_metric.append(
                        batch.id,
                        predict=noisy_wavs,
                        target=clean_wavs,
                        lengths=lens,
                    )
                    true_score = torch.tensor(
                        [[s] for s in self.target_metric.scores],
                        device=self.device,
                    )
                elif self.hparams.TargetMetric == "stoi":
                    self.target_metric.append(
                        batch.id,
                        noisy_wavs,
                        clean_wavs,
                        lens,
                        reduction="batch",
                    )
                    true_score = torch.tensor(
                        [[-s] for s in self.target_metric.scores],
                        device=self.device,
                    )
                estimated_noisy_score = self.modules.discriminator(
                    torch.cat(
                        [
                            torch.unsqueeze(noisy_spec, 1),
                            torch.unsqueeze(clean_spec, 1),
                        ],
                        1,
                    )
                )

                adv_cost = self.hparams.compute_cost(
                    estimated_noisy_score, true_score
                )

                i = 0
                for name in batch.id:
                    current_list.append(
                        str(true_score[i][0].cpu().numpy())
                        + ","
                        + self.hparams.train_noisy_folder
                        + name
                        + ".wav"
                    )
                    i += 1

                self.target_metric.clear()
            self.metrics["D"].append(adv_cost.detach())

        if stage != sb.Stage.TRAIN:
            mse_weight = 1
            # Evaluate speech quality/intelligibility
            self.stoi_metric.append(
                batch.id, predict_wav, clean_wavs, lens, reduction="batch"
            )
            self.pesq_metric.append(
                batch.id, predict=predict_wav, target=clean_wavs, lengths=lens
            )

            # Write wavs to file, for evaluation
            lens = lens * clean_wavs.shape[1]
            for name, pred_wav, length in zip(batch.id, predict_wav, lens):
                name += ".wav"
                enhance_path = os.path.join(self.hparams.enhanced_folder, name)
                torchaudio.save(
                    enhance_path,
                    torch.unsqueeze(pred_wav[: int(length)].cpu(), 0),
                    16000,
                )

        return (
            adv_cost + mse_weight * mse_cost.detach()
        )  # we do not use mse_cost to update model

    def compute_D_list_objectives(self, batch, stage):
        batch_size = len(batch)
        true_score = torch.ones(batch_size, 1, device=self.device)
        estimated_enhanced_score = torch.ones(batch_size, 1, device=self.device)
        i = 0
        for b in batch:
            score_filepath = b.split(",")
            wave_name = b.split("/")[-1].split("@")[0]
            if ".wav" not in wave_name:
                wave_name += ".wav"

            clean_wav, _ = torchaudio.load(
                self.hparams.train_clean_folder + wave_name
            )
            clean_spec = self.compute_feats(clean_wav.to(self.device))

            enhance_wav, _ = torchaudio.load(score_filepath[1])
            if enhance_wav.shape != clean_wav.shape:
                enhance_wav = torch.cat(
                    (
                        enhance_wav,
                        torch.zeros(
                            1, clean_wav.shape[1] - enhance_wav.shape[1]
                        ),
                    ),
                    1,
                )
            enhance_spec = self.compute_feats(enhance_wav.to(self.device))

            estimated_enhanced_score[i] = self.modules.discriminator(
                torch.cat(
                    [
                        torch.unsqueeze(enhance_spec, 1),
                        torch.unsqueeze(clean_spec, 1),
                    ],
                    1,
                )
            )
            true_score[i] = torch.tensor(
                [[float(score_filepath[0])]], device=self.device
            )
            i += 1
        adv_cost = self.hparams.compute_cost(
            estimated_enhanced_score, true_score
        )
        return adv_cost

    def fit_D_batch(self, batch, epoch, Replay_buffer, current_list):
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)

        d_loss0 = self.compute_objectives(
            predictions,
            batch,
            sb.Stage.TRAIN,
            "discriminator",
            epoch,
            Replay_buffer,
            current_list,
            "clean",
        )
        self.d_optimizer.zero_grad()
        d_loss0.backward()
        self.d_optimizer.step()

        d_loss1 = self.compute_objectives(
            predictions,
            batch,
            sb.Stage.TRAIN,
            "discriminator",
            epoch,
            Replay_buffer,
            current_list,
            "enhanced",
        )
        self.d_optimizer.zero_grad()
        d_loss1.backward()
        self.d_optimizer.step()

        d_loss2 = self.compute_objectives(
            predictions,
            batch,
            sb.Stage.TRAIN,
            "discriminator",
            epoch,
            Replay_buffer,
            current_list,
            "noisy",
        )
        self.d_optimizer.zero_grad()
        d_loss2.backward()
        self.d_optimizer.step()

        return (d_loss0 + d_loss1 + d_loss2).detach() / 3

    def fit_D_list_batch(self, batch, epoch):
        d_loss = self.compute_D_list_objectives(batch, sb.Stage.TRAIN)
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.detach()

    def fit_G_batch(self, batch):
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        g_loss = self.compute_objectives(
            predictions, batch, sb.Stage.TRAIN, "generator"
        )
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch"""

        def pesq_eval(pred_wav, target_wav):
            return (
                pesq(
                    fs=16000,
                    ref=target_wav.numpy(),
                    deg=pred_wav.numpy(),
                    mode="wb",
                )
                + 0.5
            ) / 5

        self.mse_metric = MetricStats(metric=self.hparams.compute_cost)

        if stage == sb.Stage.TRAIN:
            self.metrics = {"G": [], "D": []}

            if self.hparams.TargetMetric == "pesq":
                self.target_metric = MetricStats(metric=pesq_eval, n_jobs=30)
            elif self.hparams.TargetMetric == "stoi":
                self.target_metric = MetricStats(metric=stoi_loss)
            else:
                raise NotImplementedError(
                    "Right now we only support 'pesq' and 'stoi'"
                )

        if stage != sb.Stage.TRAIN:
            self.pesq_metric = MetricStats(metric=pesq_eval, n_jobs=30)
            self.stoi_metric = MetricStats(metric=stoi_loss)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            # self.train_stats = {"mse": self.mse_metric.scores} # for tensorboard

            g_loss = torch.tensor(self.metrics["G"])  # batch_size
            d_loss = torch.tensor(self.metrics["D"])  # batch_size
            print("Avg G loss: %.3f" % torch.mean(g_loss))
            print("Avg D loss: %.3f" % torch.mean(d_loss))
            if epoch >= 2:
                print(
                    "MSE distance: %.3f" % self.mse_metric.summarize("average")
                )
        else:
            stats = {
                "MSE distance": stage_loss,
                "pesq": 5 * self.pesq_metric.summarize("average") - 0.5,
                "stoi": -self.stoi_metric.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            if self.hparams.use_tensorboard:
                valid_stats = {
                    "mse": stage_loss,
                    "pesq": 5 * self.pesq_metric.summarize("average") - 0.5,
                    "stoi": -self.stoi_metric.summarize("average"),
                }
                self.hparams.tensorboard_train_logger.log_stats(valid_stats)
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stats, max_keys=[self.hparams.TargetMetric]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def init_optimizers(self):
        """Initializes the generator and discriminator optimizers"""
        self.g_optimizer = self.hparams.g_opt_class(
            self.modules.generator.parameters()
        )
        self.d_optimizer = self.hparams.d_opt_class(
            self.modules.discriminator.parameters()
        )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio piplines
    @sb.utils.data_pipeline.takes("noisy_wav")
    @sb.utils.data_pipeline.provides("noisy_sig")
    def noisy_pipeline(noisy_wav):
        return sb.dataio.dataio.read_audio(noisy_wav)

    @sb.utils.data_pipeline.takes("clean_wav")
    @sb.utils.data_pipeline.provides("clean_sig")
    def clean_pipeline(clean_wav):
        return sb.dataio.dataio.read_audio(clean_wav)

    # Define datasets
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[noisy_pipeline, clean_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig"],
        )

    # Sort train dataset
    if hparams["sorting"] == "ascending" or hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="duration", reverse=hparams["sorting"] == "descending"
        )
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] != "random":
        raise NotImplementedError(
            "Sorting must be random, ascending, or descending"
        )

    return datasets


def create_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


# Recipe begins!
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Data preparation
    from voicebank_prepare import prepare_voicebank  # noqa

    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create dataset objects
    datasets = dataio_prep(hparams)

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

    # Create the folder to save enhanced files (+ support for DDP)
    run_on_main(create_folder, kwargs={"folder": hparams["enhanced_folder"]})

    se_brain = MetricGanBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    shutil.rmtree(hparams["MetricGAN_folder"])
    run_on_main(create_folder, kwargs={"folder": hparams["MetricGAN_folder"]})

    # Load latest checkpoint to resume training
    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        training_parameters=hparams["MetricGAN_additional_parameters"],
        valid_set=datasets["test"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )

    # Load best checkpoint for evaluation
    test_stats = se_brain.evaluate(
        test_set=datasets["test"],
        max_key=hparams["TargetMetric"],
        test_loader_kwargs=hparams["dataloader_options"],
    )
