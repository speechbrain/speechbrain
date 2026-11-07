#!/usr/bin/env python3
"""Recipe for training MP-SENet on VoiceBank-DEMAND.

To run this recipe:
> python train.py hparams/train.yaml --data_folder /path/to/noisy-vctk-16k

Authors
 * Manil Bendali 2026
"""

import functools
import os
import sys

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from pesq import PesqError, pesq

import speechbrain as sb
from speechbrain.dataio import audio_io
from speechbrain.lobes.models.MPSENet import phase_losses
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.metric_stats import MetricStats


def pesq_eval(pred_wav, target_wav):
    """Wide-band PESQ at 16 kHz, or -1 when the pair can't be scored."""
    return pesq(
        16000,
        target_wav.numpy(),
        pred_wav.numpy(),
        "wb",
        on_error=PesqError.RETURN_VALUES,
    )


class MPSENetBrain(sb.Brain):
    """Adversarial training: metric discriminator step, then generator step."""

    def compute_forward(self, batch, stage):
        """Enhances the noisy waveforms."""
        batch = batch.to(self.device)
        noisy, _ = batch.noisy_sig
        mag, pha, _ = self.hparams.stft(noisy)
        mag_g, pha_g, com_g = self.modules.generator(mag, pha)
        wav_g = self.hparams.istft(mag_g, pha_g, length=noisy.shape[-1])
        return mag_g, pha_g, com_g, wav_g

    def compute_objectives(self, predictions, batch, stage):
        """Computes the generator loss."""
        mag_g, pha_g, com_g, wav_g = predictions
        clean, lens = batch.clean_sig
        clean_mag, clean_pha, clean_com = self.hparams.stft(clean)
        # Re-analysis of the output, for the consistency and metric losses
        mag_hat, _, com_hat = self.hparams.stft(wav_g)
        score = self.modules.discriminator(clean_mag, mag_hat).flatten()

        losses = {
            "mag": F.mse_loss(mag_g, clean_mag),
            "pha": sum(phase_losses(clean_pha, pha_g)),
            "com": F.mse_loss(com_g, clean_com),
            "consistency": F.mse_loss(com_g, com_hat),
            "time": F.l1_loss(wav_g, clean),
            "metric": F.mse_loss(score, torch.ones_like(score)),
        }
        weights = self.hparams.loss_weights
        loss = sum(weights[name] * value for name, value in losses.items())

        if stage != sb.Stage.TRAIN:
            self.pesq_metric.append(
                batch.id, predict=wav_g, target=clean, lengths=lens
            )
            self.stoi_metric.append(
                batch.id, wav_g, clean, lens, reduction="batch"
            )
        if stage == sb.Stage.TEST:
            self.write_wavs(batch, wav_g)
        return loss

    def batch_pesq(self, clean, enhanced):
        """Discriminator targets: PESQ scaled to [0, 1], or None if any
        pair can't be scored (as in the original implementation)."""
        # ponytail: sequential PESQ on CPU, parallelize if it bottlenecks.
        scores = [pesq_eval(e, c) for c, e in zip(clean.cpu(), enhanced.cpu())]
        if -1 in scores:
            return None
        return (torch.tensor(scores, device=self.device) - 1) / 3.5

    def fit_batch(self, batch):
        """Trains the discriminator, then the generator."""
        batch = batch.to(self.device)
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        wav_g = outputs[3].detach()
        clean, _ = batch.clean_sig
        clean_mag = self.hparams.stft(clean)[0]
        mag_hat = self.hparams.stft(wav_g)[0]

        # Discriminator: clean pairs score 1, enhanced pairs their PESQ
        score_clean = self.modules.discriminator(clean_mag, clean_mag).flatten()
        loss_d = F.mse_loss(score_clean, torch.ones_like(score_clean))
        pesq_target = self.batch_pesq(clean, wav_g)
        if pesq_target is not None:
            score_g = self.modules.discriminator(clean_mag, mag_hat).flatten()
            loss_d = loss_d + F.mse_loss(score_g, pesq_target)
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        loss_g = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()
        return loss_g.detach().cpu()

    def write_wavs(self, batch, wavs):
        """Saves enhanced waveforms at the original loudness."""
        for name, wav, norm in zip(
            batch.id, wavs, batch.norm_factor.data[:, 0]
        ):
            audio_io.save(
                os.path.join(self.hparams.enhanced_folder, name + ".wav"),
                (wav / norm).unsqueeze(0).cpu(),
                self.hparams.sample_rate,
            )

    def init_optimizers(self):
        """Creates one optimizer and LR scheduler per network."""
        self.optimizer_g = self.hparams.opt_class_generator(
            self.modules.generator.parameters()
        )
        self.optimizer_d = self.hparams.opt_class_discriminator(
            self.modules.discriminator.parameters()
        )
        self.scheduler_g = self.hparams.sch_class_generator(self.optimizer_g)
        self.scheduler_d = self.hparams.sch_class_discriminator(
            self.optimizer_d
        )
        self.optimizers_dict = {
            "optimizer_g": self.optimizer_g,
            "optimizer_d": self.optimizer_d,
        }
        if self.checkpointer is not None:
            self.checkpointer.add_recoverables(
                {
                    **self.optimizers_dict,
                    "scheduler_g": self.scheduler_g,
                    "scheduler_d": self.scheduler_d,
                }
            )

    def on_stage_start(self, stage, epoch=None):
        """Sets up PESQ and STOI tracking for validation and test."""
        if stage != sb.Stage.TRAIN:
            self.pesq_metric = MetricStats(
                metric=pesq_eval, n_jobs=self.hparams.n_jobs, batch_eval=False
            )
            self.stoi_metric = MetricStats(metric=stoi_loss)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Logs stats, steps the LR schedulers and saves checkpoints."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return
        stats = {
            "loss": stage_loss,
            "pesq": self.pesq_metric.summarize("average"),
            "stoi": -self.stoi_metric.summarize("average"),
        }
        if stage == sb.Stage.VALID:
            lr = self.optimizer_g.param_groups[0]["lr"]
            self.scheduler_g.step()
            self.scheduler_d.step()
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["pesq"])
        else:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def audio_pipeline(noisy_wav, clean_wav, segment_size=None):
    """Loads a noisy/clean pair, RMS-normalizes it and optionally crops it.

    Defined at module level so dataloader workers can pickle it.
    """
    noisy = sb.dataio.dataio.read_audio(noisy_wav)
    clean = sb.dataio.dataio.read_audio(clean_wav)
    length = min(len(noisy), len(clean))
    noisy, clean = noisy[:length], clean[:length]
    # Scale both so the noisy input has unit RMS
    norm = torch.sqrt(length / noisy.pow(2).sum())
    noisy, clean = noisy * norm, clean * norm
    if segment_size is not None:
        if length >= segment_size:
            start = torch.randint(length - segment_size + 1, ()).item()
            noisy = noisy[start : start + segment_size]
            clean = clean[start : start + segment_size]
        else:
            noisy = F.pad(noisy, (0, segment_size - length))
            clean = F.pad(clean, (0, segment_size - length))
    yield noisy
    yield clean
    yield norm.reshape(1)  # 1-element tensor so it batches as float32


def dataio_prep(hparams):
    """Creates the datasets. Training uses random fixed-length crops."""
    datasets = {}
    for split in ["train", "valid", "test"]:
        segment_size = hparams["segment_size"] if split == "train" else None
        pipeline = sb.utils.data_pipeline.takes("noisy_wav", "clean_wav")(
            sb.utils.data_pipeline.provides(
                "noisy_sig", "clean_sig", "norm_factor"
            )(functools.partial(audio_pipeline, segment_size=segment_size))
        )
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{split}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[pipeline],
            output_keys=["id", "noisy_sig", "clean_sig", "norm_factor"],
        )
    return datasets


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from voicebank_prepare import prepare_voicebank  # noqa

    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )
    datasets = dataio_prep(hparams)
    os.makedirs(hparams["enhanced_folder"], exist_ok=True)

    # Optional: start from pretrained weights (e.g. a converted checkpoint)
    if "pretrainer" in hparams:
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected()

    se_brain = MPSENetBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )
    se_brain.evaluate(
        test_set=datasets["test"],
        max_key="pesq",
        test_loader_kwargs=hparams["valid_dataloader_options"],
    )
