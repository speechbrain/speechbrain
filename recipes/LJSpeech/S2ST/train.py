#!/usr/bin/env python3
"""Recipe for training a hifi-gan vocoder.
For more details about hifi-gan: https://arxiv.org/pdf/2010.05646.pdf

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder /path/to/LJspeech

Authors
 * Duret Jarod 2021
 * Yingzhi WANG 2022
"""

import sys
import json
import itertools
import torch
import copy
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.data_utils import scalarize
import torch
import torchaudio
import os
import numpy as np
import random
from utils.audio import extract_f0


class HifiGanBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """The forward function, generates synthesized waveforms,
        calculates the scores and the features of the discriminator
        for synthesized waveforms and real waveforms.

        Arguments
        ---------
        batch: str
            a single batch
        stage: speechbrain.Stage
            the training stage

        """
        batch = batch.to(self.device)

        x, _ = batch.feats
        y, _ = batch.sig

        f0 = False
        spk_emb = emo_emb = None
        if hparams["multi_speaker"]:
            spk_emb, _ = batch.spk_emb
        if hparams["multi_emotion"]:
            emo_emb, _ = batch.emo_emb
        if hparams["f0"]:
            f0, _ = batch.f0

        # generate sythesized waveforms
        y_g_hat, (log_dur_pred, log_dur) = self.modules.generator(x, f0=f0, spk=spk_emb, emo=emo_emb, stage=stage)
        y_g_hat = y_g_hat[:, :, : y.size(2)]

        # get scores and features from discriminator for real and synthesized waveforms
        scores_fake, feats_fake = self.modules.discriminator(y_g_hat.detach())
        scores_real, feats_real = self.modules.discriminator(y)

        return (
            y_g_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur
        )

    def compute_objectives(self, predictions, batch, stage):
        """Computes and combines generator and discriminator losses
        """
        batch = batch.to(self.device)

        x, _ = batch.feats
        y, _ = batch.sig

        (
            y_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur
        ) = predictions

        loss_g = self.hparams.generator_loss(
            stage, y_hat, y, scores_fake, feats_fake, feats_real, log_dur_pred, log_dur
        )
        loss_d = self.hparams.discriminator_loss(scores_fake, scores_real)
        loss = {**loss_g, **loss_d}
        self.last_loss_stats[stage] = scalarize(loss)
        return loss

    def fit_batch(self, batch):
        """Train discriminator and generator adversarially
        """

        batch = batch.to(self.device)
        y, _ = batch.sig

        outputs = self.compute_forward(batch, sb.core.Stage.TRAIN)
        (
            y_g_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur,
        ) = outputs
        # calculate discriminator loss with the latest updated generator
        loss_d = self.compute_objectives(outputs, batch, sb.core.Stage.TRAIN)[
            "D_loss"
        ]
        # First train the discriminator
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        # calculate generator loss with the latest updated discriminator
        scores_fake, feats_fake = self.modules.discriminator(y_g_hat)
        scores_real, feats_real = self.modules.discriminator(y)
        outputs = (
            y_g_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur,
        )
        loss_g = self.compute_objectives(outputs, batch, sb.core.Stage.TRAIN)[
            "G_loss"
        ]
        # Then train the generator
        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch
        """
        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        loss_g = loss["G_loss"]
        return loss_g.detach().cpu()

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics
        """
        self.last_epoch = 0
        # self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """
        if self.opt_class is not None:
            (
                opt_g_class,
                opt_d_class,
                sch_g_class,
                sch_d_class,
            ) = self.opt_class

            self.optimizer_g = opt_g_class(self.modules.generator.parameters())
            self.optimizer_d = opt_d_class(
                self.modules.discriminator.parameters()
            )
            self.scheduler_g = sch_g_class(self.optimizer_g)
            self.scheduler_d = sch_d_class(self.optimizer_d)

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "optimizer_g", self.optimizer_g
                )
                self.checkpointer.add_recoverable(
                    "optimizer_d", self.optimizer_d
                )
                self.checkpointer.add_recoverable(
                    "scheduler_g", self.scheduler_d
                )
                self.checkpointer.add_recoverable(
                    "scheduler_d", self.scheduler_d
                )

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage (TRAIN, VALID, Or TEST)
        """
        if stage == sb.Stage.VALID:
            # Update learning rate
            self.scheduler_g.step()
            self.scheduler_d.step()
            lr_g = self.optimizer_g.param_groups[-1]["lr"]
            lr_d = self.optimizer_d.param_groups[-1]["lr"]

            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr_g": lr_g, "lr_d": lr_d},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )
            # The tensorboard_logger writes a summary to stdout and to the logfile.
            if self.hparams.use_wandb:
                self.wandb_logger.log_stats(
                    stats_meta={"Epoch": epoch, "lr_g": lr_g, "lr_d": lr_d},
                    train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                    valid_stats=self.last_loss_stats[sb.Stage.VALID],
                )

            # Save the current checkpoint and delete previous checkpoints.
            epoch_metadata = {
                **{"epoch": epoch},
                **self.last_loss_stats[sb.Stage.VALID],
            }
            if self.checkpointer is not None:
                self.checkpointer.save_and_keep_only(
                    meta=epoch_metadata,
                    end_of_epoch=True,
                    min_keys=["loss"],
                    ckpt_predicate=(
                        lambda ckpt: (
                            ckpt.meta["epoch"]
                            % self.hparams.keep_checkpoint_interval
                            != 0
                        )
                    )
                    if self.hparams.keep_checkpoint_interval is not None
                    else None,
                )

            self.run_inference_sample("Test", epoch)

        # We also write statistics about test data to stdout and to the TensorboardLogger.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(  # 1#2#
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )
            if self.hparams.use_wandb:
                self.wandb_logger.log_stats(
                    {"Epoch loaded": self.hparams.epoch_counter.current},
                    test_stats=self.last_loss_stats[sb.Stage.TEST],
                )
            self.run_inference_sample("Test", epoch)

    def run_inference_sample(self, name, epoch):
        """Produces a sample in inference mode. This is called when producing
        samples.
        """

        if self.inference_set:
            # Preparing model for inference by removing weight norm
            inference_generator = copy.deepcopy(self.hparams.generator)
            inference_generator.remove_weight_norm()

            for item in self.inference_set:
                with torch.no_grad():
                    # if self.last_batch is None:
                    #     return
                    # x, y = self.last_batch

                    x = item['feats'].unsqueeze(0).to(self.device)
                    y = item['sig'].unsqueeze(0).to(self.device)
                    uttid = item['id']

                    f0 = False
                    spk_emb = emo_emb = None
                    if hparams["multi_speaker"]:
                        spk_emb = item['spk_emb'].unsqueeze(0).to(self.device)
                    if hparams["multi_emotion"]:
                        emo_emb = item['emo_emb'].unsqueeze(0).to(self.device)
                    if hparams['f0']:
                        f0 = item['f0'].unsqueeze(0).to(self.device)

                    if inference_generator.duration_predictor:
                        x = torch.unique_consecutive(x, dim=1)

                    sig_out = inference_generator.inference(x, f0=f0, spk=spk_emb, emo=emo_emb)
                    spec_out = self.hparams.mel_spectogram(
                        audio=sig_out.squeeze(0).cpu()
                    )
                    spec_int = self.hparams.mel_spectogram(
                        audio=y.squeeze(0).cpu()
                    )
                if self.hparams.use_wandb:
                    self.wandb_logger.log_audio(
                        f"{name}_{uttid}/audio_target", 
                        y.reshape(-1).cpu().numpy(),
                        self.hparams.sample_rate,
                        "target",
                        epoch
                    )
                    self.wandb_logger.log_audio(
                        f"{name}_{uttid}/audio_pred",
                        sig_out.reshape(-1).cpu().numpy(),
                        self.hparams.sample_rate,
                        "pred",
                        epoch
                    )
                    self.wandb_logger.log_figure(f"{name}_{uttid}/mel_target", spec_int, "mel_target", epoch)
                    self.wandb_logger.log_figure(f"{name}_{uttid}/mel_pred", spec_out, "mel_pred", epoch)


def sample_interval(seqs, segment_size):
    N = max([v.shape[-1] for v in seqs])
    seq_len = segment_size if segment_size > 0 else N

    hops = [N // v.shape[-1] for v in seqs]
    lcm = np.lcm.reduce(hops)

    # Randomly pickup with the batch_max_steps length of the part
    interval_start = 0
    interval_end = N // lcm - seq_len // lcm

    start_step = random.randint(interval_start, interval_end)

    new_seqs = []
    for i, v in enumerate(seqs):
        start = start_step * (lcm // hops[i])
        end = (start_step + seq_len // lcm) * (lcm // hops[i])
        new_seqs += [v[..., start:end]]

    return new_seqs

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    segment_size = hparams["segment_size"]
    code_hop_size = hparams["code_hop_size"]

    from utils.embedding import EmbeddingManager
    units_loader = EmbeddingManager(hparams["units_folder"])

    if hparams["multi_speaker"]:
        speakers_loader = EmbeddingManager(hparams["speakers_folder"])

    if hparams["multi_emotion"]:
        emotions_loader = EmbeddingManager(hparams["emotions_folder"])

    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "id", "segment")
    @sb.utils.data_pipeline.provides("feats", "sig", "f0")
    def audio_pipeline(wav, clip_id, segment):
        audio = sb.dataio.dataio.read_audio(wav)
        features = units_loader.get_embedding_by_clip(clip_id)
        features = torch.IntTensor(features)

        # Trim audio ending
        code_length = min(audio.shape[0] // code_hop_size, features.shape[0])
        code = features[:code_length]
        audio = audio[:code_length * code_hop_size]

        assert audio.shape[0] // code_hop_size == code.shape[0], "Code audio mismatch"

        while audio.shape[0] < segment_size:
            audio = torch.hstack([audio, audio])
            code = torch.hstack([code, code])

        audio = audio.unsqueeze(0)
        assert audio.size(1) >= segment_size, "Padding not supported!!"

        if segment:
            audio, code = sample_interval([audio, code], segment_size)

        f0 = None
        if hparams["f0"]:
            try:
                f0 = extract_f0(audio.squeeze(0).numpy(), sr=hparams["sample_rate"], extractor="parselmouth").astype(np.float32)
            except:
                f0 = np.zeros((1, 1, audio.shape[-1] // 80))
            f0 = torch.from_numpy(f0).unsqueeze(0)
        
        return code, audio, f0

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("spk_emb")
    def spk_pipeline(utt_id):
        spk_emb = speakers_loader.get_embedding_by_clip(utt_id)
        yield torch.FloatTensor(spk_emb)
    
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("emo_emb")
    def emo_pipeline(utt_id):
        emo_emb = emotions_loader.get_embedding_by_clip(utt_id)
        yield torch.FloatTensor(emo_emb)

    pipelines = [audio_pipeline]
    keys = ["id", "sig", "feats", "f0"]
    if hparams["multi_speaker"]:
        pipelines.append(spk_pipeline)
        keys.append("spk_emb")
    if hparams["multi_emotion"]:
        pipelines.append(emo_pipeline)
        keys.append("emo_emb")

    datasets = {}
    for split in hparams["splits"]:
        ds_dict = {}
        for ds_path in hparams[f"{split}_json"]:
            data = json.load(open(ds_path))
            if split == "train":
                for key in data: data[key]['segment'] = True
            else:
                for key in data: data[key]['segment'] = False
            ds_dict.update(data)
        datasets[split] = sb.dataio.dataset.DynamicItemDataset(
            ds_dict,
            dynamic_items=pipelines,
            output_keys=keys,
        )

    return datasets


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets = dataio_prepare(hparams)

    # Brain class initialization
    hifi_gan_brain = HifiGanBrain(
        modules=hparams["modules"],
        opt_class=[
            hparams["opt_class_generator"],
            hparams["opt_class_discriminator"],
            hparams["sch_class_generator"],
            hparams["sch_class_discriminator"],
        ],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if hparams['test_json']:
        hifi_gan_brain.inference_set = datasets["test"]

    if hparams["use_wandb"]:
        from utils.logger import WandBLogger
        hifi_gan_brain.wandb_logger = WandBLogger(**hparams["logger_opts"])

    # Training
    hifi_gan_brain.fit(
        hifi_gan_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )