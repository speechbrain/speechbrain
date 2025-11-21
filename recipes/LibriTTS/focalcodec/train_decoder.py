#!/usr/bin/env/python

"""Recipe for training a decoder from continuous audio representations to waveform.

To run this recipe:
> python train_decoder.py hparams/<path-to-config>.yaml

Authors
 * Luca Della Libera 2025
"""

import os
import random
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataio import write_audio
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.utils.distributed import if_main_process, run_on_main


class Resynthesis(sb.Brain):
    def fit_batch(self, batch):
        """Fit one batch."""
        amp = sb.core.AMPConfig.from_name(self.precision)

        # Train discriminator
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=amp.dtype,
            enabled=self.precision != torch.float32,
        ):
            self.extract_feats(batch, sb.Stage.TRAIN)
            self.compute_forward_generator(batch, sb.Stage.TRAIN)
            outputs = self.compute_forward_discriminator(batch, sb.Stage.TRAIN)
            loss_discriminator = self.compute_objectives_discriminator(
                outputs, batch, sb.Stage.TRAIN
            )

        self.scaler.scale(loss_discriminator).backward()
        self.scaler.step(self.optimizer_discriminator)
        self.optimizer_discriminator.zero_grad(set_to_none=True)
        del loss_discriminator

        # Train generator
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=amp.dtype,
            enabled=self.precision != torch.float32,
        ):
            outputs = self.compute_forward_discriminator(
                batch, sb.Stage.TRAIN, return_discriminator=False
            )
            loss_generator = self.compute_objectives_generator(
                outputs, batch, sb.Stage.TRAIN
            )

        self.scaler.scale(loss_generator).backward()
        self.scaler.step(self.optimizer_generator)
        self.optimizer_generator.zero_grad(set_to_none=True)

        self.scaler.update()
        self.optimizer_step += 1
        self.step = self.optimizer_step

        # Cleanup
        batch.sig = batch.hyp_sig = None

        return loss_generator.detach().cpu()

    def extract_feats(self, batch, stage):
        """Extract continuous audio features from waveform."""
        batch = batch.to(self.device)
        sig, lens = batch.sig

        # Augment if specified
        if stage == sb.Stage.TRAIN and self.hparams.augment:
            sig, lens = self.hparams.augmentation(sig, lens)

        # Extract features
        with torch.no_grad():
            self.hparams.encoder.to(self.device).eval()
            feats, *encoder_state_ = self.hparams.encoder(sig, length=lens)

        # Extract segments
        if (
            stage == sb.Stage.TRAIN
            and self.hparams.segment_size_feats is not None
        ):
            segment_size_feats = self.hparams.segment_size_feats
            abs_lens = (
                (feats.shape[1] * lens)
                .ceil()
                .clamp(min=segment_size_feats, max=feats.shape[1])
                .long()
            )
            max_starts = abs_lens - segment_size_feats  # [B]
            starts = (
                torch.rand(feats.shape[0], device=self.device)
                * (max_starts + 1).float()
            ).to(torch.long)
            offsets = torch.arange(
                segment_size_feats, device=self.device
            )  # [L]
            idx = starts[:, None] + offsets[None, :]  # [B, L]
            idx_expanded = idx[:, :, None].expand(-1, -1, feats.shape[-1])
            feats = feats.gather(1, idx_expanded)  # [B, L, H]

            segment_size_sig = (
                segment_size_feats * self.hparams.generator_hop_length
            )
            starts = starts * self.hparams.generator_hop_length  # [B]
            offsets = torch.arange(
                segment_size_sig, device=self.device
            )  # [L_sig]
            idx = starts[:, None] + offsets[None, :]  # [B, L_sig]
            idx = idx.clamp(max=sig.shape[1] - 1).long()
            sig = sig.gather(1, idx)
            lens = torch.ones_like(lens)

        batch.sig = sig, lens
        batch.feats = feats, lens

    def compute_forward_generator(self, batch, stage):
        """Generator forward pass."""
        sig, lens = batch.sig

        # Forward generator
        feats, _ = batch.feats
        hyp_sig, *generator_state_ = self.modules.generator(feats)  # [B, T]
        hyp_sig = hyp_sig[:, None]  # [B, 1, T]

        # Adjust length if not matching
        sig = sig[:, None]
        if sig.shape[-1] > hyp_sig.shape[-1]:
            pad = [0, sig.shape[-1] - hyp_sig.shape[-1]]
            hyp_sig = torch.nn.functional.pad(hyp_sig, pad, mode="replicate")
        elif sig.shape[-1] < hyp_sig.shape[-1]:
            hyp_sig = hyp_sig.narrow(-1, 0, sig.shape[-1])

        batch.sig = sig, lens
        batch.hyp_sig = hyp_sig, lens  # With gradient

    def compute_forward_discriminator(
        self, batch, stage, return_discriminator=True
    ):
        """Discriminator forward pass."""
        sig, lens = batch.sig
        hyp_sig, _ = batch.hyp_sig  # With gradient

        if return_discriminator:
            # Return predictions to compute discriminator loss
            scores_fake, _ = self.modules.discriminator(hyp_sig.detach())
            scores_real, _ = self.modules.discriminator(sig)
            return scores_fake, scores_real

        # Return predictions to compute generator loss
        self.modules.discriminator.requires_grad_(False)
        scores_fake, feats_fake = self.modules.discriminator(hyp_sig)
        scores_real, feats_real = self.modules.discriminator(sig)
        self.modules.discriminator.requires_grad_()

        return hyp_sig, sig, scores_fake, feats_fake, feats_real

    def compute_objectives_generator(self, predictions, batch, stage):
        """Compute generator loss."""
        loss = self.hparams.generator_loss(
            stage,
            y_hat=predictions[0],
            y=predictions[1],
            scores_fake=predictions[2],
            feats_fake=predictions[3],
            feats_real=predictions[4],
        )
        return loss["G_loss"]

    def compute_objectives_discriminator(self, predictions, batch, stage):
        """Compute discriminator loss."""
        loss = self.hparams.discriminator_loss(
            scores_fake=predictions[0],
            scores_real=predictions[1],
        )
        return loss["D_loss"]

    def _fit_valid(self, valid_set, epoch, enable):
        """Validation stage."""
        if epoch % self.hparams.valid_freq == 0:
            return super()._fit_valid(valid_set, epoch, enable)

    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch."""
        assert stage in (sb.Stage.VALID, sb.Stage.TEST)
        self.extract_feats(batch, stage)
        self.compute_forward_generator(batch, stage)
        outputs = self.compute_forward_discriminator(
            batch, stage, return_discriminator=False
        )
        loss = self.compute_objectives_generator(outputs, batch, stage)

        IDs = batch.id
        _, lens = batch.sig
        hyp_sig, sig, *_ = outputs
        hyp_sig = hyp_sig[:, 0]
        sig = sig[:, 0]

        if (
            self.hparams.save_audios
            and self.saved_audios < 10
            and if_main_process()
        ):
            save_folder = os.path.join(
                self.hparams.output_folder,
                "audios",
                f"epoch{str(self.hparams.epoch_counter.current).zfill(4)}",
            )
            os.makedirs(save_folder, exist_ok=True)
            for i in range(len(IDs)):
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_hyp.wav"),
                    hyp_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_ref.wav"),
                    sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                self.saved_audios += 1

        if stage == sb.Stage.TEST and self.hparams.compute_metrics:
            self.utmos_metric.append(IDs, hyp_sig, lens)
            self.ref_utmos_metric.append(IDs, sig, lens)
            self.dwer_metric.append(IDs, hyp_sig, sig, lens)
            self.wavlm_sim_metric.append(IDs, hyp_sig, sig, lens)

        # Cleanup
        batch.sig = batch.hyp_sig = None

        return loss.detach().cpu()

    def init_optimizers(self):
        """Called during ``on_fit_start().``"""
        self.optimizer_generator = self.opt_class(
            self.modules.generator.parameters()
        )
        self.optimizer_discriminator = self.opt_class(
            self.modules.discriminator.parameters()
        )
        self.optimizers_dict = {
            "optimizer_generator": self.optimizer_generator,
            "optimizer_discriminator": self.optimizer_discriminator,
        }
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "optimizer_generator", self.optimizer_generator
            )
            self.checkpointer.add_recoverable(
                "optimizer_discriminator", self.optimizer_discriminator
            )

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        super().on_stage_start(stage, epoch)
        torch.backends.cudnn.benchmark = (
            stage == sb.Stage.TRAIN
            and self.hparams.segment_size is not None
            and self.hparams.segment_pad
        )
        if stage != sb.Stage.TRAIN:
            self.saved_audios = 0
        if stage == sb.Stage.TEST and self.hparams.compute_metrics:
            self.utmos_metric = self.hparams.utmos_computer()
            self.ref_utmos_metric = self.hparams.utmos_computer(
                model=self.utmos_metric.model
            )
            self.dwer_metric = self.hparams.dwer_computer()
            self.wavlm_sim_metric = self.hparams.wavlm_sim_computer()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch."""
        # Compute/store important stats
        current_epoch = self.hparams.epoch_counter.current
        stage_stats = {"loss": stage_loss}

        # Save checkpoint and anneal learning rate at the end of each epoch
        if stage == sb.Stage.TRAIN:
            self.avg_train_loss = 0.0
            self.train_stats = stage_stats
            _, lr = self.hparams.scheduler(epoch)
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer_generator, lr
            )
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer_discriminator, lr
            )
            self.stats_meta = {
                "epoch": epoch,
                "steps": self.optimizer_step,
                "lr": lr,
            }
            if if_main_process():
                self.checkpointer.save_and_keep_only(
                    meta={"loss": stage_stats["loss"], "epoch": epoch},
                    max_keys=["epoch"],
                    num_to_keep=self.hparams.keep_checkpoints,
                )
            if epoch % self.hparams.valid_freq != 0:
                self.hparams.train_logger.log_stats(
                    stats_meta=self.stats_meta,
                    train_stats=self.train_stats,
                )

        # Perform end-of-validation operations
        elif stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta=self.stats_meta,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

        elif stage == sb.Stage.TEST:
            if self.hparams.compute_metrics:
                stage_stats["UTMOS"] = self.utmos_metric.summarize("average")
                stage_stats["RefUTMOS"] = self.ref_utmos_metric.summarize(
                    "average"
                )
                stage_stats["dWER"] = self.dwer_metric.summarize("error_rate")
                stage_stats["dCER"] = self.dwer_metric.summarize(
                    "error_rate_char"
                )
                stage_stats["WavLMSim"] = self.wavlm_sim_metric.summarize(
                    "average"
                )
                if if_main_process():
                    # Save dWER
                    with open(self.hparams.dwer_file, "w") as w:
                        self.dwer_metric.write_stats(w)
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": current_epoch},
                test_stats=stage_stats,
            )


def dataio_prepare(
    data_folder,
    train_json,
    valid_json,
    test_json,
    sample_rate=16000,
    train_remove_if_longer=60.0,
    valid_remove_if_longer=60.0,
    test_remove_if_longer=60.0,
    sorting="ascending",
    debug=False,
    segment_size=None,
    segment_pad=False,
    audio_backend="soundfile",
):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    Arguments
    ---------
    data_folder : str
        Root directory containing audio files referenced by the JSON manifests.
    train_json : str
        Path to the training manifest JSON.
    valid_json : str
        Path to the validation manifest JSON.
    test_json : str
        Path to the test manifest JSON.
    sample_rate : int, optional
        Target sampling rate for loaded audio. Audio is automatically resampled
        if it does not match this rate. Default: 16000.
    train_remove_if_longer : float, optional
        Remove training examples longer than this duration (in seconds).
    valid_remove_if_longer : float, optional
        Remove validation examples longer than this duration (in seconds).
    test_remove_if_longer : float, optional
        Remove test examples longer than this duration (in seconds).
    sorting : str, optional
        Sorting strategy for dataset iteration, "ascending", "descending", or `"random"`.
        Default: "ascending".
    debug : bool, optional
        If True, load only a small subset of each dataset for faster debugging.
    segment_size : float, optional
        If provided, randomly crop each audio sample to this duration (in seconds)
        during training. Useful for training models on fixed-length segments.
    segment_pad : bool, optional
        If True, pad segments shorter than `segment_size` instead of skipping them.
    audio_backend : str, optional
        Backend to use for audio loading (e.g., "soundfile").

    Returns
    -------
    tuple
        Train data, valid data, test data.

    """
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=train_json,
        replacements={"data_root": data_folder},
    )
    # Sort training data to speed up training
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        reverse=sorting == "descending",
        key_max_value={"duration": train_remove_if_longer},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=valid_json,
        replacements={"data_root": data_folder},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=not debug,
        key_max_value={"duration": valid_remove_if_longer},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=test_json,
        replacements={"data_root": data_folder},
    )
    # Sort the test data to speed up testing
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=not debug,
        key_max_value={"duration": test_remove_if_longer},
    )

    datasets = [train_data, valid_data, test_data]

    # Define audio pipeline
    takes = ["wav"]
    provides = ["sig"]

    def audio_pipeline_train(wav):
        """Load waveform, resample, and optionally extract a random segment."""
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav, backend=audio_backend)
        sig = torchaudio.functional.resample(
            sig, original_sample_rate, sample_rate
        )

        if segment_size is not None:
            delta_length = segment_size - len(sig)
            if delta_length > 0 and segment_pad:
                sig = torch.nn.functional.pad(sig, [0, delta_length])
            elif delta_length < 0:
                start = random.randint(0, -delta_length)
                sig = sig[start : start + segment_size]

        yield sig

    def audio_pipeline_eval(wav):
        """Load waveform and resample."""
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav, backend=audio_backend)
        sig = torchaudio.functional.resample(
            sig, original_sample_rate, sample_rate
        )
        yield sig

    sb.dataio.dataset.add_dynamic_item(
        [train_data], audio_pipeline_train, takes, provides
    )
    sb.dataio.dataset.add_dynamic_item(
        [valid_data, test_data], audio_pipeline_eval, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id"] + provides)

    return train_data, valid_data, test_data


def prepare_recipe(hparams, run_opts):
    """Prepare SpeechBrain recipe.

    Arguments
    ---------
    hparams : dict
        SpeechBrain hparams dictionary loaded from the YAML recipe file.
    run_opts : dict
        SpeechBrain runtime options.

    Returns
    -------
    tuple
        Update hparams, train data, valid data, test data.

    """
    # Dataset preparation
    import libritts_prepare

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(
        libritts_prepare.prepare_libritts,
        kwargs={
            "data_folder": hparams["data_folder"],
            "train_split": hparams["train_split"],
            "valid_split": hparams["valid_split"],
            "test_split": hparams["test_split"],
            "save_json_train": hparams["train_json"],
            "save_json_valid": hparams["valid_json"],
            "save_json_test": hparams["test_json"],
            "sample_rate": hparams["sample_rate"],
            "skip_prep": hparams["skip_prep"],
            "model_name": "HiFi-GAN",
        },
    )

    # Create the datasets objects
    train_data, valid_data, test_data = dataio_prepare(
        hparams["data_folder"],
        hparams["train_json"],
        hparams["valid_json"],
        hparams["test_json"],
        hparams["sample_rate"],
        hparams["train_remove_if_longer"],
        hparams["valid_remove_if_longer"],
        hparams["test_remove_if_longer"],
        hparams["sorting"],
        run_opts["debug"],
        hparams["segment_size"],
        hparams["segment_pad"],
        hparams["audio_backend"],
    )

    # Dynamic batching
    train_dataloader_kwargs = {
        "num_workers": hparams.get("dataloader_workers", 0)
    }
    if hparams.get("dynamic_batching", False) or hparams.get(
        "train_dynamic_batching", False
    ):
        train_dataloader_kwargs["batch_sampler"] = DynamicBatchSampler(
            train_data,
            hparams["train_max_batch_length"],
            num_buckets=hparams.get("num_buckets"),
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering=hparams.get("sorting", "batch_ordering"),
            max_batch_ex=hparams.get("max_batch_size"),
            bucket_boundaries=hparams.get("bucket_boundaries", []),
            lengths_list=hparams.get("lengths_list"),
        )
    else:
        train_dataloader_kwargs["batch_size"] = hparams["train_batch_size"]
        train_dataloader_kwargs["shuffle"] = hparams["sorting"] == "random"
        train_dataloader_kwargs["pin_memory"] = run_opts["device"] != "cpu"
        train_dataloader_kwargs["drop_last"] = hparams.get(
            "segment_size", None
        ) is not None and hparams.get("segment_pad", False)
    hparams["train_dataloader_kwargs"] = train_dataloader_kwargs

    valid_dataloader_kwargs = {
        "num_workers": hparams.get("dataloader_workers", 0)
    }
    if hparams.get("dynamic_batching", False) or hparams.get(
        "valid_dynamic_batching", False
    ):
        valid_dataloader_kwargs["batch_sampler"] = DynamicBatchSampler(
            valid_data,
            hparams["valid_max_batch_length"],
            num_buckets=hparams.get("num_buckets"),
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering="descending",
            max_batch_ex=hparams.get("max_batch_size"),
            bucket_boundaries=hparams.get("bucket_boundaries", []),
            lengths_list=hparams.get("lengths_list"),
        )
    else:
        valid_dataloader_kwargs["batch_size"] = hparams["valid_batch_size"]
        valid_dataloader_kwargs["pin_memory"] = run_opts["device"] != "cpu"
    hparams["valid_dataloader_kwargs"] = valid_dataloader_kwargs

    test_dataloader_kwargs = {
        "num_workers": hparams.get("dataloader_workers", 0)
    }
    if hparams.get("dynamic_batching", False) or hparams.get(
        "test_dynamic_batching", False
    ):
        test_dataloader_kwargs["batch_sampler"] = DynamicBatchSampler(
            test_data,
            hparams["test_max_batch_length"],
            num_buckets=hparams.get("num_buckets"),
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering="descending",
            max_batch_ex=hparams.get("max_batch_size"),
            bucket_boundaries=hparams.get("bucket_boundaries", []),
            lengths_list=hparams.get("lengths_list"),
        )
    else:
        test_dataloader_kwargs["batch_size"] = hparams["test_batch_size"]
        test_dataloader_kwargs["pin_memory"] = run_opts["device"] != "cpu"
    hparams["test_dataloader_kwargs"] = test_dataloader_kwargs

    # Pretrain the specified modules
    if "pretrainer" in hparams:
        run_on_main(hparams["pretrainer"].collect_files)
        run_on_main(hparams["pretrainer"].load_collected)

    return hparams, train_data, valid_data, test_data


if __name__ == "__main__":
    # Command-line interface
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then create ddp_init_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare recipe
    hparams, train_data, valid_data, test_data = prepare_recipe(
        hparams, run_opts
    )

    # Trainer initialization
    brain = Resynthesis(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Train
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_kwargs"],
        valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
    )

    # Test
    brain.hparams.dwer_file = os.path.join(hparams["output_folder"], "dwer.txt")
    brain.evaluate(
        test_data,
        max_key="epoch",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
