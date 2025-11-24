#!/usr/bin/env/python

"""Recipe for training a quantizer on continuous audio representations.

To run this recipe:
> python train_quantizer.py hparams/<path-to-config>.yaml

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


class Quantization(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass."""
        batch = batch.to(self.device)
        sig, lens = batch.sig

        # Augment if specified
        if stage == sb.Stage.TRAIN and self.hparams.augment:
            sig, lens = self.hparams.augmentation(sig, lens)

        # Extract features
        with torch.no_grad():
            self.hparams.encoder.to(self.device).eval()
            feats, *encoder_state_ = self.hparams.encoder(sig, length=lens)

        # Forward model
        lats, *compressor_state_ = self.modules.compressor(feats)
        codes, toks, aux_loss = self.modules.quantizer(lats)
        hyp_feats, *decompressor_state_ = self.modules.decompressor(codes)

        return hyp_feats, feats, aux_loss

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        hyp_feats, feats, aux_loss = predictions

        _, lens = batch.sig

        # Reconstruction loss
        loss = self.hparams.rec_loss(hyp_feats, feats, length=lens)
        loss += aux_loss

        return loss

    def _fit_valid(self, valid_set, epoch, enable):
        """Validation stage."""
        if epoch % self.hparams.valid_freq == 0:
            return super()._fit_valid(valid_set, epoch, enable)

    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch."""
        assert stage in (sb.Stage.VALID, sb.Stage.TEST)
        outputs = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(outputs, batch, stage=stage)
        hyp_feats, feats, _ = outputs

        IDs = batch.id
        sig, lens = batch.sig

        self.hparams.decoder.to(self.device).eval()
        hyp_sig, *decoder_state_ = self.hparams.decoder(hyp_feats)
        rec_sig, *decoder_state_ = self.hparams.decoder(feats)

        # Adjust length if not matching
        if sig.shape[-1] > hyp_sig.shape[-1]:
            pad = [0, sig.shape[-1] - hyp_sig.shape[-1]]
            hyp_sig = torch.nn.functional.pad(hyp_sig, pad, mode="replicate")
            rec_sig = torch.nn.functional.pad(rec_sig, pad, mode="replicate")
        elif sig.shape[-1] < hyp_sig.shape[-1]:
            hyp_sig = hyp_sig.narrow(-1, 0, sig.shape[-1])
            rec_sig = rec_sig.narrow(-1, 0, sig.shape[-1])

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
                    os.path.join(save_folder, f"{IDs[i]}_rec.wav"),
                    rec_sig[i].cpu(),
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

        return loss.detach().cpu()

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
        current_epoch = self.hparams.epoch_counter.current
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            self.stats_meta = {"epoch": epoch, "steps": self.optimizer_step}
            if epoch % self.hparams.valid_freq != 0:
                self.hparams.train_logger.log_stats(
                    stats_meta=self.stats_meta,
                    train_stats=self.train_stats,
                )

        # Perform end-of-iteration operations, like annealing, logging, etc.
        elif stage == sb.Stage.VALID:
            _, lr = self.hparams.scheduler(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, lr)
            self.stats_meta["lr"] = lr
            self.hparams.train_logger.log_stats(
                stats_meta=self.stats_meta,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if if_main_process():
                self.checkpointer.save_and_keep_only(
                    meta={"loss": stage_stats["loss"]},
                    min_keys=["loss"],
                    num_to_keep=self.hparams.keep_checkpoints,
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
    brain = Quantization(
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
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
