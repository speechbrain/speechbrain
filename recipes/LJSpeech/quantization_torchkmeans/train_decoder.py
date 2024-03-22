#!/usr/bin/env/python

"""Recipe for training a decoder from continuous audio representations to their corresponding waveform.

Currently, it supports only HiFi-GAN decoder.

To run this recipe:
> python train_decoder.py hparams/decoder/<config>.yaml

Authors
 * Luca Della Libera 2024
"""

import csv
import os
import sys

import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import write_audio
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.utils.distributed import if_main_process, run_on_main


class Generation(sb.Brain):
    def fit_batch(self, batch):
        amp = sb.core.AMPConfig.from_name(self.precision)

        # Train discriminator
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=amp.dtype,
            enabled=self.use_amp,
        ):
            self.compute_forward_common(batch, sb.Stage.TRAIN)
            self.compute_forward_generator(batch, sb.Stage.TRAIN)
            outputs = self.compute_forward_discriminator(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives_discriminator(
                outputs, batch, sb.Stage.TRAIN
            )["D_loss"]

        scaled_loss = self.scaler.scale(loss)
        self.check_loss_isfinite(scaled_loss)
        scaled_loss.backward()
        self.optimizers_step()

        # Train generator
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=amp.dtype,
            enabled=self.use_amp,
        ):
            outputs = self.compute_forward_discriminator(
                batch, sb.Stage.TRAIN, return_discriminator=False
            )
            loss = self.compute_objectives_generator(
                outputs, batch, sb.Stage.TRAIN
            )["G_loss"]

        scaled_loss = self.scaler.scale(loss)
        self.check_loss_isfinite(scaled_loss)
        scaled_loss.backward()
        self.optimizers_step()

        self.on_fit_batch_end(batch, outputs, loss, should_step=True)

        # Generator loss
        return loss.detach().cpu()

    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        self.compute_forward_common(batch, sb.Stage.TRAIN)
        self.compute_forward_generator(batch, sb.Stage.TRAIN)
        outputs = self.compute_forward_discriminator(
            batch, sb.Stage.TRAIN, return_discriminator=False
        )
        loss = self.compute_objectives_generator(
            outputs, batch, sb.Stage.TRAIN
        )["G_loss"]

        if stage == sb.Stage.TEST and self.vocode:
            sig_pred, sig, *_ = outputs
            self.IDs += batch.id
            self.sigs_pred += sig_pred.cpu()
            self.sigs += sig.cpu()

        # Generator loss
        return loss.detach().cpu()

    def compute_forward_common(self, batch, stage):
        batch = batch.to(self.device)
        sig, lens = batch.sig

        # Augment if specified
        if stage == sb.Stage.TRAIN and self.hparams.augment:
            sig, lens = self.hparams.augmentation(sig, lens)
            batch.sig = sig, lens

        # Extract audio tokens
        with torch.no_grad():
            self.modules.codec.encoder.eval()
            self.modules.codec.quantizer.eval()
            self.modules.codec.dequantizer.eval()
            _, discrete_feats = self.modules.codec.encode_discrete(sig, lens)
            feats_pred = self.modules.codec.dequantize(discrete_feats, lens)

        batch.feats_pred = feats_pred, lens

    def compute_forward_generator(self, batch, stage):
        sig, lens = batch.sig
        feats_pred, _ = batch.feats_pred

        # Forward decoder/generator
        sig_pred = self.modules.codec.decode(feats_pred)
        min_length = min(sig.shape[-1], sig_pred.shape[-1])
        sig = sig[:, :min_length]
        sig_pred = sig_pred[:, :min_length]
        sig_pred_sig = torch.cat([sig_pred, sig])[:, None]

        batch.sig = sig, lens
        batch.sig_pred = sig_pred, lens  # With gradient
        batch.sig_pred_sig = sig_pred_sig, None  # With gradient

    def compute_forward_discriminator(
        self, batch, stage, return_discriminator=True
    ):
        sig, lens = batch.sig
        sig_pred, _ = batch.sig_pred  # With gradient
        sig_pred_sig, _ = batch.sig_pred_sig  # With gradient

        if return_discriminator:
            # Return predictions to compute discriminator loss
            scores_fake_real, _ = self.modules.discriminator(
                sig_pred_sig.detach()
            )
            scores_fake, scores_real = [], []
            for x in scores_fake_real:
                scores_fake.append(x[: len(sig)])
                scores_real.append(x[len(sig) :])
            return scores_fake, scores_real

        # Return predictions to compute generator loss
        self.modules.discriminator.requires_grad_(False)
        scores_fake_real, feats_fake_real = self.modules.discriminator(
            sig_pred_sig
        )
        self.modules.discriminator.requires_grad_()
        scores_fake = [x[: len(sig)] for x in scores_fake_real]
        feats_fake, feats_real = [], []
        for x in feats_fake_real:
            fake_list, real_list = [], []
            for y in x:
                fake_list.append(y[: len(sig)])
                real_list.append(y[len(sig) :])
            feats_fake.append(fake_list)
            feats_real.append(real_list)

        return sig_pred, sig, scores_fake, feats_fake, feats_real

    def compute_objectives_generator(self, predictions, batch, stage):
        loss = self.hparams.generator_loss(
            stage,
            y_hat=predictions[0],
            y=predictions[1],
            scores_fake=predictions[2],
            feats_fake=predictions[3],
            feats_real=predictions[4],
        )
        return loss

    def compute_objectives_discriminator(self, predictions, batch, stage):
        loss = self.hparams.discriminator_loss(
            scores_fake=predictions[0], scores_real=predictions[1],
        )
        return loss

    def evaluate(self, test_set, *args, **kwargs):
        """Evaluation loop."""
        if self.vocode:
            self.IDs = []
            self.sigs_pred = []
            self.sigs = []
        return super().evaluate(test_set, *args, **kwargs)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of each epoch."""
        # Compute/store important stats
        current_epoch = self.hparams.epoch_counter.current
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration operations, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            _, lr = self.hparams.scheduler(current_epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, lr)
            steps = self.optimizer_step
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr, "steps": steps},
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
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": current_epoch},
                test_stats=stage_stats,
            )
            # Vocode
            if self.hparams.vocode:
                self.vocode()

    def vocode(self):
        from speechbrain.nnet.loss.si_snr_loss import si_snr_loss
        from tqdm import tqdm

        from metrics.dnsmos import DNSMOS
        from metrics.dwer import DWER
        from metrics.spk_sim import SpkSim

        metrics = []
        for ID, sig_pred, sig in tqdm(
            zip(self.IDs, self.sigs_pred, self.sigs),
            dynamic_ncols=True,
            total=len(self.IDs),
        ):
            if self.hparams.save_audios:
                save_folder = os.path.join(self.hparams.output_folder, "audios")
                os.makedirs(save_folder, exist_ok=True)
                write_audio(
                    os.path.join(save_folder, f"{ID}_pred.wav"),
                    sig_pred.cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{ID}_ref.wav"),
                    sig.cpu(),
                    self.hparams.sample_rate,
                )

            # Compute metrics
            sig_pred = sig_pred.to(self.device)
            sig = sig.to(self.device)
            sisnr = -si_snr_loss(
                sig_pred[None],
                sig[None],
                torch.as_tensor([1.0], device=self.device),
            ).item()
            spk_sim = SpkSim(sig_pred, sig, self.hparams.sample_rate)
            dnsmos = DNSMOS(sig_pred, self.hparams.sample_rate)
            ref_dnsmos = DNSMOS(sig, self.hparams.sample_rate)
            dwer, text, ref_text = DWER(sig_pred, sig, self.hparams.sample_rate)
            metrics.append(
                [
                    ID,
                    sisnr,
                    spk_sim,
                    *dnsmos,
                    *ref_dnsmos,
                    dwer,
                    text,
                    ref_text,
                ]
            )

        headers = ["ID", "SI-SNR", "SpkSim"]
        headers += ["SigMOS", "BakMOS", "OvrMOS", "p808MOS"]
        headers += ["RefSigMOS", "RefBakMOS", "RefOvrMOS", "Refp808MOS"]
        headers += ["dWER", "text", "ref_text"]
        with open(
            os.path.join(self.hparams.output_folder, "metrics.csv"),
            "w",
            encoding="utf-8",
        ) as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            csv_writer.writeheader()

            for entry in metrics:
                entry = dict(zip(headers, entry))
                csv_writer.writerow(entry)

            columns = list(zip(*metrics))
            entry = dict(
                zip(
                    headers,
                    ["Average"]
                    + [sum(c) / len(c) for c in columns[1:-2]]
                    + ["", ""],
                )
            )
            csv_writer.writerow(entry)
            self.hparams.train_logger.log_stats(
                stats_meta={k: v for k, v in list(entry.items())[1:-2]},
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
    **_,
):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=train_json, replacements={"DATA_ROOT": data_folder},
    )
    # Sort training data to speed up training
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        reverse=sorting == "descending",
        key_max_value={"duration": train_remove_if_longer},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=valid_json, replacements={"DATA_ROOT": data_folder},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=not debug,
        key_max_value={"duration": valid_remove_if_longer},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=test_json, replacements={"DATA_ROOT": data_folder},
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

    def audio_pipeline(wav):
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav)

        sig = torchaudio.functional.resample(
            sig, original_sample_rate, sample_rate,
        )
        yield sig

    sb.dataio.dataset.add_dynamic_item(
        [train_data, valid_data, test_data], audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id"] + provides)

    return train_data, valid_data, test_data


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

    # Dataset preparation
    from ljspeech_prepare import prepare_ljspeech as prepare_data

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(
        prepare_data,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
        },
    )

    # Create the datasets objects and tokenization
    train_data, valid_data, test_data = dataio_prepare(
        debug=run_opts.get("debug", False), **hparams
    )

    # Pretrain the specified modules
    run_on_main(hparams["pretrainer"].collect_files)
    run_on_main(hparams["pretrainer"].load_collected)

    # Trainer initialization
    brain = Generation(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Dynamic batching
    hparams["train_dataloader_kwargs"] = {
        "num_workers": hparams["dataloader_workers"]
    }
    if hparams["dynamic_batching"]:
        hparams["train_dataloader_kwargs"][
            "batch_sampler"
        ] = DynamicBatchSampler(
            train_data,
            hparams["train_max_batch_length"],
            num_buckets=hparams["num_buckets"],
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering=hparams["sorting"],
            max_batch_ex=hparams["max_batch_size"],
        )
    else:
        hparams["train_dataloader_kwargs"]["batch_size"] = hparams[
            "train_batch_size"
        ]
        hparams["train_dataloader_kwargs"]["shuffle"] = (
            hparams["sorting"] == "random"
        )

    hparams["valid_dataloader_kwargs"] = {
        "num_workers": hparams["dataloader_workers"]
    }
    if hparams["dynamic_batching"]:
        hparams["valid_dataloader_kwargs"][
            "batch_sampler"
        ] = DynamicBatchSampler(
            valid_data,
            hparams["valid_max_batch_length"],
            num_buckets=hparams["num_buckets"],
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering="descending",
            max_batch_ex=hparams["max_batch_size"],
        )
    else:
        hparams["valid_dataloader_kwargs"]["batch_size"] = hparams[
            "valid_batch_size"
        ]

    hparams["test_dataloader_kwargs"] = {
        "num_workers": hparams["dataloader_workers"]
    }
    if hparams["dynamic_batching"]:
        hparams["test_dataloader_kwargs"][
            "batch_sampler"
        ] = DynamicBatchSampler(
            test_data,
            hparams["test_max_batch_length"],
            num_buckets=hparams["num_buckets"],
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering="descending",
            max_batch_ex=hparams["max_batch_size"],
        )
    else:
        hparams["test_dataloader_kwargs"]["batch_size"] = hparams[
            "test_batch_size"
        ]

    # Train
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_kwargs"],
        valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
    )

    # Test
    brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
