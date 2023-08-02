#!/usr/bin/env python3
"""
Authors
 * Yingzhi WANG 2023
"""

import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import torch
import json
import itertools
from speechbrain.utils.EDER import EDER


class EmoDiaBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier.
        """
        batch = batch.to(self.device)

        self.modules = self.modules.to(self.device)

        wavs, lens = batch.sig
        wavs = self.hparams.input_norm(wavs, lens)
        outputs = self.modules.wav2vec2(wavs)
        averaged_out = self.hparams.avg_pool(outputs)

        outputs = self.modules.output_mlp(averaged_out)

        outputs = self.hparams.log_softmax(outputs)

        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        emoid, _ = batch.emo_encoded

        if stage == sb.Stage.TEST:
            if self.hparams.use_threshold:
                preds = threshold_tuning(predictions, self.hparams.threshold)
            else:
                preds = torch.argmax(predictions, dim=2)

            emoid_decoded = label_encoder.decode_ndim(emoid)
            preds_decoded = label_encoder.decode_ndim(preds)

            self.load_ZED()
            with open(self.hparams.eder_file, "a") as w:
                for i in range(len(batch.id)):
                    if len(preds_decoded[i]) < len(emoid_decoded[i]):
                        preds_decoded[i].append(preds_decoded[i][-1])
                    eder = EDER(
                        prediction=preds_decoded[i],
                        id=batch.id[i],
                        duration=self.ZED[batch.id[i]]["duration"],
                        emotion=self.ZED[batch.id[i]]["emotion"],
                        window_length=self.hparams.window_length * 0.02,
                        stride=self.hparams.stride * 0.02,
                    )

                    w.write("    wav_id : " + batch.id[i] + "\n")
                    w.write(" reference : " + "".join(emoid_decoded[i]) + "\n")
                    w.write("prediction : " + "".join(preds_decoded[i]) + "\n")
                    w.write(
                        " ctc_label : "
                        + "".join(del_adjacent(emoid_decoded[i]))
                        + "\n"
                    )
                    w.write(
                        "  ctc_pred : "
                        + "".join(del_adjacent(preds_decoded[i]))
                        + "\n"
                    )
                    w.write("      EDER : " + str(eder) + "\n")
                    w.write("\n")

                    self.eder.append(eder)

        loss = self.hparams.compute_cost(predictions, emoid)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emoid)

        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec2_optimizer.step()
            self.optimizer.step()

        self.wav2vec2_optimizer.zero_grad()
        self.optimizer.zero_grad()

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )
        self.eder = []
        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "wave2vec_lr": old_lr_wav2vec2},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={
                    "loss": stats["loss"],
                    "error_rate": stats["error_rate"],
                    "EDER": sum(self.eder) / len(self.eder),
                },
            )
            # with open(self.hparams.cer_file, "a") as w:
            #     self.error_metrics.write_stats(w)

    def load_ZED(self):
        with open(self.hparams.test_annotation, "r") as f:
            ZED_data = json.load(f)
        self.ZED = ZED_data

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """
    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    # Define label pipeline:
    @sb.utils.data_pipeline.takes("frame_label")
    @sb.utils.data_pipeline.provides("emo_encoded")
    def label_pipeline(frame_label):
        emo_encoded = label_encoder.encode_sequence_torch(frame_label)
        yield emo_encoded

    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["zed_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "emo_encoded"],
        )

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")

    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="frame_label",
        sequence_input=True,
    )

    return datasets, label_encoder


def threshold_tuning(batch_predictions, threshold):
    """Post processing by finding a threshold

    Args:
        predictions torch.Tensor: (b, t, 4)
    """
    argmax_preds = torch.argmax(batch_predictions, dim=2)
    max, _ = torch.max(batch_predictions, dim=2)
    index = torch.gt(max, threshold)
    return torch.mul(index, argmax_preds)


def del_adjacent(list):
    """delete adjacent elements that is the same as the f
    """
    return [k for k, g in itertools.groupby(list)]


# RECIPE BEGINS!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        from zed_prepare import prepare_train, prepare_test

        sb.utils.distributed.run_on_main(
            prepare_train,
            kwargs={
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "split_ratio": hparams["split_ratio"],
                "win_len": hparams["window_length"] * 0.02,
                "stride": hparams["stride"] * 0.02,
                "seed": hparams["seed"],
                "emovdb_folder": hparams["emovdb_folder"],
                "esd_folder": hparams["esd_folder"],
                "iemocap_folder": hparams["iemocap_folder"],
                "jlcorpus_folder": hparams["jlcorpus_folder"],
                "ravdess_folder": hparams["ravdess_folder"],
            },
        )
        sb.utils.distributed.run_on_main(
            prepare_test,
            kwargs={
                "ZED_folder": hparams["zed_folder"],
                "save_json_test": hparams["test_annotation"],
                "win_len": hparams["window_length"] * 0.02,
                "stride": hparams["stride"] * 0.02,
            },
        )
    # Create dataset objects "train", "valid", and "test".
    datasets, label_encoder = dataio_prep(hparams)

    hparams["wav2vec2"] = hparams["wav2vec2"].to(run_opts["device"])

    # freeze the feature extractor part when unfreezing
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()

    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = EmoDiaBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
