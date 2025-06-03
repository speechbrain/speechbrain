#!/usr/bin/python3

"""Recipe to train a classifier on ESC50 data.

To run this recipe, use the following command:
> python train.py hparams/<config>.yaml --data_folder yourpath/ESC-50-master

Authors
    * Cem Subakan 2022, 2023
    * Francesco Paissan 2022, 2023
    * Luca Della Libera 2024

Based on the Urban8k recipe by
    * David Whipps 2021
    * Ala Eddine Limame 2021
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision
from confusion_matrix_fig import create_cm_fig
from esc50_prepare import prepare_esc50
from hyperpyyaml import load_hyperpyyaml
from sklearn.metrics import confusion_matrix
from wham_prepare import combine_batches, prepare_wham

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main


class ESC50Brain(sb.core.Brain):
    """Class for classifier training."""

    def compute_forward(self, batch, stage):
        """Computation pipeline based on an encoder + sound classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Augment if specified
        if hasattr(self.hparams, "augmentation") and stage == sb.Stage.TRAIN:
            wavs, lens = self.hparams.augmentation(wavs, lens)

        # augment batch with WHAM!
        if hasattr(self.hparams, "add_wham_noise"):
            if self.hparams.add_wham_noise:
                wavs = combine_batches(wavs, iter(self.hparams.wham_dataset))

        X_stft = self.modules.compute_stft(wavs)
        net_input = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        if (
            hasattr(self.hparams, "use_melspectra")
            and self.hparams.use_melspectra
        ):
            net_input = self.modules.compute_fbank(net_input)

        if (not self.hparams.use_melspectra) or self.hparams.use_log1p_mel:
            net_input = torch.log1p(net_input)

        # Embeddings + sound classifier
        if hasattr(self.modules.embedding_model, "config"):
            # Hugging Face model
            config = self.modules.embedding_model.config
            # Resize to match expected resolution
            net_input = torchvision.transforms.functional.resize(
                net_input, (config.image_size, config.image_size)
            )
            # Expand to have 3 channels
            net_input = net_input[:, None, ...].expand(-1, 3, -1, -1)
            if config.model_type == "focalnet":
                embeddings = self.modules.embedding_model(
                    net_input
                ).feature_maps[-1]
                embeddings = embeddings.mean(dim=(-1, -2))
            elif config.model_type == "vit":
                embeddings = self.modules.embedding_model(
                    net_input
                ).last_hidden_state.movedim(-1, -2)
                embeddings = embeddings.mean(dim=-1)
            else:
                raise NotImplementedError
        else:
            # SpeechBrain model
            embeddings = self.modules.embedding_model(net_input)
            if isinstance(embeddings, tuple):
                embeddings, _ = embeddings

            if embeddings.ndim == 4:
                embeddings = embeddings.mean((-1, -2))

        # run through classifier
        outputs = self.modules.classifier(embeddings)

        if outputs.ndim == 2:
            outputs = outputs.unsqueeze(1)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using class-id as label."""
        predictions, lens = predictions
        uttid = batch.id
        classid, _ = batch.class_string_encoded

        # Target augmentation
        N_augments = int(predictions.shape[0] / classid.shape[0])
        classid = torch.cat(N_augments * [classid], dim=0)

        # loss = self.hparams.compute_cost(predictions.squeeze(1), classid, lens)
        target = F.one_hot(
            classid.squeeze(), num_classes=self.hparams.out_n_neurons
        )
        loss = (
            -(F.log_softmax(predictions.squeeze(1), 1) * target).sum(1).mean()
        )

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        # Append this batch of losses to the loss metric
        self.loss_metric.append(
            uttid, predictions, classid, lens, reduction="batch"
        )

        # Confusion matrices
        if stage != sb.Stage.TRAIN:
            y_true = classid.cpu().detach().numpy().squeeze(-1)
            y_pred = predictions.cpu().detach().numpy().argmax(-1).squeeze(-1)

        if stage == sb.Stage.VALID:
            confusion_matix = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
            )
            self.valid_confusion_matrix += confusion_matix
        if stage == sb.Stage.TEST:
            confusion_matix = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
            )
            self.test_confusion_matrix += confusion_matix

        # Compute accuracy using MetricStats
        self.acc_metric.append(
            uttid, predict=predictions, target=classid, lengths=lens
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, classid, lens)

        return loss

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

        # Compute accuracy using MetricStats
        # Define function taking (prediction, target, length) for eval
        def accuracy_value(predict, target, lengths):
            """Computes accuracy."""
            nbr_correct, nbr_total = sb.utils.Accuracy.Accuracy(
                predict, target, lengths
            )
            acc = torch.tensor([nbr_correct / nbr_total])
            return acc

        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )

        # Confusion matrices
        if stage == sb.Stage.VALID:
            self.valid_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons, self.hparams.out_n_neurons),
                dtype=int,
            )
        if stage == sb.Stage.TEST:
            self.test_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons, self.hparams.out_n_neurons),
                dtype=int,
            )

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
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {
                "loss": self.train_loss,
                "acc": self.acc_metric.summarize("average"),
            }
        # Summarize Valid statistics from the stage for record-keeping
        elif stage == sb.Stage.VALID:
            valid_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "error": self.error_metrics.summarize("average"),
            }
        # Summarize Test statistics from the stage for record-keeping
        else:
            test_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "error": self.error_metrics.summarize("average"),
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # Tensorboard logging
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(
                    stats_meta={"Epoch": epoch},
                    train_stats=self.train_stats,
                    valid_stats=valid_stats,
                )
                # Log confusion matrix fig to tensorboard
                cm_fig = create_cm_fig(
                    self.valid_confusion_matrix,
                    display_labels=list(
                        self.hparams.label_encoder.ind2lab.values()
                    ),
                )
                self.hparams.tensorboard_train_logger.writer.add_figure(
                    "Validation Confusion Matrix", cm_fig, epoch
                )

            # The train_logger writes a summary to stdout and to the log file
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=valid_stats, min_keys=["error"]
            )

        # We also write statistics about test data to stdout and to the log file
        if stage == sb.Stage.TEST:
            # Per class accuracy from Test confusion matrix
            per_class_acc_arr = np.diag(self.test_confusion_matrix) / np.sum(
                self.test_confusion_matrix, axis=1
            )
            per_class_acc_arr_str = "\n" + "\n".join(
                "{:}: {:.3f}".format(class_id, class_acc)
                for class_id, class_acc in enumerate(per_class_acc_arr)
            )

            self.hparams.train_logger.log_stats(
                {
                    "Epoch loaded": self.hparams.epoch_counter.current,
                    "\n Per Class Accuracy": per_class_acc_arr_str,
                    "\n Confusion Matrix": "\n{:}\n".format(
                        self.test_confusion_matrix
                    ),
                },
                test_stats=test_stats,
            )


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""
    data_audio_folder = hparams["audio_data_folder"]
    config_sample_rate = hparams["sample_rate"]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        wave_file = data_audio_folder + "/{:}".format(wav)

        sig, read_sr = torchaudio.load(wave_file)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        sig = sig.float()
        sig = sig / sig.max()
        return sig

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_string")
    @sb.utils.data_pipeline.provides("class_string", "class_string_encoded")
    def label_pipeline(class_string):
        """The label pipeline."""
        yield class_string
        class_string_encoded = label_encoder.encode_label_torch(class_string)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "class_string_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="class_string",
    )

    return datasets, label_encoder


if __name__ == "__main__":
    # This flag enables the built-in cuDNN auto-tuner
    # torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

    run_on_main(
        prepare_esc50,
        kwargs={
            "data_folder": hparams["data_folder"],
            "audio_data_folder": hparams["audio_data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "train_fold_nums": hparams["train_fold_nums"],
            "valid_fold_nums": hparams["valid_fold_nums"],
            "test_fold_nums": hparams["test_fold_nums"],
            "skip_manifest_creation": hparams["skip_manifest_creation"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder

    if "wham_folder" in hparams:
        hparams["wham_dataset"] = prepare_wham(
            hparams["wham_folder"],
            hparams["add_wham_noise"],
            hparams["sample_rate"],
            hparams["signal_length_s"],
            hparams["wham_audio_folder"],
        )

    if hparams["wham_dataset"] is not None:
        assert hparams["signal_length_s"] == 5, "Fix wham sig length!"

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    ESC50_brain = ESC50Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Load pretrained encoder if it exists in the yaml file
    if not hasattr(ESC50_brain.modules, "embedding_model"):
        ESC50_brain.hparams.embedding_model.to(ESC50_brain.device)

    if "pretrained_encoder" in hparams and hparams["use_pretrained"]:
        run_on_main(hparams["pretrained_encoder"].collect_files)
        hparams["pretrained_encoder"].load_collected()

    if not hparams["test_only"]:
        ESC50_brain.fit(
            epoch_counter=ESC50_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )

    # Load the best checkpoint for evaluation
    test_stats = ESC50_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
    )
