#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
"""
import os
import sys
import torch
import speechbrain as sb


class XvectorBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, x, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        ids, wavs, lens = x
        wavs, lens = wavs.to(self.device), lens.to(self.device)

        if stage == sb.Stage.TRAIN:
            # Addding noise and reverberation
            wavs_aug = self.modules.env_corrupt(wavs, lens)

            # Adding time-domain augmentation
            wavs_aug = self.modules.augmentation(wavs_aug, lens)

            # Concatenate noisy and clean batches
            wavs = torch.cat([wavs, wavs_aug], dim=0)
            lens = torch.cat([lens, lens], dim=0)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid, spkid, _ = targets

        spkid, lens = spkid.to(self.device), lens.to(self.device)

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            spkid = torch.cat([spkid, spkid], dim=0)

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            if self.root_process:
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch, "lr": old_lr},
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )
                self.checkpointer.save_and_keep_only(
                    meta={"ErrorRate": stage_stats["ErrorRate"]},
                    min_keys=["ErrorRate"],
                )


if __name__ == "__main__":

    # This flag enable the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # This hack needed to import data preparation script from ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from voxceleb_prepare import prepare_voxceleb  # noqa E402

    # Load hyperparameters file with command-line overrides
    hparams_file, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.yaml.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    prepare_voxceleb(
        data_folder=hparams["data_folder"],
        save_folder=hparams["save_folder"],
        splits=["train", "dev"],
        split_ratio=[90, 10],
        seg_dur=300,
        rand_seed=hparams["seed"],
        random_segment=hparams["random_segment"],
    )

    # Data loaders
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()

    # Brain class initialization
    xvect_brain = XvectorBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    xvect_brain.fit(xvect_brain.hparams.epoch_counter, train_set, valid_set)
