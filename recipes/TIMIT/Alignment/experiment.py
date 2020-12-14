#!/usr/bin/env python3
"""Recipe for doing HMM-DNN Alignment on the TIMIT dataset

To run this recipe, do the following:
> python experiment.py hyperparams.yaml --data_folder /path/to/TIMIT

Authors
 * Elena Rastorgueve 2020
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""
import os
import sys
import torch
import speechbrain as sb


# Define training procedure
class ASR_Brain(sb.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_type = hparams["init_training_type"]
        print("Starting training type:", self.training_type)

    def compute_forward(self, x, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Adding augmentation when specified:
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        if hasattr(self.hparams, "normalize"):
            feats = self.modules.normalize(feats, wav_lens)
        out = self.modules.model(feats)
        out = self.modules.output(out)
        out = out - out.mean(1).unsqueeze(1)
        pout = self.hparams.log_softmax(out)
        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage):
        """Computes the loss with the specified alignement algorithm"""
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets[0]
        _, ends, end_lens = targets[1]

        if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)
        phns_orig = sb.utils.data_utils.undo_padding(phns, phn_lens)
        phns = self.hparams.aligner.expand_phns_by_states_per_phoneme(
            phns, phn_lens
        )

        if self.training_type == "forward":
            forward_scores = self.hparams.aligner(
                pout, pout_lens, phns, phn_lens, "forward"
            )
            loss = -forward_scores

        elif self.training_type == "ctc":
            loss = self.hparams.compute_cost_ctc(
                pout, phns, pout_lens, phn_lens
            )
        elif self.training_type == "viterbi":
            prev_alignments = self.hparams.aligner.get_prev_alignments(
                ids, pout, pout_lens, phns, phn_lens
            )
            prev_alignments = prev_alignments.to(self.hparams.device)
            loss = self.hparams.compute_cost_nll(pout, prev_alignments)

        viterbi_scores, alignments = self.hparams.aligner(
            pout, pout_lens, phns, phn_lens, "viterbi"
        )

        if self.training_type in ["viterbi", "forward"]:
            self.hparams.aligner.store_alignments(ids, alignments)

        if stage != sb.Stage.TRAIN:
            self.accuracy_metrics.append(ids, alignments, ends, phns_orig)

        return loss

    def fit_batch(self, batch):
        """
        Modify slightly from original version as batch returns ends as well.
        Only first 2 lines are modified
        """
        inputs, phns, ends = batch
        targets = phns, ends
        predictions = self.compute_forward(inputs, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, targets, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """
        Modify slightly from original version as batch returns ends as well.
        Only first 2 lines are modified.
        """
        inputs, phns, ends = batch
        targets = phns, ends
        out = self.compute_forward(inputs, stage=stage)
        loss = self.compute_objectives(out, targets, stage=stage)
        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.accuracy_metrics = self.hparams.accuracy_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if hasattr(self.hparams, "switch_training_type"):
            if not hasattr(self.hparams, "switch_training_epoch"):
                raise ValueError(
                    "Please specify `switch_training_epoch` in `params`"
                )
            if (
                self.hparams.epoch_counter.current + 1
                == self.hparams.switch_training_epoch
            ):
                self.training_type = self.hparams.switch_training_type
                print("Switching to training type", self.training_type)

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            acc = self.accuracy_metrics.summarize("average")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(acc)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "accuracy": acc},
            )
            self.checkpointer.save_and_keep_only(
                meta={"accuracy": acc}, min_keys=["accuracy"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "accuracy": acc},
            )


# Begin Recipe!
if __name__ == "__main__":

    # This hack needed to import data preparation script from ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from timit_prepare import prepare_timit  # noqa E402

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    prepare_timit(
        data_folder=hparams["data_folder"],
        splits=["train", "dev", "test"],
        save_folder=hparams["data_folder"],
        phn_set=str(hparams["ground_truth_phn_set"]),
    )

    # Collect index to label dictionary for decoding
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    hparams["ind2lab"] = hparams["train_loader"].label_dict["phn"]["index2lab"]

    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.fit(asr_brain.hparams.epoch_counter, train_set, valid_set)
    asr_brain.evaluate(hparams["test_loader"](), max_key="accuracy")
