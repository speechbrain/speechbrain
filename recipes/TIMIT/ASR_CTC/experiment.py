#!/usr/bin/env/python3
"""Recipe for doing ASR with phoneme targets and CTC loss on the TIMIT dataset

To run this recipe, do the following:
> python experiment.py {hyperparameter file} --data_folder /path/to/TIMIT

Using your own hyperparameter file or one of the following:
 * hyperparams/augment_CRDNN.yaml
 * hyperparams/augment_noise_CRDNN.yaml
 * hyperparams/augment_CRDNN_selfatt.yaml

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""
import os
import sys
import torch
import speechbrain as sb
from speechbrain.decoders.ctc import ctc_greedy_decode


# Define training procedure
class ASR_Brain(sb.Brain):
    def compute_forward(self, x, stage):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Adding environmental corruption if specified (i.e., noise+rev)
        if hasattr(self, "env_corrupt") and stage == sb.Stage.TRAIN:
            wavs_noise = self.env_corrupt(wavs, wav_lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        # Adding time-domain SpecAugment if specified
        if hasattr(self, "augmentation"):
            wavs = self.augmentation(wavs, wav_lens)

        feats = self.compute_features(wavs)
        feats = self.normalize(feats, wav_lens)
        out = self.model(feats)
        out = self.output(out)
        pout = self.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        if stage == sb.Stage.TRAIN and hasattr(self, "env_corrupt"):
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        loss = self.compute_cost(pout, phns, pout_lens, phn_lens)
        self.ctc_metrics.append(ids, pout, phns, pout_lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            self.per_metrics.append(ids, sequence, phns, phn_lens, self.ind2lab)

        return loss

    def on_stage_start(self, stage, epoch=None):
        self.ctc_metrics = self.ctc_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.lr_annealing(per)
            sb.nnet.update_learning_rate(self.optimizer, new_lr)

            self.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "PER": per},
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"],
            )
        elif stage == sb.Stage.TEST:
            self.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            with open(self.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print("CTC and PER stats written to file ", self.wer_file)


# Begin Recipe!
if __name__ == "__main__":

    # This hack needed to import data preparation script from ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from timit_prepare import prepare_timit  # noqa E402

    # Load hyperparameters file with command-line overrides
    params_file, overrides = sb.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = sb.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=params.output_folder,
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data
    prepare_timit(
        data_folder=params.data_folder,
        splits=["train", "dev", "test"],
        save_folder=params.data_folder,
    )

    # Create brain object for training
    train_set = params.train_loader()
    valid_set = params.valid_loader()
    ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
    asr_brain = ASR_Brain(
        modules=dict(params.modules, ind2lab=ind2lab),
        optimizers=["optimizer"],
        jit_modules=["model"],
        torch_ddp_procs=1,
    )

    # Load latest checkpoint to resume training
    asr_brain.checkpointer.recover_if_possible()
    asr_brain.fit(params.epoch_counter, train_set, valid_set)

    # Load best checkpoint for evaluation
    asr_brain.checkpointer.recover_if_possible(min_key="PER")
    test_loss = asr_brain.evaluate(params.test_loader())
