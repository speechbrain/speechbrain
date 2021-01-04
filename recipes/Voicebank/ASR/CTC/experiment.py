#!/usr/bin/env python3
"""Recipe for doing ASR with phoneme targets and CTC loss on Voicebank

To run this recipe, do the following:
> python experiment.py {hyperparameter file} --data_folder /path/to/noisy-vctk

Use your own hyperparameter file or the provided `hyperparams.yaml`

To use noisy inputs, change `input_type` field from `clean_wav` to `noisy_wav`.
To use pretrained model, enter path in `pretrained` field.

Authors
 * Peter Plantinga 2020
"""
import os
import sys
import torch
import speechbrain as sb


# Define training procedure
class ASR_Brain(sb.Brain):
    def compute_forward(self, x, stage):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = self.modules.augmentation(wavs, wav_lens)
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        out = self.modules.model(feats)
        out = self.modules.output(out)
        pout = self.hparams.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)
        loss = self.hparams.compute_cost(pout, phns, pout_lens, phn_lens)
        self.ctc_metrics.append(ids, pout, phns, pout_lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            sequence = sb.decoders.ctc_greedy_decode(
                pout, pout_lens, blank_id=-1
            )
            self.per_metrics.append(
                ids, sequence, phns, None, phn_lens, self.hparams.ind2lab
            )

        return loss

    def on_stage_start(self, stage, epoch):
        self.ctc_metrics = self.hparams.ctc_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.update_learning_rate(self.optimizer, new_lr)

            # In distributed setting, only want to save model/stats once
            if self.root_process:
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch, "lr": old_lr},
                    train_stats={"loss": self.train_loss},
                    valid_stats={"loss": stage_loss, "PER": per},
                )
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per}, min_keys=["PER"],
                )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print("CTC and PER stats written to ", self.hparams.per_file)


# Begin Recipe!
if __name__ == "__main__":

    # This hack needed to import data preparation script from ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from voicebank_prepare import prepare_voicebank  # noqa E402

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
    prepare_voicebank(
        data_folder=hparams["data_folder"], save_folder=hparams["data_folder"],
    )

    # Collect index to label dictionary for decoding
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    hparams["ind2lab"] = hparams["train_loader"].label_dict["phn"]["index2lab"]

    # Load pretrained model
    if "pretrained" in hparams:
        state_dict = torch.load(hparams["pretrained"])
        hparams["modules"]["model"].load_state_dict(state_dict)

    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.fit(asr_brain.hparams.epoch_counter, train_set, valid_set)
    asr_brain.evaluate(hparams["test_loader"](), min_key="PER")
