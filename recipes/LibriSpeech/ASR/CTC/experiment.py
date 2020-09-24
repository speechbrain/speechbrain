#!/usr/bin/env/python3
import os
import sys
import torch
import speechbrain as sb


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, x, stage):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)
        out = self.jit_modules.enc(feats)
        out = self.hparams.output(out)
        pout = self.hparams.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage):
        pout, pout_lens = predictions
        ids, char, char_len = targets
        char, char_len = char.to(self.device), char_len.to(self.device)

        if hasattr(self, "env_corrupt"):
            char = torch.cat([char, char], dim=0)
            char_len = torch.cat([char_len, char_len], dim=0)

        loss = self.hparams.ctc_cost(pout, char, pout_lens, char_len)

        if stage != sb.Stage.TRAIN:
            sequence = sb.decoders.ctc_greedy_decode(
                pout, pout_lens, blank_id=-1
            )
            self.cer_metric.append(
                ids, sequence, char, None, char_len, self.hparams.ind2lab
            )
            self.wer_metric.append(
                ids, sequence, char, None, char_len, self.hparams.ind2lab
            )

        return loss

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_tracker()
            self.wer_metric = self.hparams.wer_tracker()

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            wer = self.wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(wer)
            sb.nnet.update_learning_rate(self.optimizer, new_lr)

            cer = self.cer_metric.summarize("error_rate")
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "CER": cer, "WER": wer},
            )

            self.hparams.checkpointer.save_and_keep_only(
                meta={"WER": wer}, min_keys=["WER"],
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": epoch},
                test_stats={"loss": stage_loss, "WER": wer},
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
                print("WER stats written to file", self.hparams.wer_file)

    def on_fit_start(self):
        self.compile_jit()

        params = list(self.jit_modules.enc.parameters())
        params.extend(self.hparams.output.parameters())
        self.optimizer = self.opt_class(params)
        self.hparams.checkpointer.add_recoverable("optimizer", self.optimizer)

        # Load latest checkpoint to resume training
        self.hparams.checkpointer.recover_if_possible()

    def on_evaluate_start(self):
        # Load best checkpoint for evaluation
        self.hparams.checkpointer.recover_if_possible(min_key="WER")

        # Return loaded epoch for logging
        return self.hparams.epoch_counter.current


if __name__ == "__main__":

    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from librispeech_prepare import prepare_librispeech  # noqa E402

    # Load hyperparameters file with command-line overrides
    hparams_file, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    prepare_librispeech(
        data_folder=hparams["data_folder"],
        splits=["train-clean-100", "dev-clean", "test-clean"],
        save_folder=hparams["data_folder"],
    )

    asr_brain = ASR(
        hparams=hparams["hparams"],
        opt_class=hparams["opt_class"],
        jit_modules=hparams["jit_modules"],
        device=hparams["device"],
        ddp_procs=hparams["ddp_procs"],
    )

    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    ind2lab = hparams["train_loader"].label_dict["char"]["index2lab"]
    asr_brain.hparams.ind2lab = ind2lab

    asr_brain.fit(asr_brain.hparams.epoch_counter, train_set, valid_set)
    asr_brain.evaluate(hparams["test_loader"]())
