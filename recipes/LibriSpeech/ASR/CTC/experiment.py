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
        if hasattr(self, "env_corrupt"):
            wavs_noise = self.env_corrupt(wavs, wav_lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        if hasattr(self, "augmentation"):
            wavs = self.augmentation(wavs, wav_lens)
        feats = self.compute_features(wavs)
        feats = self.normalize(feats, wav_lens)
        out = self.enc(feats)
        out = self.output(out)
        pout = self.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage):
        pout, pout_lens = predictions
        ids, char, char_len = targets
        char, char_len = char.to(self.device), char_len.to(self.device)

        if hasattr(self, "env_corrupt"):
            char = torch.cat([char, char], dim=0)
            char_len = torch.cat([char_len, char_len], dim=0)

        loss = self.ctc_cost(pout, char, pout_lens, char_len)

        if stage != sb.Stage.TRAIN:
            sequence = sb.decoders.ctc_greedy_decode(
                pout, pout_lens, blank_id=-1
            )
            self.cer_metric.append(
                ids, sequence, char, target_len=char_len, ind2lab=self.ind2lab
            )
            self.wer_metric.append(
                ids, sequence, char, target_len=char_len, ind2lab=self.ind2lab
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.cer_tracker()
            self.wer_metric = self.wer_tracker()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            wer = self.wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.lr_annealing(wer)
            sb.nnet.update_learning_rate(self.optimizer, new_lr)

            cer = self.cer_metric.summarize("error_rate")
            self.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "CER": cer, "WER": wer},
            )

            self.checkpointer.save_and_keep_only(
                meta={"WER": wer}, min_keys=["WER"],
            )

        if stage == sb.Stage.TEST:
            self.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.epoch_counter.current},
                test_stats={"loss": stage_loss, "WER": wer},
            )
            with open(self.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
                print("WER stats written to file", self.wer_file)


if __name__ == "__main__":

    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from librispeech_prepare import prepare_librispeech  # noqa E402

    # Load hyperparameters file with command-line overrides
    params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = sb.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=params.output_folder,
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data
    prepare_librispeech(
        data_folder=params.data_folder,
        splits=["train-clean-100", "dev-clean", "test-clean"],
        save_folder=params.data_folder,
    )
    train_set = params.train_loader()
    valid_set = params.valid_loader()
    ind2lab = params.train_loader.label_dict["char"]["index2lab"]
    params.modules["ind2lab"] = ind2lab
    asr_brain = ASR(modules=params.modules, optimizers=["optimizer"])

    # Load latest checkpoint to resume training
    asr_brain.checkpointer.recover_if_possible()
    asr_brain.fit(asr_brain.epoch_counter, train_set, valid_set)

    # Load best checkpoint for evaluation
    asr_brain.checkpointer.recover_if_possible(min_key="WER")
    test_stats = asr_brain.evaluate(params.test_loader())
