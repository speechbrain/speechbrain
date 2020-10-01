#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, x, stage):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        if hasattr(self.hparams, "augmentation") and stage == sb.Stage.TRAIN:
            wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)
        out = self.jit_modules.model(feats)
        out = self.hparams.output(out)
        pout = self.hparams.log_softmax(out)
        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage):
        pout, pout_lens = predictions
        ids, chars, char_lens = targets
        chars, char_lens = chars.to(self.device), char_lens.to(self.device)
        loss = self.hparams.compute_cost(pout, chars, pout_lens, char_lens)

        if stage != sb.Stage.TRAIN:
            pred_chars = sb.decoders.ctc_greedy_decode(pout, pout_lens)
            self.cer_metrics.append(
                ids, pred_chars, chars, None, char_lens, self.hparams.ind2lab
            )

        return loss

    """
    def fit_batch(self, batch):
        if self.hparams.loaders == "noisy_loaders":
            (ids, clean, lens), (_, noisy, _), (_, chars, char_lens) = batch
            joint_batch = torch.cat((clean, noisy))
            inputs = (ids + ids, joint_batch, torch.cat((lens, lens)))
            joint_targets = torch.cat((chars, chars))
            targets = (ids, joint_targets, torch.cat((char_lens, char_lens)))
        else:
            inputs, targets = batch

        predictions = self.compute_forward(inputs, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, targets, sb.Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()
    """

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.cer_metrics = self.hparams.cer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            cer = self.cer_metrics.summarize("error_rate")
            stage_stats = {"loss": stage_loss, "CER": cer}

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(cer)
            sb.nnet.update_learning_rate(self.optimizer, new_lr)

            epoch_stats = {"epoch": epoch, "lr": old_lr}
            self.hparams.train_logger.log_stats(
                epoch_stats, {"loss": self.train_loss}, stage_stats
            )
            self.checkpointer.save_and_keep_only(
                meta={"CER": cer}, min_keys=["CER"],
            )
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w") as w:
                self.cer_metrics.write_stats(w)


if __name__ == "__main__":
    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from voicebank_prepare import prepare_voicebank  # noqa E402

    # Load hyperparameters file with command-line overrides
    hparams_file, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

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

    loaders = hparams["loaders"]
    train_set = hparams[loaders]["train"]()
    valid_set = hparams[loaders]["valid"]()
    test_set = hparams[loaders]["test"]()
    ind2lab = hparams[loaders]["train"].label_dict["char"]["index2lab"]
    hparams["hparams"]["ind2lab"] = ind2lab

    if "pretrained" in hparams:
        params = torch.load(hparams["pretrained"])
        hparams["jit_modules"]["model"].load_state_dict(params)

    asr_brain = ASR(
        hparams=hparams["hparams"],
        opt_class=hparams["opt_class"],
        jit_modules=hparams["jit_modules"],
        checkpointer=hparams["checkpointer"],
        device=hparams["device"],
    )

    asr_brain.fit(asr_brain.hparams.epoch_counter, train_set, valid_set)
    asr_brain.evaluate(test_set)
