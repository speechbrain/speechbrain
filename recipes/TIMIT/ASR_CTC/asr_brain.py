#!/usr/bin/env python3
import torch
import speechbrain as sb
from speechbrain.decoders.ctc import ctc_greedy_decode


# Define training procedure
class ASR_Brain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Adding environmental corruption if specified (i.e., noise+rev)
        if hasattr(self, "env_corrupt") and stage == "train":
            wavs_noise = self.env_corrupt(wavs, wav_lens, init_params)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        # Adding time-domain SpecAugment if specified
        if hasattr(self, "augmentation"):
            wavs = self.augmentation(wavs, wav_lens, init_params)

        feats = self.compute_features(wavs, init_params)
        feats = self.normalize(feats, wav_lens)
        out = self.model(feats, init_params)
        out = self.output(out, init_params)
        pout = self.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage="train"):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        if stage == "train" and hasattr(self, "env_corrupt"):
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        loss = self.compute_cost(pout, phns, pout_lens, phn_lens)
        self.stats[stage]["loss"].append(ids, pout, phns, pout_lens, phn_lens)

        if stage != "train":
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            self.stats[stage]["PER"].append(
                ids, sequence, phns, phn_lens, self.ind2lab
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        self.stats = {stage: {"loss": self.ctc_stats()}}

        if stage != "train":
            self.stats[stage]["PER"] = self.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == "train":
            self.train_loss = stage_loss
        elif stage == "valid":
            per = self.stats["valid"]["PER"].summarize()
            old_lr, new_lr = self.lr_annealing(
                self.optimizers.values(), epoch, per
            )
            self.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "PER": per},
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"],
            )
        elif stage == "test":
            test_per = self.stats["test"]["PER"].summarize()
            self.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": test_per},
            )
            with open(self.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.stats["test"]["loss"].write_stats(w)
                w.write("\nPER stats:\n")
                self.stats["test"]["PER"].write_stats(w)
                print("CTC and PER stats written to file ", self.wer_file)
