#!/usr/bin/env python3
import torch
import speechbrain as sb
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.train_logger import summarize_error_rate


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
        stats = {}

        if stage == "train":
            if hasattr(self, "env_corrupt"):
                phns = torch.cat([phns, phns], dim=0)
                phn_lens = torch.cat([phn_lens, phn_lens], dim=0)
            loss = self.compute_cost(pout, phns, pout_lens, phn_lens)
        else:
            loss = self.compute_cost(pout, phns, pout_lens, phn_lens)
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            sequence = convert_index_to_lab(sequence, self.ind2lab)
            phns = undo_padding(phns, phn_lens)
            phns = convert_index_to_lab(phns, self.ind2lab)
            per_stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats["PER"] = per_stats
        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        per = summarize_error_rate(valid_stats["PER"])
        old_lr, new_lr = self.lr_annealing(self.optimizers.values(), epoch, per)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        self.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        self.checkpointer.save_and_keep_only(
            meta={"PER": per}, min_keys=["PER"],
        )
