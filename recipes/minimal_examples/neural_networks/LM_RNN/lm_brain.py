#!/usr/bin/env python3
import torch
import math
import speechbrain as sb

from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token
from speechbrain.utils.train_logger import summarize_average


class LMBrain(sb.core.Brain):
    def compute_forward(self, y, stage="train", init_params=False):
        ids, phns, phn_lens = y
        y_in = prepend_bos_token(phns, bos_index=self.bos_index)
        logits = self.model(y_in, init_params=init_params)
        pout = self.log_softmax(logits)
        return pout

    def compute_objectives(self, predictions, targets, stage="train"):
        pout = predictions
        ids, phns, phn_lens = targets

        abs_length = torch.round(phn_lens * phns.shape[1])

        # Append eos token at the end of the label sequences
        phns_with_eos = append_eos_token(
            phns, length=abs_length, eos_index=self.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / phns_with_eos.shape[1]
        loss = self.compute_cost(pout, phns_with_eos, length=rel_length)

        return loss, {}

    def fit_batch(self, batch):
        for optimizer in self.optimizers.values():
            inputs = batch[0]
            predictions = self.compute_forward(inputs)
            loss, stats = self.compute_objectives(predictions, inputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def evaluate_batch(self, batch, stage="test"):
        inputs = batch[0]
        out = self.compute_forward(inputs, stage=stage)
        loss, stats = self.compute_objectives(out, inputs, stage=stage)
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        val_loss = summarize_average(valid_stats["loss"])
        print("Valid loss: %.2f" % val_loss)
        perplexity = math.e ** val_loss
        print("Valid perplexity: %.2f" % perplexity)
