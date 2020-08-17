#!/usr/bin/python
import torch
import speechbrain as sb
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token


class seq2seqBrain(sb.core.Brain):
    def compute_forward(self, x, y, stage="train", init_params=False):
        id, wavs, wav_lens = x
        id, phns, phn_lens = y
        feats = self.compute_features(wavs, init_params)
        feats = self.mean_var_norm(feats, wav_lens)
        x = self.enc(feats, init_params=init_params)

        # Prepend bos token at the beginning
        y_in = prepend_bos_token(phns, bos_index=self.bos)
        e_in = self.emb(y_in, init_params=init_params)
        h, w = self.dec(e_in, x, wav_lens, init_params=init_params)
        logits = self.lin(h, init_params=init_params)
        outputs = self.softmax(logits)

        if stage != "train":
            seq, _ = self.searcher(x, wav_lens)
            return outputs, seq

        return outputs

    def compute_objectives(self, predictions, targets, stage="train"):
        if stage == "train":
            outputs = predictions
        else:
            outputs, seq = predictions

        ids, phns, phn_lens = targets

        # Add phn_lens by one for eos token
        abs_length = torch.round(phn_lens * phns.shape[1])

        # Append eos token at the end of the label sequences
        phns = append_eos_token(phns, length=abs_length, eos_index=self.eos)

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / phns.shape[1]
        loss = self.compute_cost(outputs, phns, length=rel_length)

        if stage != "train":
            self.per_metrics.append(ids, seq, phns, phn_lens)

        return loss

    def fit_batch(self, batch):
        for optimizer in self.optimizers.values():
            inputs, targets = batch
            predictions = self.compute_forward(inputs, targets)
            loss = self.compute_objectives(predictions, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage="test"):
        inputs, targets = batch
        out = self.compute_forward(inputs, targets, stage)
        loss = self.compute_objectives(out, targets, stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        if stage != "train":
            self.per_metrics = self.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == "train":
            self.train_loss = stage_loss
        if stage == "valid" and epoch is not None:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != "train":
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % self.per_metrics.summarize("error_rate"))
