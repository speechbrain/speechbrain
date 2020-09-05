#!/usr/bin/python
import os
import torch
import speechbrain as sb
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token
from speechbrain.utils.train_logger import summarize_average
from speechbrain.lobes.models.transformer.Transformer import (
    get_key_padding_mask,
    get_lookahead_mask,
)


experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


class seq2seqBrain(sb.core.Brain):
    def compute_forward(self, x, y, stage="train", init_params=False):
        id, wavs, wav_lens = x
        id, phns, phn_lens = y
        feats = hyperparams.compute_features(wavs, init_params)
        feats = hyperparams.mean_var_norm(feats, wav_lens)
        x = hyperparams.CNN(feats, init_params=init_params)

        # Prepend bos token at the beginning
        y_in = prepend_bos_token(phns, bos_index=hyperparams.bos)
        src = hyperparams.CNN(feats, init_params=init_params)

        # generate attn mask and padding mask for transformer
        src_key_padding_mask = None
        trg_key_padding_mask = get_key_padding_mask(
            y_in, pad_idx=hyperparams.pad_id
        )

        src_mask = None
        trg_mask = get_lookahead_mask(y_in)

        enc_out, pred = hyperparams.Transformer(
            src,
            y_in,
            src_mask,
            trg_mask,
            src_key_padding_mask,
            trg_key_padding_mask,
            init_params=init_params,
        )
        logits = hyperparams.lin(pred, init_params=init_params)
        outputs = hyperparams.softmax(logits)

        return outputs

    def compute_objectives(self, predictions, targets, stage="train"):
        outputs = predictions

        ids, phns, phn_lens = targets

        # Add phn_lens by one for eos token
        abs_length = torch.round(phn_lens * phns.shape[1])

        # Append eos token at the end of the label sequences
        phns = append_eos_token(
            phns, length=abs_length, eos_index=hyperparams.eos
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / phns.shape[1]
        loss = hyperparams.compute_cost(outputs, phns, length=rel_length)

        stats = {}
        return loss, stats

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets)
        loss, stats = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def evaluate_batch(self, batch, stage="test"):
        inputs, targets = batch
        out = self.compute_forward(inputs, targets, stage="test")
        loss, stats = self.compute_objectives(out, targets, stage="test")
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))


train_set = hyperparams.train_loader()
first_x, first_y = next(iter(train_set))
seq2seq_brain = seq2seqBrain(
    modules=[hyperparams.CNN, hyperparams.Transformer, hyperparams.lin],
    optimizer=hyperparams.optimizer,
    first_inputs=[first_x, first_y],
)
seq2seq_brain.fit(
    range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
)
test_stats = seq2seq_brain.evaluate(hyperparams.test_loader())


def test_error():
    assert seq2seq_brain.avg_train_loss < 1.0
