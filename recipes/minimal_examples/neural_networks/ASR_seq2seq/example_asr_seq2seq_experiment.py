#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token
from speechbrain.decoders.decoders import undo_padding
from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.utils.train_logger import summarize_average
from speechbrain.utils.train_logger import summarize_error_rate
import torch

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

searcher = S2SRNNGreedySearcher(
    modules=[
        hyperparams.emb,
        hyperparams.dec,
        hyperparams.lin,
        hyperparams.softmax,
    ],
    bos_index=hyperparams.bos,
    eos_index=hyperparams.eos,
    min_decode_ratio=0,
    max_decode_ratio=0.1,
)


class seq2seqBrain(sb.core.Brain):
    def compute_forward(self, x, y, stage="train", init_params=False):
        id, wavs, wav_lens = x
        id, phns, phn_lens = y
        feats = hyperparams.compute_features(wavs, init_params)
        feats = hyperparams.mean_var_norm(feats, wav_lens)
        x = hyperparams.enc(feats, init_params=init_params)

        # Prepend bos token at the beginning
        y_in = prepend_bos_token(phns, bos_index=hyperparams.bos)
        e_in = hyperparams.emb(y_in, init_params=init_params)
        h, w = hyperparams.dec(e_in, x, wav_lens, init_params=init_params)
        logits = hyperparams.lin(h, init_params=init_params)
        outputs = hyperparams.softmax(logits)

        if stage != "train":
            seq, _ = searcher(x, wav_lens)
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
        phns = append_eos_token(
            phns, length=abs_length, eos_index=hyperparams.eos
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / phns.shape[1]
        loss = hyperparams.compute_cost(outputs, phns, length=rel_length)

        stats = {}
        if stage != "train":
            phns = undo_padding(phns, phn_lens)
            stats["PER"] = wer_details_for_batch(ids, phns, seq)
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
        print("Valid PER: %.2f" % summarize_error_rate(valid_stats["PER"]))


train_set = hyperparams.train_loader()
first_x, first_y = next(iter(train_set))
seq2seq_brain = seq2seqBrain(
    modules=[
        hyperparams.enc,
        hyperparams.emb,
        hyperparams.dec,
        hyperparams.lin,
    ],
    optimizer=hyperparams.optimizer,
    first_inputs=[first_x, first_y],
)
seq2seq_brain.fit(
    range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
)
test_stats = seq2seq_brain.evaluate(hyperparams.test_loader())
print("Test PER: %.2f" % summarize_error_rate(test_stats["PER"]))


def test_error():
    assert seq2seq_brain.avg_train_loss < 1.0
