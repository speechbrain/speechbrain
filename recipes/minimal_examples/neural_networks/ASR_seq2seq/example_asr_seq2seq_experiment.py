#!/usr/bin/python
import os
import torch
import speechbrain as sb
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token


class seq2seqBrain(sb.Brain):
    def compute_forward(self, x, y, stage):
        id, wavs, wav_lens = x
        id, phns, phn_lens = y
        feats = self.compute_features(wavs)
        feats = self.mean_var_norm(feats, wav_lens)
        x = self.enc(feats)

        # Prepend bos token at the beginning
        y_in = prepend_bos_token(phns, bos_index=self.bos)
        e_in = self.emb(y_in)
        h, w = self.dec(e_in, x, wav_lens)
        logits = self.lin(h)
        outputs = self.softmax(logits)

        if stage != sb.Stage.TRAIN:
            seq, _ = self.searcher(x, wav_lens)
            return outputs, seq

        return outputs

    def compute_objectives(self, predictions, targets, stage):
        if stage == sb.Stage.TRAIN:
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

        if stage != sb.Stage.TRAIN:
            self.per_metrics.append(ids, seq, phns, phn_lens)

        return loss

    def fit_batch(self, batch):
        for optimizer_name in self.optimizers:
            optimizer = getattr(self, optimizer_name)
            inputs, targets = batch
            preds = self.compute_forward(inputs, targets, sb.Stage.TRAIN)
            loss = self.compute_objectives(preds, targets, sb.Stage.TRAIN)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage=sb.Stage.TEST):
        inputs, targets = batch
        out = self.compute_forward(inputs, targets, stage)
        loss = self.compute_objectives(out, targets, stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID and epoch is not None:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % self.per_metrics.summarize("error_rate"))


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hyperparams_file) as fin:
        hyperparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    seq2seq_brain = seq2seqBrain(
        modules=hyperparams.modules, optimizers=["optimizer"], device="cpu",
    )
    seq2seq_brain.fit(
        range(hyperparams.N_epochs),
        hyperparams.train_loader(),
        hyperparams.valid_loader(),
    )
    seq2seq_brain.evaluate(hyperparams.test_loader())

    # Check that model overfits for integration test
    assert seq2seq_brain.train_loss < 1.0


if __name__ == "__main__":
    main()


def test_error():
    main()
