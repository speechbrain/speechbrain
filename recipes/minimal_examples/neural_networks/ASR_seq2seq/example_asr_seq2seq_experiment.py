#!/usr/bin/python
import os
import torch
import speechbrain as sb


class seq2seqBrain(sb.Brain):
    def compute_forward(self, x, y, stage):
        id, wavs, wav_lens = x
        id, phns, phn_lens = y
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, wav_lens)
        x = self.modules.enc(feats)

        # Prepend bos token at the beginning
        y_in = sb.data_io.data_io.prepend_bos_token(phns, self.hparams.bos)
        e_in = self.modules.emb(y_in)
        h, w = self.modules.dec(e_in, x, wav_lens)
        logits = self.modules.lin(h)
        outputs = self.hparams.softmax(logits)

        if stage != sb.Stage.TRAIN:
            seq, _ = self.hparams.searcher(x, wav_lens)
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
        phns = sb.data_io.data_io.append_eos_token(
            phns, length=abs_length, eos_index=self.hparams.eos
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / phns.shape[1]
        loss = self.hparams.compute_cost(outputs, phns, length=rel_length)

        if stage != sb.Stage.TRAIN:
            self.per_metrics.append(ids, seq, phns, target_len=phn_lens)

        return loss

    def fit_batch(self, batch):
        inputs, targets = batch
        preds = self.compute_forward(inputs, targets, sb.Stage.TRAIN)
        loss = self.compute_objectives(preds, targets, sb.Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage=sb.Stage.TEST):
        inputs, targets = batch
        out = self.compute_forward(inputs, targets, stage)
        loss = self.compute_objectives(out, targets, stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

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
    hparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    seq2seq_brain = seq2seqBrain(
        hparams["modules"], hparams["opt_class"], hparams
    )
    seq2seq_brain.fit(
        range(hparams["N_epochs"]),
        hparams["train_loader"](),
        hparams["valid_loader"](),
    )
    seq2seq_brain.evaluate(hparams["test_loader"]())

    # Check that model overfits for integration test
    assert seq2seq_brain.train_loss < 1.0


if __name__ == "__main__":
    main()


def test_error():
    main()
