#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.decoders.transducer import decode_batch
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.utils.train_logger import summarize_average
from speechbrain.utils.train_logger import summarize_error_rate
import torch
import pytest

pytest.importorskip("numba")
experiment_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")
data_folder = "../../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.abspath(experiment_dir + data_folder)
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


class TransducerBrain(sb.core.Brain):
    def compute_forward(self, x, y, train_mode=True, init_params=False):
        ids, wavs, lens = x
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        if train_mode:
            _, targets, _ = y
            targets = targets.to(params.device)

        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)
        # Transcription Network: input-output dependency
        TN_output = params.encoder_rnn(feats, init_params=init_params)
        TN_output = params.encoder_lin(TN_output, init_params)
        if train_mode:
            # Generate input seq for PN
            blank_vect = (
                torch.ones(
                    (targets.size(0), 1),
                    device=params.device,
                    dtype=torch.int64,
                )
                * params.blank_id
            )
            prediction_seq = torch.cat((blank_vect, targets.long()), dim=1)
            # Prediction Network: output-output dependency
            PN_output = params.embedding_PN(prediction_seq, init_params)
            PN_output, _ = params.decoder_rnn(
                PN_output, init_params=init_params
            )
            PN_output = params.decoder_lin(PN_output, init_params)
            # Joint the networks
            joint = params.Tjoint(
                TN_output.unsqueeze(2), PN_output.unsqueeze(1)
            )
            # Output network
            outputs = params.output(joint, init_params)
        else:
            outputs = decode_batch(
                TN_output,
                [params.embedding_PN, params.decoder_rnn, params.decoder_lin],
                params.Tjoint,
                [params.output, params.log_softmax],
                params.blank_id,
            )

        outputs = params.log_softmax(outputs)
        return outputs, lens

    def compute_objectives(self, predictions, targets, train_mode=True):
        predictions, lens = predictions
        ids, phns, phn_lens = targets
        if not train_mode:
            # transducer tensor
            predictions = predictions.expand(-1, -1, 2, -1)
        loss = params.compute_cost(
            predictions, phns, [lens.cuda(), phn_lens.cuda()]
        )

        if not train_mode:
            seq = ctc_greedy_decode(predictions, lens, blank_id=-1)
            phns = undo_padding(phns, phn_lens)
            stats = {"PER": wer_details_for_batch(ids, phns, seq)}
            return loss, stats

        return loss

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid PER: %.2f" % summarize_error_rate(valid_stats["PER"]))

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets)
        loss = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer(self.modules)
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch):
        inputs, targets = batch
        out = self.compute_forward(inputs, None, train_mode=False)
        loss, stats = self.compute_objectives(out, targets, train_mode=False)
        stats["loss"] = loss.detach()
        return stats


train_set = params.train_loader()
first_x, first_y = next(zip(*train_set))
transducer_brain = TransducerBrain(
    modules=[
        params.encoder_rnn,
        params.encoder_lin,
        params.decoder_rnn,
        params.decoder_lin,
        params.output,
    ],
    optimizer=params.optimizer,
    first_inputs=[first_x, first_y],
)
transducer_brain.fit(range(params.N_epochs), train_set, params.valid_loader())
test_stats = transducer_brain.evaluate(params.test_loader())
print("Test PER: %.2f" % summarize_error_rate(test_stats["PER"]))


# For such a small dataset, the PER can be unpredictable.
# Instead, check that at the end of training, the error is acceptable.
def test_error():
    assert summarize_average(test_stats["loss"]) < 15.0
