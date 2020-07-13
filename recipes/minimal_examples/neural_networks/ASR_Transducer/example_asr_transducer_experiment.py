#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.transducer import decode_batch
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.utils.train_logger import summarize_average
from speechbrain.utils.train_logger import summarize_error_rate
import pytest

pytest.importorskip("numba")
experiment_dir = os.path.dirname(os.path.realpath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


class TransducerBrain(sb.core.Brain):
    def compute_forward(self, x, y, stage="train", init_params=False):
        id, wavs, lens = x
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)
        # Transcription network: input-output dependency
        TN_output = params.encoder_crdnn(feats, init_params=init_params)
        TN_output = params.encoder_lin(TN_output, init_params)
        if stage == "train":
            _, targets, _ = y
            targets = targets.to(params.device)
            # Prediction network: output-output dependency
            decoder_input = prepend_bos_token(
                targets, bos_index=params.blank_id
            )
            PN_output = params.decoder_embedding(
                decoder_input, init_params=init_params
            )
            PN_output, _ = params.decoder_gru(
                PN_output, init_params=init_params
            )
            PN_output = params.decoder_lin(PN_output, init_params=init_params)
            # Joint the networks
            joint = params.Tjoint(
                TN_output.unsqueeze(2),
                PN_output.unsqueeze(1),
                init_params=init_params,
            )
            # projection layer
            outputs = params.output(joint, init_params=init_params)
        else:
            outputs = decode_batch(
                TN_output,
                [
                    params.decoder_embedding,
                    params.decoder_gru,
                    params.decoder_lin,
                ],
                params.Tjoint,
                [params.output],
                params.blank_id,
            )
        outputs = params.log_softmax(outputs)
        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        ids, phns, phn_lens = targets
        if stage != "train":
            pout = predictions.squeeze(2)
            predictions = predictions.expand(-1, -1, phns.shape[1] + 1, -1)

        loss = params.compute_cost(
            predictions,
            phns.to(params.device).long(),
            lens.to(params.device),
            phn_lens.to(params.device),
        )

        stats = {}
        if stage != "train":
            seq = ctc_greedy_decode(pout, lens, blank_id=params.blank_id)
            phns = undo_padding(phns, phn_lens)
            stats["PER"] = wer_details_for_batch(ids, phns, seq)

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid PER: %.2f" % summarize_error_rate(valid_stats["PER"]))

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
        out = self.compute_forward(inputs, None, stage=stage)
        loss, stats = self.compute_objectives(out, targets, stage=stage)
        stats["loss"] = loss.detach()
        return stats


train_set = params.train_loader()
first_x, first_y = next(iter(train_set))
transducer_brain = TransducerBrain(
    modules=[
        params.encoder_crdnn,
        params.encoder_lin,
        params.decoder_gru,
        params.decoder_lin,
        params.joint_lin,
        params.output,
    ],
    optimizer=params.optimizer,
    first_inputs=[first_x, first_y],
)
transducer_brain.fit(range(params.N_epochs), train_set, params.valid_loader())
test_stats = transducer_brain.evaluate(params.test_loader())
print("Test PER: %.2f" % summarize_error_rate(test_stats["PER"]))


# Integration test: check that the model overfits the training data
def test_error():
    assert transducer_brain.avg_train_loss <= 70.0
