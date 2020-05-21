#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.data_io.data_io import put_bos_token
from speechbrain.utils.train_logger import summarize_average

experiment_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")
data_folder = "../../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.abspath(experiment_dir + data_folder)
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


class CTCBrain(sb.core.Brain):
    def compute_forward(self, x, y, train_mode=True, init_params=False):
        id, wavs, wav_lens = x
        id, phns, phn_lens = y
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, wav_lens)
        x = params.rnn(feats, init_params=init_params)

        y_in = put_bos_token(phns, bos_index=params.bos)
        e_in = params.emb(y_in)
        h, w = params.decoder(e_in, x, wav_lens, init_params=init_params)
        logits = params.lin(h, init_params=init_params)
        outputs = params.softmax(logits)

        return outputs

    def compute_objectives(self, predictions, targets, train_mode=True):
        ids, phns, phn_lens = targets
        loss = params.compute_cost(predictions, phns, [phn_lens, phn_lens])

        return loss

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets)
        loss = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer(self.modules)
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch):
        inputs, targets = batch
        out = self.compute_forward(inputs, targets, train_mode=False)
        loss = self.compute_objectives(out, targets, train_mode=False)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))


train_set = params.train_loader()
first_x, first_y = next(zip(*train_set))
ctc_brain = CTCBrain(
    modules=[params.rnn, params.emb, params.decoder, params.lin],
    optimizer=params.optimizer,
    first_inputs=[first_x, first_y],
)
ctc_brain.fit(range(params.N_epochs), train_set, params.valid_loader())
test_stats = ctc_brain.evaluate(params.test_loader())
print("Test loss: %.2f" % summarize_average(test_stats["loss"]))
