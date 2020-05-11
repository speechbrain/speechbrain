#!/usr/bin/python
import os
import torch
import speechbrain as sb

current_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(current_dir, "params.yaml")
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin)

sb.core.create_experiment_directory(
    experiment_directory=params.output_folder, params_to_save=params_file,
)


class AutoBrain(sb.core.Brain):
    def forward(self, x, init_params=False):
        id, wavs, lens = x
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        encoded = params.linear1(feats, init_params)
        encoded = params.activation(encoded)
        decoded = params.linear2(encoded, init_params)

        return decoded

    def compute_objectives(self, predictions, targets, train=True):
        id, wavs, lens = targets
        feats = params.compute_features(wavs, init_params=False)
        feats = params.mean_var_norm(feats, lens)
        return params.compute_cost(predictions, feats, lens)

    def fit_batch(self, batch):
        inputs = batch[0]
        predictions = self.forward(inputs)
        loss = self.compute_objectives(predictions, inputs)
        loss.backward()
        self.optimizer(self.modules)
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch):
        inputs = batch[0]
        predictions = self.forward(inputs)
        loss = self.compute_objectives(predictions, inputs, train=False)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Completed epoch %d" % epoch)
        print("Train loss: %.3f" % torch.Tensor(train_stats["loss"]).mean())
        print("Valid loss: %.3f" % torch.Tensor(valid_stats["loss"]).mean())


train_set = params.train_loader()
auto_brain = AutoBrain(
    modules=[params.linear1, params.linear2],
    optimizer=params.optimizer,
    first_input=next(iter(train_set[0])),
)
auto_brain.fit(range(params.N_epochs), train_set, params.valid_loader())
test_stats = auto_brain.evaluate(params.test_loader())
print("Test loss: %.3f" % torch.Tensor(test_stats["loss"]).mean())
