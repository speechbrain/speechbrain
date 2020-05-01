#!/usr/bin/python
import speechbrain as sb

params_file = "recipes/minimal_examples/neural_networks/autoencoder/params.yaml"
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


auto_brain = AutoBrain([params.linear1, params.linear2], params.optimizer)
auto_brain.fit(params.train_loader(), params.valid_loader(), params.N_epochs)
auto_brain.evaluate(params.train_loader())
