#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

if hyperparams.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    train_logger = TensorboardLogger(hyperparams.tensorboard_logs)


class AutoBrain(sb.core.Brain):
    def compute_forward(self, x, init_params=False):
        id, wavs, lens = x
        feats = hyperparams.compute_features(wavs, init_params)
        feats = hyperparams.mean_var_norm(feats, lens)

        encoded = hyperparams.linear1(feats, init_params)
        encoded = hyperparams.activation(encoded)
        decoded = hyperparams.linear2(encoded, init_params)

        return decoded

    def compute_objectives(self, predictions, targets):
        id, wavs, lens = targets
        feats = hyperparams.compute_features(wavs, init_params=False)
        feats = hyperparams.mean_var_norm(feats, lens)
        return hyperparams.compute_cost(predictions, feats, lens)

    def fit_batch(self, batch):
        inputs = batch[0]
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, inputs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch, stage="test"):
        inputs = batch[0]
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, inputs)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        if hyperparams.use_tensorboard:
            train_logger.log_stats({"Epoch": epoch}, train_stats, valid_stats)
        print("Completed epoch %d" % epoch)
        print("Train loss: %.3f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.3f" % summarize_average(valid_stats["loss"]))


train_set = hyperparams.train_loader()
first_x = next(iter(train_set))
auto_brain = AutoBrain(
    modules=[hyperparams.linear1, hyperparams.linear2],
    optimizer=hyperparams.optimizer,
    first_inputs=first_x,
)
auto_brain.fit(
    range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
)
test_stats = auto_brain.evaluate(hyperparams.test_loader())
print("Test loss: %.3f" % summarize_average(test_stats["loss"]))


# Integration test: make sure we are overfitting training data
def test_loss():
    assert auto_brain.avg_train_loss < 0.08
