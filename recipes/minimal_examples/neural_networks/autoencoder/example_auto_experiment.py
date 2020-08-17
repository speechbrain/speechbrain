#!/usr/bin/python
import os
import speechbrain as sb


class AutoBrain(sb.core.Brain):
    def compute_forward(self, x, init_params=False):
        id, wavs, lens = x
        feats = self.compute_features(wavs, init_params)
        feats = self.mean_var_norm(feats, lens)

        encoded = self.linear1(feats, init_params)
        encoded = self.activation(encoded)
        decoded = self.linear2(encoded, init_params)

        return decoded

    def compute_objectives(self, predictions, targets):
        id, wavs, lens = targets
        feats = self.compute_features(wavs, init_params=False)
        feats = self.mean_var_norm(feats, lens)
        self.mse_metric.append(id, predictions, feats, lens)
        return self.compute_cost(predictions, feats, lens)

    def fit_batch(self, batch):
        for optimizer in self.optimizers.values():
            inputs = batch[0]
            predictions = self.compute_forward(inputs)
            loss = self.compute_objectives(predictions, inputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage=sb.core.Stage.TEST):
        inputs = batch[0]
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, inputs)
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        self.mse_metric = self.loss_tracker()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if self.use_tensorboard:
            if stage == sb.core.Stage.TRAIN:
                self.train_logger.log_stats(
                    {"Epoch": epoch},
                    train_stats={"loss": self.mse_metric.scores},
                )
            elif stage == sb.core.Stage.VALID:
                self.train_logger.log_stats(
                    {"Epoch": epoch},
                    valid_stats={"loss": self.mse_metric.scores},
                )
            if stage == sb.core.Stage.TEST:
                self.train_logger.log_stats(
                    {}, test_stats={"loss": self.mse_metric.scores}
                )

        if stage == sb.core.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.core.Stage.VALID:
            print("Completed epoch %d" % epoch)
            print("Train loss: %.3f" % self.train_loss)
            print("Valid loss: %.3f" % stage_loss)
        if stage == sb.core.Stage.TEST:
            print("Test loss: %.3f" % stage_loss)


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hyperparams_file) as fin:
        hyperparams = sb.yaml.load_extended_yaml(
            fin, {"data_folder": data_folder}
        )

    if hyperparams.use_tensorboard:
        from speechbrain.utils.train_logger import TensorboardLogger

        train_logger = TensorboardLogger(hyperparams.tensorboard_logs)
        hyperparams.modules["train_logger"] = train_logger

    train_set = hyperparams.train_loader()
    first_x = next(iter(train_set))
    auto_brain = AutoBrain(
        modules=hyperparams.modules,
        optimizers={("linear1", "linear2"): hyperparams.optimizer},
        first_inputs=first_x,
    )
    auto_brain.fit(
        range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
    )
    auto_brain.evaluate(hyperparams.test_loader())

    # Check that model overfits for integration test
    assert auto_brain.train_loss < 0.08


if __name__ == "__main__":
    main()


def test_loss():
    main()
