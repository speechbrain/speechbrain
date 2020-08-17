#!/usr/bin/python
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
