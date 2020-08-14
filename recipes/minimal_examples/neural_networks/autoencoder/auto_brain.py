#!/usr/bin/python
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average


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
        return self.compute_cost(predictions, feats, lens)

    def fit_batch(self, batch):
        for optimizer in self.optimizers.values():
            inputs = batch[0]
            predictions = self.compute_forward(inputs)
            loss = self.compute_objectives(predictions, inputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch, stage="test"):
        inputs = batch[0]
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, inputs)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        if self.use_tensorboard:
            self.train_logger.log_stats(
                {"Epoch": epoch}, train_stats, valid_stats
            )
        print("Completed epoch %d" % epoch)
        print("Train loss: %.3f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.3f" % summarize_average(valid_stats["loss"]))
