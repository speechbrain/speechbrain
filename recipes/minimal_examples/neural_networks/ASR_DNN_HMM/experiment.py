#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average

current_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(current_dir, "params.yaml")
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin)

sb.core.create_experiment_directory(
    experiment_directory=params.output_folder, params_to_save=params_file,
)


class ASR_Brain(sb.core.Brain):
    def forward(self, x, init_params=False):
        id, wavs, lens = x
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        x = params.linear1(feats, init_params)
        x = params.activation(x)
        x = params.linear2(x, init_params)
        outputs = params.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, train=True):
        predictions, lens = predictions
        ids, ali, ali_lens = targets
        loss = params.compute_cost(predictions, ali, [lens, ali_lens])

        if not train:
            err = params.compute_error(predictions, ali, [lens, ali_lens])
            stats = {"error": err}
            return loss, stats

        return loss

    def summarize(self, stats, test=False):
        summary = {"loss": float(sum(stats["loss"]) / len(stats["loss"]))}

        if "error" in stats:
            summary["error"] = float(sum(stats["error"]) / len(stats["error"]))

        return summary

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid error: %.2f" % summarize_average(valid_stats["error"]))


train_set = params.train_loader()
asr_brain = ASR_Brain(
    modules=[params.linear1, params.linear2],
    optimizer=params.optimizer,
    first_input=next(iter(train_set[0])),
)
asr_brain.fit(range(params.N_epochs), train_set, params.valid_loader())
test_stats = asr_brain.evaluate(params.test_loader())
print("Test error: %.2f" % summarize_average(test_stats["error"]))
