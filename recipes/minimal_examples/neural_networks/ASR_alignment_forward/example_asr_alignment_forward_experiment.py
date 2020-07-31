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


class AlignBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x
        feats = hyperparams.compute_features(wavs, init_params)
        feats = hyperparams.mean_var_norm(feats, lens)
        x = hyperparams.model(feats, init_params=init_params)
        x = hyperparams.lin(x, init_params)
        outputs = hyperparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        ids, phns, phn_lens = targets

        sum_alpha_T = hyperparams.aligner(
            predictions, lens, phns, phn_lens, "forward"
        )

        loss = -sum_alpha_T.sum()

        stats = {}

        if stage != "train":
            viterbi_scores, alignments = hyperparams.aligner(
                predictions, lens, phns, phn_lens, "viterbi"
            )

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))


train_set = hyperparams.train_loader()
first_x, first_y = next(iter(train_set))
align_brain = AlignBrain(
    modules=[hyperparams.model, hyperparams.lin],
    optimizer=hyperparams.optimizer,
    first_inputs=[first_x],
)
align_brain.fit(
    range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
)
test_stats = align_brain.evaluate(hyperparams.test_loader())


# Integration test: check that the model overfits the training data
def test_error():
    assert align_brain.avg_train_loss < 300.0
