#!/usr/bin/python
import speechbrain as sb

params_file = "recipes/minimal_examples/neural_networks/ASR_DNN_HMM/params.yaml"
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

    def summarize(self, stats, write=False):
        summary = {"loss": float(sum(s["loss"] for s in stats) / len(stats))}

        if "error" in stats[0]:
            summary["error"] = float(
                sum(s["error"] for s in stats) / len(stats)
            )

        return summary


asr_brain = ASR_Brain([params.linear1, params.linear2], params.optimizer)
asr_brain.fit(params.train_loader(), params.valid_loader(), params.N_epochs)
asr_brain.evaluate(params.train_loader())
