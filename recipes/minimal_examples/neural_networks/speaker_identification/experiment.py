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


class SpkIdBrain(sb.core.Brain):
    def forward(self, x, init_params=False):
        id, wavs, lens = x
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        x = params.linear1(feats, init_params)
        x = params.activation(x)
        x = params.linear2(x, init_params)
        x = torch.mean(x, dim=1, keepdim=True)
        outputs = params.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, train=True):
        predictions, lens = predictions
        uttid, spkid, _ = targets
        loss = params.compute_cost(predictions, spkid, lens)

        if not train:
            stats = {"error": params.compute_error(predictions, spkid, lens)}
            return loss, stats

        return loss

    def summarize(self, stats, write=False):
        summary = {"loss": float(sum(s["loss"] for s in stats) / len(stats))}

        if "error" in stats[0]:
            summary["error"] = float(
                sum(s["error"] for s in stats) / len(stats)
            )

        return summary


spk_id_brain = SpkIdBrain([params.linear1, params.linear2], params.optimizer)
spk_id_brain.fit(params.train_loader(), params.valid_loader(), params.N_epochs)
spk_id_brain.evaluate(params.train_loader())
