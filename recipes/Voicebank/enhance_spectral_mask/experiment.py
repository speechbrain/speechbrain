#!/usr/bin/python
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from joblib import Parallel, delayed

try:
    from pesq import pesq
except ImportError:
    print("Please install PESQ from https://pypi.org/project/pesq/")

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from voicebank_prepare import prepare_voicebank  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    hyperparams_to_save=params_file,
    overrides=overrides,
)

if params.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    tensorboard_train_logger = TensorboardLogger(params.tensorboard_logs)

# Create the folder to save enhanced files
if not os.path.exists(params.enhanced_folder):
    os.mkdir(params.enhanced_folder)


def multiprocess_evaluation(pred_wavs, target_wavs, lengths):
    pesq_scores = Parallel(n_jobs=30)(
        delayed(pesq)(
            fs=params.Sample_rate,
            ref=clean[: int(length)],
            deg=enhanced[: int(length)],
            mode="wb",
        )
        for enhanced, clean, length in zip(pred_wavs, target_wavs, lengths)
    )
    return pesq_scores


class SEBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, lens = x
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        feats = params.compute_STFT(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        mask = params.model(feats, init_params=init_params)
        predict_spec = torch.mul(
            mask, feats
        )  # mask with "signal approximation (SA)"

        # Also return predicted wav
        predict_wav = self.resynthesize(torch.expm1(predict_spec), x)

        return predict_spec, predict_wav

    def compute_objectives(self, predictions, targets, stage="train"):
        predict_spec, predict_wav = predictions
        ids, target_wav, lens = targets
        target_wav, lens = target_wav.to(params.device), lens.to(params.device)

        if hasattr(params, "waveform_target") and params.waveform_target:
            loss = params.compute_cost(predict_wav, target_wav, lens)
        else:
            targets = params.compute_STFT(target_wav)
            targets = spectral_magnitude(targets, power=0.5)
            targets = torch.log1p(targets)
            loss = params.compute_cost(predict_spec, targets, lens)

        stats = {}
        if stage != "train":
            stats["stoi"] = -stoi_loss(predict_wav, target_wav, lens)

            # Comprehensive but slow evaluation for test
            lens = lens * target_wav.shape[1]

            # Evaluate PESQ
            pesq_scores = multiprocess_evaluation(
                predict_wav.cpu().numpy(),
                target_wav.cpu().numpy(),
                lens.cpu().numpy(),
            )

            # Write wavs to file
            if stage == "test":
                for name, pred_wav, length in zip(ids, predict_wav, lens):
                    name += ".wav"
                    enhance_path = os.path.join(params.enhanced_folder, name)
                    torchaudio.save(
                        enhance_path, predict_wav[: int(length)].cpu(), 16000
                    )

            stats["pesq"] = pesq_scores

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        if params.use_tensorboard:
            tensorboard_train_logger.log_stats(
                {"Epoch": epoch}, train_stats, valid_stats
            )

        params.train_logger.log_stats(
            {"Epoch": epoch}, train_stats, valid_stats
        )

        loss = summarize_average(valid_stats["loss"])
        params.checkpointer.save_and_keep_only(
            meta={"loss": loss}, min_keys=["loss"],
        )

    def resynthesize(self, predictions, inputs):
        ids, wavs, lens = inputs
        lens = lens * wavs.shape[1]
        wavs = wavs.to(params.device)

        # Extract noisy phase
        feats = params.compute_STFT(wavs)
        phase = torch.atan2(feats[:, :, :, 1], feats[:, :, :, 0])
        complex_predictions = torch.mul(
            torch.unsqueeze(predictions, -1),
            torch.cat(
                (
                    torch.unsqueeze(torch.cos(phase), -1),
                    torch.unsqueeze(torch.sin(phase), -1),
                ),
                -1,
            ),
        )

        # Get the predicted waveform
        pred_wavs = params.compute_ISTFT(complex_predictions)

        # Normalize the waveform
        abs_max, _ = torch.max(torch.abs(pred_wavs), dim=1, keepdim=True)
        pred_wavs = pred_wavs / abs_max * 0.99
        padding = (0, wavs.shape[1] - pred_wavs.shape[1])
        return torch.nn.functional.pad(pred_wavs, padding)


# Prepare data
prepare_voicebank(
    data_folder=params.data_folder, save_folder=params.data_folder,
)
train_set = params.train_loader()
valid_set = params.valid_loader()
test_set = params.test_loader()
first_x, first_y = next(iter(train_set))

se_brain = SEBrain(
    modules=[params.model], optimizer=params.optimizer, first_inputs=[first_x],
)

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
se_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
params.checkpointer.recover_if_possible(min_key="loss")
test_stats = se_brain.evaluate(params.test_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)
