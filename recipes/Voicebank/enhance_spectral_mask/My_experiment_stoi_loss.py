#!/usr/bin/python
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_average
from speechbrain.processing.features import spectral_magnitude
from joblib import Parallel, delayed
from stoi.stoi import stoi

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
    stoi_scores = Parallel(n_jobs=30)(
        delayed(stoi)(clean[0 : int(lens)], enhanced[0 : int(lens)], 16000)
        for enhanced, clean, lens in zip(pred_wavs, target_wavs, lengths)
    )
    pesq_scores = Parallel(n_jobs=30)(
        delayed(pesq)(
            16000, clean[0 : int(lens)], enhanced[0 : int(lens)], "wb"
        )
        for enhanced, clean, lens in zip(pred_wavs, target_wavs, lengths)
    )
    return pesq_scores, stoi_scores


class SEBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, lens = x
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        feats = params.compute_STFT(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        mask = torch.squeeze(params.model(feats, init_params=init_params))
        out = torch.mul(mask, feats)  # mask with "signal approximation (SA)"

        return out

    def compute_objectives(self, pred_wavs, targets, stage="train"):
        ids, wavs, lens = targets
        wavs, lens = wavs.to(params.device), lens.to(params.device)

        loss = params.compute_cost(pred_wavs, wavs, lens)

        stats = {}

        return loss, stats

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs)
        pred_wavs = self.resynthesize(torch.expm1(predictions), inputs)

        loss, stats = self.compute_objectives(pred_wavs, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()

        return stats

    def evaluate_batch(self, batch, stage="valid"):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, stage=stage)
        pred_wavs = self.resynthesize(torch.expm1(predictions), inputs)

        ids, target_wavs, lens = targets
        lens = lens * target_wavs.shape[1]

        loss, stats = self.compute_objectives(pred_wavs, targets, stage=stage)
        stats["loss"] = loss.detach()

        # Evaluate PESQ and STOI
        pesq_scores, stoi_scores = multiprocess_evaluation(
            pred_wavs.numpy(), target_wavs.numpy(), lens.numpy(),
        )
        stats["pesq"] = pesq_scores
        stats["stoi"] = stoi_scores

        # Comprehensive but slow evaluation for test
        if stage == "test":
            # Write wavs to file
            for name, pred_wav, length in zip(ids, pred_wavs, lens):
                name += ".wav"
                enhance_path = os.path.join(params.enhanced_folder, name)
                pred_wav = pred_wav / torch.max(torch.abs(pred_wav)) * 0.99
                torchaudio.save(
                    enhance_path, pred_wav[: int(length)].to("cpu"), 16000
                )

        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        if params.use_tensorboard:
            tensorboard_train_logger.log_stats(
                {"Epoch": epoch}, train_stats, valid_stats
            )

        params.train_logger.log_stats(
            {"Epoch": epoch}, train_stats, valid_stats
        )

        stoi_score = summarize_average(valid_stats["stoi"])
        params.checkpointer.save_and_keep_only(
            meta={"stoi_score": stoi_score},
            importance_keys=[ckpt_recency, lambda c: c.meta["stoi_score"]],
        )

    def resynthesize(self, predictions, inputs):
        ids, wavs, lens = inputs
        lens = lens * wavs.shape[1]
        predictions = predictions.cpu()

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
params.checkpointer.recover_if_possible(lambda c: c.meta["stoi_score"])
test_stats = se_brain.evaluate(test_set)
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)
