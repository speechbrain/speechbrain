#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
import multiprocessing
import torchaudio
from speechbrain.utils.train_logger import summarize_average
from speechbrain.processing.features import spectral_magnitude
from speechbrain.utils.checkpoints import ckpt_recency

try:
    from pesq import pesq
except ImportError:
    print("Please install PESQ from https://pypi.org/project/pesq/")
try:
    from pystoi import stoi
except ImportError:
    print("Please install STOI from https://pypi.org/project/pystoi/")


# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from dns_prepare import prepare_dns  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

if params.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    tensorboard_logger = TensorboardLogger(params.tensorboard_logs)

# Create the folder to save enhanced files
if not os.path.exists(params.enhanced_folder):
    os.mkdir(params.enhanced_folder)


def evaluation(clean, enhanced, length):
    clean = clean[:length]
    enhanced = enhanced[:length]
    pesq_score = pesq(params.samplerate, clean, enhanced, "wb",)
    stoi_score = stoi(clean, enhanced, params.samplerate, extended=False)

    return pesq_score, stoi_score


def multiprocess_evaluation(pred_wavs, target_wavs, lens, num_cores):
    processes = []

    pool = multiprocessing.Pool(processes=num_cores)

    for clean, enhanced, length in zip(target_wavs, pred_wavs, lens):
        processes.append(
            pool.apply_async(evaluation, args=(clean, enhanced, int(length)))
        )

    pool.close()
    pool.join()

    pesq_scores, stoi_scores = [], []
    for process in processes:
        pesq_score, stoi_score = process.get()
        pesq_scores.append(pesq_score)
        stoi_scores.append(stoi_score)

    return pesq_scores, stoi_scores


class SEBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, lens = x
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        if stage == "train":
            wavs = params.add_noise(wavs, lens)

        feats = params.compute_stft(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        output = params.model(feats, init_params)

        return output

    def compute_objectives(self, predictions, targets, stage="train"):
        ids, wavs, lens = targets
        wavs, lens = wavs.to(params.device), lens.to(params.device)

        feats = params.compute_stft(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)

        loss = params.compute_cost(predictions, feats, lens)

        return loss, {}

    def fit_batch(self, batch):
        inputs = batch[0]
        predictions = self.compute_forward(inputs)
        loss, stats = self.compute_objectives(predictions, inputs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()

        return stats

    def evaluate_batch(self, batch, stage="valid"):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, stage=stage)
        epoch = params.epoch_counter.current

        # Write batch enhanced files to directory
        pred_wavs = self.write_wavs(
            torch.expm1(predictions), targets, epoch, stage
        )

        # Evaluating PESQ and STOI
        _, target_wavs, lens = targets

        lens = lens * target_wavs.shape[1]
        pesq_scores, stoi_scores = multiprocess_evaluation(
            pred_wavs.numpy(),
            target_wavs.numpy(),
            lens.numpy(),
            multiprocessing.cpu_count(),
        )

        loss, stats = self.compute_objectives(predictions, targets, stage=stage)
        stats["loss"] = loss.detach()
        stats["pesq"] = pesq_scores
        stats["stoi"] = stoi_scores

        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        epoch_pesq = summarize_average(valid_stats["pesq"])
        epoch_stoi = summarize_average(valid_stats["stoi"])

        if params.use_tensorboard:
            tensorboard_logger.log_stats(
                {
                    "Epoch": epoch,
                    "Valid PESQ": epoch_pesq,
                    "Valid STOI": epoch_stoi,
                },
                train_stats,
                valid_stats,
            )

        params.train_logger.log_stats(
            {"Epoch": epoch}, train_stats, valid_stats
        )

        params.checkpointer.save_and_keep_only(
            meta={"PESQ": epoch_pesq},
            importance_keys=[ckpt_recency, lambda c: c.meta["PESQ"]],
        )

    def write_wavs(self, predictions, inputs, epoch, stage):
        ids, wavs, lens = inputs
        lens = lens * wavs.shape[1]
        predictions = predictions.cpu()

        # Extract noisy phase
        feats = params.compute_stft(wavs)
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
        pred_wavs = params.compute_istft(complex_predictions)

        # Normalize the waveform
        abs_max, _ = torch.max(torch.abs(pred_wavs), dim=1, keepdim=True)
        pred_wavs = pred_wavs / abs_max * 0.99

        if stage == "test":
            for name, pred_wav, length in zip(ids, pred_wavs, lens):
                enhance_path = os.path.join(params.enhanced_folder, name)
                torchaudio.save(enhance_path, pred_wav[: int(length)], 16000)

        return pred_wavs


prepare_dns(
    data_folder=params.data_folder,
    save_folder=params.data_folder,
    valid_folder=os.path.join(params.data_folder, "valid"),
    seg_size=10.0,
)

train_set = params.train_loader()
valid_set = params.valid_loader()
first_x = next(iter(train_set))

se_brain = SEBrain(
    modules=[params.model], optimizer=params.optimizer, first_inputs=first_x,
)

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
se_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
params.checkpointer.recover_if_possible(lambda c: c.meta["PESQ"])

# Create the folder to save enhanced files
if not os.path.exists(params.enhanced_folder):
    os.mkdir(params.enhanced_folder)

test_stats = se_brain.evaluate(params.test_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)
