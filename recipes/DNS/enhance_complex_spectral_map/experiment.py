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
    hyperparams_to_save=params_file,
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


def multiprocess_evaluation(pred_wavs, clean_wavs, lens, num_cores):
    processes = []

    pool = multiprocessing.Pool(processes=num_cores)

    for clean, enhanced, length in zip(clean_wavs, pred_wavs, lens):
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

        feats = params.compute_stft(wavs)  # [N, T, F, 2]
        real, imag = torch.split(feats, 1, dim=-1)
        real = params.input_norm_real(real, lens)
        imag = params.input_norm_imag(imag, lens)

        feat = torch.squeeze(torch.cat([real, imag], dim=2))  # [N, T, 2*F]
        output = params.model(feat, init_params)

        output = torch.cat(
            torch.split(
                torch.unsqueeze(output, dim=-1), params.n_fft // 2 + 1, dim=2
            ),
            dim=-1,
        )  # [N, T, F, 2]
        return output

    def compute_objectives(self, predictions, cleans, stage="train"):
        ids, wavs, lens = cleans
        wavs, lens = wavs.to(params.device), lens.to(params.device)

        com_clean = params.compute_stft(wavs)
        mag_clean = spectral_magnitude(com_clean, power=0.5)
        mag_pred = spectral_magnitude(predictions, power=0.5)

        real, imag = torch.split(com_clean, 1, dim=-1)
        real = params.output_norm_real(real, lens)
        imag = params.output_norm_imag(imag, lens)
        com_clean = torch.squeeze(torch.cat([real, imag], dim=2))  # [N, T, 2*F]

        com_loss = params.compute_cost(
            torch.squeeze(
                torch.cat(torch.split(predictions, 1, dim=-1), dim=2)
            ),
            com_clean,
            lens,
        )
        mag_loss = params.compute_cost(mag_pred, mag_clean, lens)

        return 0.5 * mag_loss + 0.5 * com_loss, {}

    def fit_batch(self, batch):
        cleans = batch[0]
        ids, clean_wavs, lens = cleans

        # Dynamically mix noises
        noisy_wavs = params.add_noise(clean_wavs, lens)

        predictions = self.compute_forward([ids, noisy_wavs, lens])
        loss, stats = self.compute_objectives(predictions, cleans)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()

        return stats

    def evaluate_batch(self, batch, stage="valid"):
        noisys, cleans = batch
        predictions = self.compute_forward(noisys, stage=stage)

        pred_wavs = params.compute_istft(predictions)

        # Normalize the waveform
        abs_max, _ = torch.max(torch.abs(pred_wavs), dim=1, keepdim=True)
        pred_wavs = pred_wavs / abs_max * 0.99

        # Evaluating PESQ and STOI
        _, clean_wavs, lens = cleans

        lens = lens * clean_wavs.shape[1]
        pesq_scores, stoi_scores = multiprocess_evaluation(
            pred_wavs.cpu().numpy(),
            clean_wavs.numpy(),
            lens.numpy(),
            multiprocessing.cpu_count(),
        )

        loss, stats = self.compute_objectives(predictions, cleans, stage=stage)
        stats["loss"] = loss.detach()
        stats["pesq"] = pesq_scores
        stats["stoi"] = stoi_scores

        if stage == "test":
            for name, pred_wav, length in zip(noisys[0], pred_wavs, lens):
                enhance_path = os.path.join(params.enhanced_folder, name)
                pred_wav = pred_wav.cpu()
                torchaudio.save(enhance_path, pred_wav[: int(length)], 16000)

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

test_stats = se_brain.evaluate(params.test_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)
