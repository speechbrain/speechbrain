#!/usr/bin/python
import os
import sys
import torch
import torchaudio
import multiprocessing
import speechbrain as sb
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_average
from pystoi.stoi import stoi
from pesq import pesq

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
    params_to_save=params_file,
    overrides=overrides,
)

if params.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    tensorboard_train_logger = TensorboardLogger(params.tensorboard_logs)

# Create the folder to save enhanced files
if not os.path.exists(params.enhanced_folder):
    os.mkdir(params.enhanced_folder)


def truncate(wavs, lengths, max_length):
    lengths *= max_length / wavs.shape[1]
    lengths = lengths.clamp(max=1)
    wavs = wavs[:, :max_length]
    return wavs, lengths


def evaluation(clean, enhanced, length):
    clean = clean[:length]
    enhanced = enhanced[:length]
    pesq_score = pesq(params.Sample_rate, clean, enhanced, "wb")
    stoi_score = stoi(clean, enhanced, params.Sample_rate)
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
        wavs, lens = truncate(wavs, lens, params.max_length)
        wavs = torch.unsqueeze(wavs, -1)
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        out = params.model(wavs, init_params=init_params)[:, :, 0]
        return out

    def compute_objectives(self, predictions, targets, stage="train"):
        ids, wavs, lens = targets
        wavs, lens = truncate(wavs, lens, params.max_length)
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        loss = params.compute_cost(predictions, wavs, lens)
        return loss, {}

    def evaluate_batch(self, batch, stage="valid"):
        inputs, targets = batch
        predict_wavs = self.compute_forward(inputs, stage=stage)

        ids, target_wavs, lens = targets
        target_wavs, lens = truncate(target_wavs, lens, params.max_length)
        lens = lens * target_wavs.shape[1]

        loss, stats = self.compute_objectives(
            predict_wavs, targets, stage=stage
        )
        stats["loss"] = loss.detach()

        pesq_scores, stoi_scores = multiprocess_evaluation(
            predict_wavs.cpu().numpy(),
            target_wavs.cpu().numpy(),
            lens.cpu().numpy(),
            multiprocessing.cpu_count(),
        )
        stats["pesq"] = pesq_scores
        stats["stoi"] = stoi_scores

        if stage == "test":
            # Write wavs to file
            for name, pred_wav, length in zip(ids, predict_wavs, lens):
                name += ".wav"
                enhance_path = os.path.join(params.enhanced_folder, name)
                torchaudio.save(enhance_path, pred_wav[: int(length)], 16000)

        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        if params.use_tensorboard:
            tensorboard_train_logger.log_stats(
                {"Epoch": epoch}, train_stats, valid_stats
            )

        params.train_logger.log_stats(
            {"Epoch": epoch}, train_stats, valid_stats
        )

        pesq_score = summarize_average(valid_stats["pesq"])
        params.checkpointer.save_and_keep_only(
            meta={"pesq_score": pesq_score},
            importance_keys=[ckpt_recency, lambda c: c.meta["pesq_score"]],
        )


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
params.checkpointer.recover_if_possible(lambda c: c.meta["pesq_score"])
test_stats = se_brain.evaluate(test_set)
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)
