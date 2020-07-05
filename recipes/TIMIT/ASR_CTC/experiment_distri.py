#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_error_rate
import torch.distributed as dist
import torch.multiprocessing as mp


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, wav_lens = x
        wavs, wav_lens = (
            wavs.to(self.params.device, non_blocking=True),
            wav_lens.to(self.params.device, non_blocking=True),
        )
        if hasattr(self.params, "env_corrupt"):
            wavs_noise = self.params.env_corrupt(wavs, wav_lens, init_params)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        if hasattr(self.params, "augmentation"):
            wavs = self.params.augmentation(wavs, wav_lens, init_params)
        feats = self.params.compute_features(wavs, init_params)
        feats = self.params.normalize(feats, wav_lens)
        out = self.params.model(feats, init_params)
        out = self.params.output(out, init_params)
        pout = self.params.log_softmax(out)
        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage="train"):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = (
            phns.to(self.params.device, non_blocking=True),
            phn_lens.to(self.params.device, non_blocking=True),
        )
        if hasattr(self.params, "env_corrupt"):
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)
        loss = self.params.compute_cost(pout, phns, pout_lens, phn_lens)

        stats = {}
        if stage != "train":
            ind2lab = self.params.train_loader.label_dict["phn"]["index2lab"]
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            sequence = convert_index_to_lab(sequence, ind2lab)
            phns = undo_padding(phns, phn_lens)
            phns = convert_index_to_lab(phns, ind2lab)
            per_stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats["PER"] = per_stats

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        per = summarize_error_rate(valid_stats["PER"])
        old_lr, new_lr = self.params.lr_annealing(
            [self.params.optimizer], epoch, per
        )
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        self.params.train_logger.log_stats(
            epoch_stats, train_stats, valid_stats
        )

        self.params.checkpointer.save_and_keep_only(
            meta={"PER": per},
            importance_keys=[ckpt_recency, lambda c: -c.meta["PER"]],
        )


def start_ASR(gpu, params):

    # Load hyperparameters file with command-line overrides
    params_file, overrides = sb.core.parse_arguments(sys.argv[1:])

    overrides = {
        "train_loader": {"gpu_id": gpu},
        "valid_loader": {"gpu_id": gpu},
        "test_loader": {"gpu_id": gpu},
        "device": "cuda:" + str(gpu),
    }

    with open(params_file) as fin:
        params = sb.yaml.load_extended_yaml(fin, overrides)

    world_size = params.nb_gpus
    rank = params.rank_wrt_node * params.nb_gpus + gpu
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    train_set = params.train_loader()
    valid_set = params.valid_loader()
    first_x, first_y = next(iter(train_set))

    # Modules are passed to optimizer and have train/eval called on them
    modules = [params.model, params.output]
    if hasattr(params, "augmentation"):
        modules.append(params.augmentation)

    # Create brain object for training
    asr_brain = ASR(
        modules=modules,
        optimizer=params.optimizer,
        first_inputs=[first_x],
        params=params,
    )

    # Check if the model should be trained on multiple GPUs.
    # Important: DataParallel MUST be called after the ASR (Brain) class init.
    if params.multigpu:
        params.model = torch.nn.parallel.DistributedDataParallel(
            params.model, device_ids=[gpu]
        )
        params.output = torch.nn.parallel.DistributedDataParallel(
            params.output, device_ids=[gpu]
        )

    # Load latest checkpoint to resume training
    params.checkpointer.recover_if_possible()
    asr_brain.fit(params.epoch_counter, train_set, valid_set)

    # Load best checkpoint for evaluation
    params.checkpointer.recover_if_possible(lambda c: -c.meta["PER"])
    test_stats = asr_brain.evaluate(params.test_loader())
    params.train_logger.log_stats(
        stats_meta={"Epoch loaded": params.epoch_counter.current},
        test_stats=test_stats,
    )

    # Write alignments to file
    per_summary = edit_distance.wer_summary(test_stats["PER"])
    with open(params.wer_file, "w") as fo:
        wer_io.print_wer_summary(per_summary, fo)
        wer_io.print_alignments(test_stats["PER"], fo)


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # This hack needed to import data preparation script from ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from timit_prepare import prepare_timit  # noqa E402

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

    # Prepare data
    prepare_timit(
        data_folder=params.data_folder,
        splits=["train", "dev", "test"],
        save_folder=params.data_folder,
        uppercase=True,
    )
    print("starting processes ...")
    mp.spawn(start_ASR, nprocs=params.nb_gpus, args=(sys.argv,), join=True)


if __name__ == "__main__":
    main()
