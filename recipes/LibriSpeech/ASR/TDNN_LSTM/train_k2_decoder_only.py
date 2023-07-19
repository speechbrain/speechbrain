#!/usr/bin/env/python3
"""Recipe for training an ASR system on Librispeech using k2's ctc
loss and decoding capabilities (wfst-based decoding).
"""

import collections
from datetime import datetime
import os
import pathlib
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Dict, Tuple, List, Optional
from pathlib import Path

import torch
import logging
import k2
import speechbrain as sb
import torch.multiprocessing as mp
from torch import distributed as dist
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.utils.data_utils import batch_pad_right
from speechbrain.utils.distributed import run_on_main
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from speechbrain.dataio.batch import PaddedBatch
from icefall.decode import get_lattice, rescore_with_whole_lattice, rescore_with_n_best_list
from icefall.utils import get_texts

from librispeech_prepare import prepare_librispeech
from lexicon import LexiconBPE, UNK
from prepare_musan import prepare_musan
from topology import Topology
from bpe_tokenizer import LibriSpeechTokenizer
from graph_compiler import CtcTrainingGraphCompiler
from train_k2 import setup_dist, test_decode, PaddedData, K2ASR


logger = logging.getLogger(__name__)
SUBSAMPLING_FACTOR = 3  # NOTE: This is hacky. Make sure this is the same as the one used in hparams.


class K2DecOnlyASR(K2ASR):
    def compute_forward(self, batch: PaddedBatch, stage: sb.Stage):
        """Forward computations from the waveform batches to the output probabilities."""
        # tokens, tokens_lens = batch.tokens
        wavs, wav_lens = batch["sig"]
        wavs = wavs.to(self.device)
        wav_lens = wav_lens.to(self.device)
        # Downsample the inputs if specified
        if hasattr(self.modules, "downsampler"):
            wavs = self.modules.downsampler(wavs)
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats: torch.Tensor = self.modules.normalize(feats, wav_lens).to(self.device)

        if stage == sb.Stage.TEST:  # will be used for k2 decoding
            if "supervision_segments" not in batch:
                n_frames = (wav_lens * feats.shape[1]).ceil()
                batch["supervision_segments"] = self.get_supervision_segments(
                    n_frames, batch['seq_ids']
                )
        with torch.set_grad_enabled(stage == sb.Stage.TRAIN):
            p_ctc = self.modules.enc(feats)
        return p_ctc

    def compute_objectives(
            self, 
            predictions: Tuple[torch.Tensor, Optional[torch.Tensor]],
            batch: PaddedBatch, 
            stage: sb.Stage
        ):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc = predictions

        tokens, token_lens = batch["tokens"]
        tokens = tokens.to(self.device)
        token_lens = token_lens.to(self.device)
        wav_lens = batch["sig"][1].to(self.device)
        loss = sb.nnet.losses.ctc_loss(
            p_ctc, tokens, wav_lens, token_lens,
            blank_index=self.blank_index,
        )

        # logger.info(f"{batch=}\n{loss.item()=}")
        assert loss.requires_grad == (stage == sb.Stage.TRAIN), \
            "Loss should have requires_grad={} but got requires_grad={}".format(
                stage == sb.Stage.TRAIN, loss.requires_grad
            )

        self.update_results(p_ctc, batch, stage)
        return loss

def train_collate_fn(batch: List[dict]):
    """ Collate function used for DataLoader.
        Initially (before collation), each item in the batch is a dictionary
        containing the following keys: id, sig, wrd, tokens, (duration).
        We want to convert them to tensors, pad them and return them as an
        object whose attributes have the same names as the keys.
        In addition, we want to create a supervision_segments attribute
        which will be used by k2.
    """
    ids = []
    seq_ids = []
    sigs = []
    wrds = []
    tokens = []
    tokens_bos = []
    # supervision_segments = []
    sig_type = batch[0]["sig"].dtype
    # tok_type = batch[0]["tokens"].dtype
    for idx, item in enumerate(batch):
        # the id needs to be an integer so we will remove all letters and non-numeric
        # characters from item['id'] and convert it to an integer
        # unique_idx = int(re.sub("[^0-9]", "", item["sequence_idx"]))
        ids.append(item["id"])
        sigs.append(item["sig"])
        wrds.append(item["wrd"])
        tokens.append(item["tokens"])
        tokens_bos.append(item["tokens_bos"])
        seq_ids.append(idx)

    seq_ids = torch.LongTensor(seq_ids)
    sigs, sig_lens = batch_pad_right(sigs)
    sigs = sigs.to(sig_type)#[indices]
    sig_lens = sig_lens.to(sig_type)#[indices]
    sigs = PaddedData(data=sigs, lengths=sig_lens)
    tokens, tokens_lens = batch_pad_right(tokens)
    tokens = PaddedData(data=tokens, lengths=tokens_lens)
    tokens_bos, tokens_bos_lens = batch_pad_right(tokens_bos)
    tokens_bos = PaddedData(data=tokens_bos, lengths=tokens_bos_lens)    
    
    batch = {
        "ids": ids,
        "sig": sigs,
        "wrd": wrds,
        "seq_ids": seq_ids,
        "tokens": tokens,
        "tokens_bos": tokens_bos,
    }
    return batch

def dataio_prepare(
        hparams: dict,
        tokenizer: LibriSpeechTokenizer
    ) -> Tuple[
        FilteredSortedDynamicItemDataset, 
        FilteredSortedDynamicItemDataset, 
        List[FilteredSortedDynamicItemDataset]
    ]:
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([tokenizer.sp.piece_to_id("<bos>")] + (tokens_list))
        yield tokens_bos
        # tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        # yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, [
            "id", 
            "sig", 
            "wrd", 
            "tokens_bos",
            # "tokens_eos",
            "tokens"
        ],
    )

    return (
        train_data,
        valid_data,
        test_datasets
    )

def run(rank, hparams, run_opts, world_size=1):
    if world_size > 1:
        logger.info("Rank {} initializing...".format(rank))
        setup_dist(rank, world_size)
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"  # noqa
        date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(f"{hparams['output_folder']}/log", exist_ok=True)
        log_filename = f"{hparams['output_folder']}/log/log-train-{date_time}-{rank}"
        logger.info("Rank {} done initializing...".format(rank))
        logging.basicConfig(
            filename=log_filename,
            format=formatter,
            level=logging.INFO,
            filemode="w",
        )
    tokenizer = hparams["tokenizer"]
    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets
    ) = dataio_prepare(hparams, tokenizer)
    logger.info("Rank {} done preparing data...".format(rank))
    
    lexicon = LexiconBPE(
        librispeech_dir=hparams["data_folder"],
        train_sets=hparams["train_splits"],
        bpe_model_path=hparams["bpe_model_path"],
        tokenizer=tokenizer.sp,
    )
    logger.info("Rank {} done preparing lexicon...".format(rank))
    # Trainer initialization
    asr_brain = K2DecOnlyASR(
        rank=rank,
        device=run_opts.get("device", hparams["device"]),
        lexicon=lexicon,
        world_size=world_size,
        tokenizer=tokenizer,
        modules=hparams["modules"],
        opt_class=hparams["model_opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    logger.info("Rank {} done initializing brain...".format(rank))

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    test_dataloader_opts = hparams["test_dataloader_opts"]
    train_dataloader_opts["collate_fn"] = train_collate_fn
    # TODO: create a valid/test_collate_fn that doesn't need the
    #       supervised segments sorted (i.e. we won't have to sort)
    valid_dataloader_opts["collate_fn"] = train_collate_fn 
    test_dataloader_opts["collate_fn"] = train_collate_fn 
    
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )
    # TODO: Fix naming of wer_files so that when we run the evaluation on 
    #       the 2nd test set (test-other-500) the wer of the 1st won't 
    #       be overwritten.
    # # Testing
    # for k in test_datasets.keys():  # keys are test_clean, test_other etc
    #     asr_brain.hparams.wer_file = os.path.join(
    #         hparams["output_folder"], "wer_{}.txt".format(k)
    #     )
    #     asr_brain.evaluate(
    #         test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
    #     )
    asr_brain.evaluate(
        test_datasets["test-clean"], 
        test_loader_kwargs=test_dataloader_opts
    )
    logging.info("Done!")
    if world_size > 1:
        torch.distributed.barrier()
        dist.destroy_process_group()
  
def main(hparams_file, run_opts, overrides):
    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    # sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    corrupt_models = {}
    if hparams.get("musan_aug_prob", 0.) > 0.:
        run_on_main(
            prepare_musan,
            kwargs={
                "folder": hparams["musan_folder"],
                "music_csv": hparams["music_csv"],
                "speech_csv": hparams["speech_csv"],
                "csv_file_paths": [hparams["noise_csv"]],
                "max_noise_len": hparams["musan_max_noise_len"],
                "overwrite": False
            }
        )
        corrupt_models = {
            "speech": hparams["add_speech_musan"],
            "music": hparams["add_music_musan"],
            "noise": hparams["add_noise_musan"],
        }

    tr_splits = hparams["train_splits"]
    assert isinstance(tr_splits, list), \
        f"train_splits must be a list, but {tr_splits} given"
    assert "train-clean-100" in tr_splits, \
        f"train-clean-100 must be in train_splits, but {tr_splits} given"
    # Dataset prep (parsing Librispeech)
    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
            "corrupt_models": corrupt_models,
            "corrupt_prob": hparams["musan_aug_prob"],
        },
    )
    
    # Delete all entries related to musan
    keys_to_delete = []
    for key in hparams.keys():
        if key.endswith("musan"):
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del hparams[key]
    # Decide on whether to use cuda and multi-gpu
    use_cuda = run_opts.get("use_cuda", hparams["use_cuda"]) and torch.cuda.is_available()
    device = run_opts.get("device", hparams["device"])
    if not use_cuda:
        run_opts["device"] = "cpu"
    n_devices = 1
    if device.startswith("cuda") and use_cuda:
        n_devices = len(device.split(','))
    
    args = [
        hparams, run_opts, n_devices
    ]
    
    if n_devices > 1:
        mp.spawn(run, args=args, nprocs=n_devices, join=True)
    else:
        rank = int(device[-1]) if use_cuda else 0
        run(rank, *args)
    # test_decode(hparams, run_opts, ASR=K2DecOnlyASR)


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    main(hparams_file, run_opts, overrides)