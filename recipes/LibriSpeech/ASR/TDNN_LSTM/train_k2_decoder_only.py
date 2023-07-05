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


logger = logging.getLogger(__name__)
SUBSAMPLING_FACTOR = 3  # NOTE: This is hacky. Make sure this is the same as the one used in hparams.


class K2DecASR(sb.Brain):
    def __init__(
            self, 
            rank: int,
            device: str,
            lexicon: LexiconBPE,
            world_size: int,
            tokenizer: LibriSpeechTokenizer,
            modules=None, 
            opt_class=None, 
            hparams=None, 
            run_opts={}, 
            checkpointer=None, 
            profiler=None
        ):
        self.opt_class = opt_class
        self.checkpointer = checkpointer
        self.profiler = profiler

        # Arguments passed via the run opts dictionary
        run_opt_defaults = {
            "debug": False,
            "debug_batches": 2,
            "debug_epochs": 2,
            "debug_persistently": False,
            "device": "cpu",
            "data_parallel_backend": False,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "find_unused_parameters": False,
            "jit_module_keys": None,
            "auto_mix_prec": False,
            "bfloat16_mix_prec": False,
            "max_grad_norm": 5.0,
            "nonfinite_patience": 3,
            "noprogressbar": False,
            "ckpt_interval_minutes": 0,
            "grad_accumulation_factor": 1,
            "optimizer_step_limit": None,
            "tqdm_colored_bar": False,
            "tqdm_barcolor": {
                "train": "GREEN",
                "valid": "MAGENTA",
                "test": "CYAN",
            },
        }
        # The first part is copy-pasted from the original Brain class. The reason
        #   __init__ is not called is because `device` is calculated a bit differently.
        for arg, default in run_opt_defaults.items():
            if run_opts is not None and arg in run_opts:
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: "
                        + arg
                        + " arg overridden by command line input to: "
                        + str(run_opts[arg])
                    )
                setattr(self, arg, run_opts[arg])
            else:
                # If any arg from run_opt_defaults exist in hparams and
                # not in command line args "run_opts"
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: " + arg + " arg from hparam file is used"
                    )
                    setattr(self, arg, hparams[arg])
                else:
                    setattr(self, arg, default)

        if self.data_parallel_backend and self.distributed_launch:
            sys.exit(
                "To use data_parallel backend, start your script with:\n\t"
                "python experiment.py hyperparams.yaml "
                "--data_parallel_backend=True"
                "To use DDP backend, start your script with:\n\t"
                "python -m torch.distributed.lunch [args]\n"
                "experiment.py hyperparams.yaml --distributed_launch=True "
                "--distributed_backend=nccl"
            )

        # Checkpointer should point at a temporary directory in debug mode
        if (
            self.debug
            and not self.debug_persistently
            and self.checkpointer is not None
            and hasattr(self.checkpointer, "checkpoints_dir")
        ):
            tempdir = tempfile.TemporaryDirectory()
            logger.info(
                "Since debug mode is active, switching checkpointer "
                f"output to temporary directory: {tempdir.name}"
            )
            self.checkpointer.checkpoints_dir = pathlib.Path(tempdir.name)

            # Keep reference to tempdir as long as checkpointer exists
            self.checkpointer.tempdir = tempdir
        # Sampler should be handled by `make_dataloader`
        # or if you provide a DataLoader directly, you can set
        # this.train_sampler = your_sampler
        # to have your_sampler.set_epoch() called on each epoch.
        self.train_sampler = None

        # Prepare iterating variables
        self.avg_train_loss = 0.0
        self.step = 0
        self.optimizer_step = 0

        # Add this class to the checkpointer for intra-epoch checkpoints
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("brain", self)

        # Force default color for tqdm progrressbar
        if not self.tqdm_colored_bar:
            self.tqdm_barcolor = dict.fromkeys(self.tqdm_barcolor, "")

        #####################################################################
        # Area related to changes
        #####################################################################
        if device != "cpu":
            self.device = torch.device("cuda", rank)
            # logger.info("setup device to cuda:{}".format(rank))
            torch.cuda.set_device(rank)
        else:
            self.device = torch.device("cpu")
        self.tokenizer = tokenizer
        # Put modules on the right device, accessible with dot notation
        self.modules = torch.nn.ModuleDict(modules).to(self.device)

        # List parameter count for the user
        total_params = sum(
            p.numel() for p in self.modules.parameters() if p.requires_grad
        )
        if total_params > 0:
            clsname = self.__class__.__name__
            fmt_num = sb.utils.logger.format_order_of_magnitude(total_params)
            logger.info(f"{fmt_num} trainable parameters in {clsname}")

        # Make hyperparams available with dot notation too
        if hparams is not None:
            self.hparams = SimpleNamespace(**hparams)
        # self.train_dl = train_dl
        self.lexicon = lexicon
        self.blank_index = lexicon.tokenizer.piece_to_id("<blk>")
        # self.model = self.modules.enc
        if world_size > 1:
            logger.info("Setting up DDP")
            # self.modules = DDP(self.modules, device_ids=[rank])
            for name, module in self.modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = DDP(
                        module,
                        device_ids=[rank]
                    )
                    self.modules[name] = module
        
        logger.info("Setting up topology")
        self.topology = Topology(
            hparams=self.hparams,
            lexicon=self.lexicon,
            device=self.device
        )

        logger.info("Setting up graph compiler")
        self.graph_compiler = CtcTrainingGraphCompiler(
            lexicon=self.lexicon, 
            device=self.device,
            oov=UNK
        )
        self.subsampling_factor = hparams.get("subsampling_factor", SUBSAMPLING_FACTOR)
        if self.subsampling_factor != SUBSAMPLING_FACTOR:
            logger.warning(
                f"Subsampling factor {self.subsampling_factor} is different from "
                f"the one used during training {SUBSAMPLING_FACTOR}. "
                f"Make sure this is what you want (it's probably not so "
                f"you should consider changing the global SUBSAMPLING_FACTOR to "
                f"match the one in the hparams file."
            )

        self.lm_scale_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.lm_scale_list += [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        self.lm_scale_list += [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

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
            p_ctc = self.modules.enc(feats.detach())
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
        ids = batch["ids"]
        loss = sb.nnet.losses.ctc_loss(
            p_ctc, tokens, wav_lens, token_lens,
            blank_index=self.blank_index,
        )

        # logger.info(f"{batch=}\n{loss.item()=}")
        assert loss.requires_grad == (stage == sb.Stage.TRAIN), \
            "Loss should have requires_grad={} but got requires_grad={}".format(
                stage == sb.Stage.TRAIN, loss.requires_grad
            )

        if stage != sb.Stage.TRAIN:

            # Compute outputs
            predicted_text = None
            if stage == sb.Stage.VALID:
                # TODO: Maybe do greedy decoding for the validation set and full-dec for testing
                # predicted_text: dict = self.decode_batch(
                #     batch, p_ctc=p_ctc
                # )
                wav_lens = batch["sig"][1]
                p_tokens: List[List[int]] = sb.decoders.ctc_greedy_decode(
                    p_ctc, wav_lens, blank_id=self.blank_index
                )  # padding is already removed from here
                predicted_words = [
                    self.tokenizer.sp.decode_ids(utt_seq).split(" ")
                    for utt_seq in p_tokens
                ]
                target_words = [sent.split(" ") for sent in batch["wrd"]]  # TODO: wrong tokenization (?)
                # print(f"{predicted_words[0]=}\n{target_words[0]=}")
                self.wer_metrics[0].append(ids, predicted_words, target_words)
                self.cer_metrics[0].append(ids, predicted_words, target_words)
            elif stage == sb.Stage.TEST:
                predicted_text: dict = self.decode_batch(
                    batch, p_ctc=p_ctc
                )
                # predicted_words = self.tokenizer_decode(predicted_tokens)
                predicted_words = predicted_text  # TODO: check if the outputs are tokens or words.
                target_words = [sent.split(" ") for sent in batch["wrd"]]  # TODO: wrong tokenization (?)
                for i, lm_scale in enumerate(self.lm_scale_list):
                    key = f"lm_scale_{lm_scale:.1f}"
                    # index 0 is for the validation set's greedy decoding 
                    self.wer_metrics[i+1].append(ids, predicted_words[key], target_words)
                    self.cer_metrics[i+1].append(ids, predicted_words[key], target_words)
                # Do greedy decoding for the test set, similar to the validation set
                p_tokens: List[List[int]] = sb.decoders.ctc_greedy_decode(
                    p_ctc, wav_lens, blank_id=self.blank_index
                )
                predicted_words = [
                    self.tokenizer.sp.decode_ids(utt_seq).split(" ")
                    for utt_seq in p_tokens
                ]
                self.wer_metrics[-1].append(ids, predicted_words, target_words)
        return loss
    
    def get_supervision_segments(
            self, 
            n_frames: torch.Tensor, 
            seq_ids: List[int], 
            wrds: Optional[List[str]] = None
        ):
        """ Build the supervision segments which are required for building the FSA.
            NOTE: We assume that the audio does not contain segments and that all 
                  utterances start at duration 0.
            Args:
                n_frames: tensor of number of frames in each input segment
                seq_ids: a list of the sequence ids (starting from 0, up to batch size)
                wrds: A list of the transcriptions (reordered according to how the 
                      supervision_segments are sorted). If not provided, the function
                      will return only the supervision_segments.
        """
        supervision_segments = torch.stack(
            (
                seq_ids.to(self.device),
                torch.zeros(len(seq_ids), dtype=torch.int32, device=self.device),
                torch.div(
                    n_frames.to(self.device),
                    self.subsampling_factor,
                    rounding_mode="floor",
                ),
            ),
            1
        ).to(torch.int32)

        # Sort based on duration (longest to shortest) -> required by k2.DenseFsaVec
        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices].to("cpu")
        if wrds is None:
            return supervision_segments
        wrds = [wrds[i] for i in indices]
        return supervision_segments, wrds

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metrics = []
            self.wer_metrics = []
            self.metrics_log_dir = Path(self.hparams.output_folder) / "metrics"
            os.makedirs(
                self.metrics_log_dir,
                exist_ok=True
            )
            # +1 for greedy decoding in valid
            # +1 for greedy decoding in test
            for _ in range(len(self.lm_scale_list)+2):
                self.cer_metrics.append(self.hparams.cer_computer())
                self.wer_metrics.append(self.hparams.error_rate_computer())

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.TEST:
            # best_wer: float = min([wer_metric.summarize("error_rate") for wer_metric in self.wer_metrics])
            # best_cer: float = min([cer_metric.summarize("error_rate") for cer_metric in self.cer_metrics])
            best_wer, best_lm_scale_wer = 2000, ""
            best_cer, best_lm_scale_cer = 2000, ""
            for lm_scale, cer_metric, wer_metric in zip(
                self.lm_scale_list, self.cer_metrics[1:-1], self.wer_metrics[1:-1]
            ):
                wer = wer_metric.summarize(
                    "error_rate"
                )
                # stage_stats[f"WER_lm_scale_{lm_scale:.1f}"] = wer
                cer = cer_metric.summarize(
                    "error_rate"
                )
                # stage_stats[f"CER_lm_scale_{lm_scale:.1f}"] = cer
                if wer < best_wer:
                    best_wer = wer
                    best_lm_scale_wer = lm_scale
                if cer < best_cer:
                    best_cer = cer
                    best_lm_scale_cer = lm_scale
            stage_stats["best_wer"] = best_wer
            stage_stats["best_cer"] = best_cer
            stage_stats["best_lm_scale_wer"] = best_lm_scale_wer
            stage_stats["best_lm_scale_cer"] = best_lm_scale_cer

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            stage_stats["valid WER"] = self.wer_metrics[0].summarize("error_rate")
            stage_stats["valid CER"] = self.cer_metrics[0].summarize("error_rate")
            old_lr, new_lr = self.hparams.scheduler(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer, new_lr
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={
                    "WER": stage_stats["valid WER"],
                    "CER": stage_stats["valid CER"],
                },
                min_keys=["valid WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            # To allow multiple test set names and not override
            wer_pattern = getattr(self, "wer_file_pattern", "wer")
            for lm_scale, wer_metric in zip(
                self.lm_scale_list, self.wer_metrics[1:-1]
            ):
                with open(self.metrics_log_dir / "{}_test_lm_scale={}.txt".format(wer_pattern, lm_scale), "w") as w:
                    wer_metric.write_stats(w)
            with open(self.metrics_log_dir / "{}_greedy.txt".format(wer_pattern), "w") as w:
                self.wer_metrics[-1].write_stats(w)
    
    def fit_batch(self, batch):
        """
        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        should_step = ((self.step % self.grad_accumulation_factor) == 0)
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

        if should_step:
            self.optimizer.zero_grad()
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            # if self.check_gradients(loss):
            #     self.optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.modules.parameters(), 5.0, 2.0)
            self.optimizer.step()
            self.optimizer_step += 1
        else:
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()
    
    @property
    def HLG(self):
        HLG = self.topology.HLG
        if not hasattr(HLG, "lm_scores"):
            HLG.lm_scores = HLG.scores.clone()
        return HLG

    @property
    def G(self):
        if self.hparams.decoding_method in ['nbest-rescoring', 'whole-lattice-rescoring']:
            return self.topology.G_4gram
        return None

    def decode(self, testset: DataLoader) -> Dict[str, List[Tuple[int, List[str], List[str]]]]:
        results = collections.defaultdict(list)
        for batch in testset:
            texts = batch["wrd"]
            ids = batch["ids"]
            hyps_dict = self.decode_batch(batch)
            for lm_scale, hyps in hyps_dict.items():
                this_batch = []
                assert len(hyps) == len(texts)
                for i, hyp_words, ref_text in zip(ids, hyps, texts):
                    ref_words = ref_text.split()
                    this_batch.append((i, ref_words, hyp_words))

                results[lm_scale].extend(this_batch)

        return results
        
    def decode_batch(self, batch, p_ctc=None) -> Dict[str, List[List[str]]]:
        """Decode a single batch assuming no forward pass has been done."""
        if p_ctc is None:
            with torch.no_grad():
                # Use train as the stage so that it won't try to do decoding
                p_ctc, _ = self.compute_forward(batch, sb.Stage.TRAIN)
        return self._k2_decode_from_probs(batch, p_ctc)
    
    def _k2_decode_from_probs(self, batch, p_ctc) -> Dict[str, List[List[str]]]:
        """Decode using k2 library."""
        supervision_segments = batch["supervision_segments"]

        # logger.info("Creating lattice...")
        try:
            lattice = get_lattice(
                nnet_output=p_ctc,
                decoding_graph=self.HLG,
                supervision_segments=supervision_segments,
                search_beam=self.hparams.search_beam,
                output_beam=self.hparams.output_beam,
                min_active_states=self.hparams.min_active_states,
                max_active_states=self.hparams.max_active_states,
                subsampling_factor=self.subsampling_factor,
            )
        except RuntimeError as e:
            msg = (
                "RuntimeError caught during lattice creation. "
                "This may be caused by a bad choice of beam size. "
                "Try to decrease the beam size."
            )
            msg += f"\n{supervision_segments=}\n\n{p_ctc.shape=}\n\n{self.HLG=}\n\n"
            raise RuntimeError(msg) from e

        # logger.info("Done creating lattice. Moving on to rescoring...")

        if self.hparams.decoding_method == 'nbest-rescoring':
            best_path_dict = rescore_with_n_best_list(
                lattice=lattice,
                G=self.G,
                num_paths=100,
                lm_scale_list=self.lm_scale_list,
                nbest_scale=0.5  # scale for lattice.scores
            )
        elif self.hparams.decoding_method == 'whole-lattice-rescoring':
            best_path_dict = rescore_with_whole_lattice(
                lattice=lattice,
                G_with_epsilon_loops=self.G,
                lm_scale_list=self.lm_scale_list
            )
        else:
            raise NotImplementedError(f"Decoding method {self.hparams.decoding_method} not implemented.")
        
        ans = {}
        for lm_scale_str, best_path in best_path_dict.items():
            hyps = get_texts(best_path)
            hyps = [[self.lexicon.word_table[i] for i in ids] for ids in hyps]  # TODO: word_table[idx] is wrong
            ans[lm_scale_str] = hyps
        return ans

PaddedData = collections.namedtuple("PaddedData", ["data", "lengths"])
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
    asr_brain = K2DecASR(
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
    
def setup_dist(
    rank, world_size, use_ddp_launch=False
):
    """
    rank and world_size are used only if use_ddp_launch is False.
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = (
            "localhost"
        )
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12354"
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(rank)
    if use_ddp_launch is False:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group("nccl")

def test_decode(hparams, run_opts):
    world_size = 1  # only on cpu
    run_opts["device"] = "cpu"
    tokenizer = hparams["tokenizer"]
    lexicon = LexiconBPE(
        librispeech_dir=hparams["data_folder"],
        train_sets=hparams["train_splits"],
        bpe_model_path=hparams["bpe_model_path"],
        tokenizer=tokenizer.sp,
    )
    # Load the asr_brain model
    asr_brain = K2DecASR(
        rank=0,
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
    test_dataloader_opts = hparams["test_dataloader_opts"]
    test_dataloader_opts["collate_fn"] = train_collate_fn 
    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets
    ) = dataio_prepare(hparams)
    del train_data, valid_data

    # Testing
    for testset_name in test_datasets.keys():  # keys are test_clean, test_other etc

        asr_brain.wer_file_pattern = f"wer_{testset_name}"
        asr_brain.evaluate(
            test_datasets[testset_name], test_loader_kwargs=test_dataloader_opts
        )

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
    # test_decode(hparams, run_opts)


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    main(hparams_file, run_opts, overrides)