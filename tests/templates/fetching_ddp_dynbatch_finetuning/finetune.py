#!/usr/bin/env/python3
"""Recipe for debugging/testing, based upon SpeechBrain's minilibrispeech ASR template.

Does the following feature set work out together on some environment?
    DDP; dynamic batching; fine-tuning; mixed pretrainer fetching & testing using pretrained interface

Authors:
    * Andreas Nautsch 2023
"""
import os
import sys
import torch
import logging
import speechbrain as sb
from copy import deepcopy
from tqdm.contrib import tqdm
from torch.utils.data import DataLoader
from hyperpyyaml import load_hyperpyyaml
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.distributed import run_on_main, ddp_barrier
from speechbrain.utils.data_utils import batch_pad_right
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.dataio.dataloader import (
    LoopedLoader,
    make_dataloader,
    distributed_loader_specifics,
)
from speechbrain.dataio.dataio import (
    read_audio,
    # read_audio_multichannel,
)
from ASR_template_train import ASR, dataio_prepare


logger = logging.getLogger(__name__)


def eval_reporting(reports, single_node=False):
    """Performance logging independent of who logs what.

    Parameters
    ----------
    reports: dict
        Maps metric labels to performance trackers (instances) which need summarise certain fields for final reporting.
    single_node: bool
        Flag for whether/not DDP-gather results (Default: False).
    """
    for log_metric, specs in reports.items():
        if not single_node:
            print(
                f'{log_metric} on DDP rank {int(os.environ["RANK"])}: {specs["tracker"].summarize()}'
            )
            result_list = [None for _ in range(int(os.environ["WORLD_SIZE"]))]
            # WARNING: https://pytorch.org/docs/stable/distributed.html - underlying `pickle` module is known to be insecure
            torch.distributed.all_gather_object(
                result_list, specs["tracker"].scores
            )
            specs["tracker"].scores = list()
            for r in result_list:
                specs["tracker"].scores.extend(r)
        summary = specs["tracker"].summarize()
        print(f"\tSummary: {summary}")
        logger.info(f'{log_metric}: {summary[specs["field"]]}\n')


def eval_test_use_recipe_dataio(
    encoder_decoder_asr, test_set, test_kwargs, reporter, single_node=False
):
    """Bypassing speechbrain.pretrained.Pretrained.load_audio with recipe dataio (speechbrain.dataio.dataio.read_audio).

    Parameters
    ----------
    encoder_decoder_asr: speechbrain.pretrained.EncoderDecoderASR
        Pretrained interface (other interfaces will require other functions to be called; this is an example).
    test_set: dict
        Data loader options for testing.
    test_kwargs: dict
        Data loader options for testing.
    reporter: dict
        Maps metric labels to performance trackers (instances) which need summarise certain fields for final reporting.
    single_node: bool
        Flag for whether/not DDP-gather results (Default: False).
    """
    if "ckpt_prefix" in test_kwargs:
        del test_kwargs["ckpt_prefix"]
    if not (
        isinstance(test_set, DataLoader) or isinstance(test_set, LoopedLoader)
    ):
        test_set = make_dataloader(test_set, **test_kwargs)

    with torch.no_grad():
        for batch in tqdm(test_set, dynamic_ncols=True, disable=False):
            batch = batch.to(encoder_decoder_asr.device)
            predictions = encoder_decoder_asr.transcribe_batch(*batch.sig)

            # prepare for metric reporting
            predicted = [wrd.split(" ") for wrd in predictions[0]]
            targeted = [wrd.split(" ") for wrd in batch.words]
            ids = batch.id
            for metric in reporter.keys():
                reporter[metric]["tracker"].append(
                    ids=ids, predict=predicted, target=targeted
                )

        # Report summary
        eval_reporting(reports=reporter, single_node=single_node)


def eval_test_batch_from_scratch(
    encoder_decoder_asr,
    test_set,
    test_kwargs,
    reporter,
    pretrainer_load_audio=False,
):
    """Relies only on batched audio paths to create batches using the pretrained interface only.

    Parameters
    ----------
    encoder_decoder_asr: speechbrain.pretrained.EncoderDecoderASR
        Pretrained interface (other interfaces will require other functions to be called; this is an example).
    test_set: Dataset, DataLoader
        If a DataLoader is given, it is iterated directly. Otherwise passed to `sb.dataio.dataloader.make_dataloader()`.
    test_kwargs: dict
        Data loader options for testing.
    reporter: dict
        Maps metric labels to performance trackers (instances) which need summarise certain fields for final reporting.
    pretrainer_load_audio: bool (Default: False)
        Whether to use Pretrainer.load_audio (True) or dataio.read_audio (False).
    """
    if "ckpt_prefix" in test_kwargs:
        del test_kwargs["ckpt_prefix"]
    if not (
        isinstance(test_set, DataLoader) or isinstance(test_set, LoopedLoader)
    ):
        test_set = make_dataloader(test_set, **test_kwargs)

    with torch.no_grad():
        for batch in tqdm(test_set, dynamic_ncols=True, disable=False):
            # instead of using batch.sig, we desire to see pretrained_hf_asr.load_audio in action
            wavs = []
            for audio_path in batch.wav:  # get the paths only
                if pretrainer_load_audio:
                    wavs.append(
                        encoder_decoder_asr.load_audio(
                            path=audio_path, silent_local_fetch=True
                        )
                    )
                else:
                    wavs.append(
                        read_audio(audio_path).to(encoder_decoder_asr.device)
                    )
            wavs, wav_lens = batch_pad_right(wavs)
            predictions = encoder_decoder_asr(wavs, wav_lens)

            # prepare for metric reporting
            predicted = [wrd.split(" ") for wrd in predictions[0]]
            targeted = [wrd.split(" ") for wrd in batch.words]
            ids = batch.id
            for metric in reporter.keys():
                reporter[metric]["tracker"].append(
                    ids=ids, predict=predicted, target=targeted
                )

        # Report summary
        ddp_barrier()
        eval_reporting(reports=reporter)


if __name__ == "__main__":
    # CLI parsing
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # DDP init
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Kept aside for later
    reporting = {
        "WER": {
            "field": "error_rate",
            "tracker": deepcopy(
                hparams["error_rate_computer"]()
            ),  # n_jobs=int(os.environ['WORLD_SIZE']))),
        },
        "CER": {
            "field": "error_rate",
            "tracker": deepcopy(
                hparams["cer_computer"]()
            ),  # n_jobs=int(os.environ['WORLD_SIZE']))),
        },
    }

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Spared: dataset prepare script & its run_on_main - prepare dataio pipeline only
    datasets = dataio_prepare(hparams)

    # Add: dynamic batching (only in training & validation)
    dynamic_hparams = hparams["dynamic_batch_sampler"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    train_dataloader_opts = {
        "batch_sampler": DynamicBatchSampler(
            datasets["train"],
            dynamic_hparams["max_batch_len"],
            num_buckets=dynamic_hparams["num_buckets"],
            length_func=lambda x: x["length"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        ),
        "num_workers": train_dataloader_opts["num_workers"],
    }
    valid_dataloader_opts = {
        "batch_sampler": DynamicBatchSampler(
            datasets["valid"],
            dynamic_hparams["max_batch_len"],
            num_buckets=dynamic_hparams["num_buckets"],
            length_func=lambda x: x["length"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        ),
        "num_workers": valid_dataloader_opts["num_workers"],
    }

    # We download:
    # * the tokenizer from URL - https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech/
    # * the pretrained LM from HuggingFace - HF: speechbrain/asr-crdnn-rnnlm-librispeech
    # * the pretrained ASR from the local template checkpoint - local: speechbrain/asr-crdnn-rnnlm-librispeech
    run_on_main(hparams["pretrainer_tokenizer"].collect_files,)
    run_on_main(hparams["pretrainer_LM"].collect_files,)
    hparams["pretrainer_tokenizer"].load_collected(run_opts["device"])
    hparams["pretrainer_LM"].load_collected(run_opts["device"])
    # LOCAL fetching takes sources directly from their location
    hparams["pretrainer_ASR"].collect_files()
    hparams["pretrainer_ASR"].load_collected(run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=deepcopy(run_opts),
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.hparams.valid_search = asr_brain.hparams.valid_search.to(
        asr_brain.device
    )
    asr_brain.hparams.test_search = asr_brain.hparams.test_search.to(
        asr_brain.device
    )
    asr_brain.cer_metric = deepcopy(reporting["CER"]["tracker"])
    asr_brain.wer_metric = deepcopy(reporting["WER"]["tracker"])

    # Freeze all but LM
    for mod in [
        "encoder",
        "embedding",
    ]:  # decoder, ctc_lin & seq_lin are for fine-tuning
        for param in getattr(asr_brain.modules, mod).parameters():
            param.requires_grad = False

    # Fine-tuning
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    asr_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    # Save, so it can be found as pre-trained model source
    if not os.path.exists(f"{hparams['save_folder']}/CKPT+latest"):
        ddp_barrier()  # just to be sure
        run_on_main(
            asr_brain.checkpointer.save_checkpoint, kwargs={"name": "latest"}
        )

    # Clean-up memory (for using EncoderDecoderASR interface only) but preserve testing-relevant objects
    test_datasets = deepcopy(datasets["test"])
    test_loader_kwargs = deepcopy(hparams["test_dataloader_opts"])
    del asr_brain, datasets

    # Pre-trained interface
    pretrained_asr = EncoderDecoderASR.from_hparams(
        source="source_pretrained",
        savedir="source_pretrained",
        hparams_file="pretrained.yaml",
        run_opts=deepcopy(run_opts),
    )

    # Test w/ DDP
    ddp_test_set = DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": hparams["data_folder"]},
        dynamic_items=[],
        output_keys=["id", "wav", "words"],
    )
    ddp_test_kwargs = distributed_loader_specifics(
        distributed_launch=True,
        rank=int(os.environ["RANK"]),
        dataset=ddp_test_set,
        loader_kwargs=deepcopy(test_loader_kwargs),
    )
    for flag in [True, False]:
        logger.info(f"\nBatch from scratch w/ pretrainer_load_audio={flag}")
        eval_test_batch_from_scratch(
            encoder_decoder_asr=pretrained_asr,
            test_set=deepcopy(ddp_test_set),
            test_kwargs=deepcopy(ddp_test_kwargs),
            reporter=deepcopy(reporting),
            pretrainer_load_audio=flag,
        )

    # Re:testing w/ previous dataloader // note: needs to run as last item (the script might get stuck otherwise)
    logger.info(f"\nTesting w/ asr_brain's eval dataloader")
    run_on_main(
        eval_test_use_recipe_dataio,
        kwargs={
            "encoder_decoder_asr": pretrained_asr,
            "test_set": deepcopy(test_datasets),
            "test_kwargs": deepcopy(test_loader_kwargs),
            "reporter": deepcopy(reporting),
            "single_node": True,
        },
    )
