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
from speechbrain.pretrained.fetching import FetchFrom
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import batch_pad_right
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.dataio.dataloader import LoopedLoader, make_dataloader
from speechbrain.dataio.dataio import (
    read_audio,
    # read_audio_multichannel,
)
from ASR_template_train import ASR, dataio_prepare


logger = logging.getLogger(__name__)


def eval_reporting(reports):
    """Performance logging independent of who logs what.

    Parameters
    ----------
    reports: dict
        Maps metric labels to performance trackers (instances) which need summarise certain fields for final reporting.
    """
    for log_metric, specs in reports.items():
        logger.info(
            f'{log_metric}: {specs["tracker"].summarize(specs["field"])}'
        )


def eval_test_use_recipe_dataio(encoder_decoder_asr, test_set):
    """Bypassing speechbrain.pretrained.Pretrained.load_audio with recipe dataio (speechbrain.dataio.dataio.read_audio).

    Parameters
    ----------
    encoder_decoder_asr: speechbrain.pretrained.EncoderDecoderASR
        Pretrained interface (other interfaces will require other functions to be called; this is an example).
    test_set: dict
        Data loader options for testing.
    """
    if not (
        isinstance(test_set, DataLoader) or isinstance(test_set, LoopedLoader)
    ):
        if "ckpt_prefix" in test_loader_kwargs:
            del test_loader_kwargs["ckpt_prefix"]
        test_set = make_dataloader(test_set, **test_loader_kwargs)

    with torch.no_grad():
        for batch in tqdm(test_set, dynamic_ncols=True, disable=False):
            batch = batch.to(encoder_decoder_asr.device)
            predictions = encoder_decoder_asr.transcribe_batch(*batch.sig)

            # prepare for metric reporting
            predicted = [wrd.split(" ") for wrd in predictions[0]]
            targeted = [wrd.split(" ") for wrd in batch.words]
            ids = batch.id
            for metric in reporting.keys():
                reporting[metric]["tracker"].append(
                    ids=ids, predict=predicted, target=targeted
                )

        # Report summary
        eval_reporting(reports=reporting)


def eval_test_batch_from_scratch(encoder_decoder_asr, test_set):
    """Relies only on batched audio paths to create batches using the pretrained interface only.

    Parameters
    ----------
    encoder_decoder_asr: speechbrain.pretrained.EncoderDecoderASR
        Pretrained interface (other interfaces will require other functions to be called; this is an example).
    test_set: dict
        Data loader options for testing.
    """
    if not (
        isinstance(test_set, DataLoader) or isinstance(test_set, LoopedLoader)
    ):
        if "ckpt_prefix" in test_loader_kwargs:
            del test_loader_kwargs["ckpt_prefix"]
        test_set = make_dataloader(test_set, **test_loader_kwargs)

    with torch.no_grad():
        for batch in tqdm(test_set, dynamic_ncols=True, disable=False):
            # instead of using batch.sig, we desire to see pretrained_hf_asr.load_audio in action
            wavs = []
            for audio_path in batch.wavs:  # get the paths only
                wavs.append(encoder_decoder_asr.load_audio(path=audio_path))
                loaded = read_audio(audio_path)
                wavs.append(loaded)
            wavs, wav_lens = batch_pad_right(wavs)
            wav_lens = wav_lens.to(encoder_decoder_asr.device)
            enc = encoder_decoder_asr.encode_batch(wavs, wav_lens)
            predictions = encoder_decoder_asr.mods.decoder(enc, wav_lens)

            # prepare for metric reporting
            predicted = [wrd.split(" ") for wrd in predictions[0]]
            targeted = [wrd.split(" ") for wrd in batch.words]
            ids = batch.id
            for metric in reporting.keys():
                reporting[metric]["tracker"].append(
                    ids=ids, predict=predicted, target=targeted
                )

        # Report summary
        eval_reporting(reports=reporting)


if __name__ == "__main__":
    # CLI parsing
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # DDP init
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

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
    # * the tokenizer from URL - https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech/resolve/main/...
    # * the pretrained LM from HuggingFace - HF: speechbrain/asr-crdnn-rnnlm-librispeech
    # * the pretrained ASR from the local template checkpoint - local: speechbrain/asr-crdnn-rnnlm-librispeech
    run_on_main(
        hparams["pretrainer_tokenizer"].collect_files,
        kwargs={"fetch_from": FetchFrom.ONLINE},
    )
    run_on_main(
        hparams["pretrainer_LM"].collect_files,
        kwargs={"fetch_from": FetchFrom.HUGGING_FACE},
    )
    hparams["pretrainer_tokenizer"].load_collected(run_opts["device"])
    hparams["pretrainer_LM"].load_collected(run_opts["device"])
    # LOCAL fetching takes sources directly from their location
    hparams["pretrainer_ASR"].collect_files(fetch_from=FetchFrom.LOCAL)
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
        run_on_main(
            asr_brain.checkpointer.save_checkpoint, kwargs={"name": "latest"}
        )

    # Clean-up memory (for using EncoderDecoderASR interface only) but preserve testing-relevant objects
    test_datasets = deepcopy(datasets["test"])
    reporting = {
        "WER": {
            "field": "error_rate",
            "tracker": deepcopy(hparams["error_rate_computer"]()),
        },
        "CER": {
            "field": "error_rate",
            "tracker": deepcopy(hparams["cer_computer"]()),
        },
    }
    test_loader_kwargs = deepcopy(hparams["test_dataloader_opts"])
    del asr_brain, hparams, datasets

    # Pre-trained interface
    pretrained_asr = EncoderDecoderASR.from_hparams(
        source="source_pretrained",
        savedir="source_pretrained",
        hparams_file="pretrained.yaml",
        fetch_from=FetchFrom.LOCAL,
        run_opts=deepcopy(run_opts),
    )

    # Re:testing w/ previous dataloader
    run_on_main(
        eval_test_use_recipe_dataio,
        kwargs={
            "encoder_decoder_asr": pretrained_asr,
            "test_set": deepcopy(test_datasets),
        },
    )

    # Test w/ DDP
    run_on_main(
        eval_test_batch_from_scratch,
        kwargs={
            "encoder_decoder_asr": pretrained_asr,
            "test_set": deepcopy(test_datasets),
        },
    )

    # Test on both nodes w/ pretrained HF model; ensure first it's fetched
    repo = "speechbrain/asr-crdnn-rnnlm-librispeech"
    run_on_main(
        EncoderDecoderASR.from_hparams,
        kwargs={
            "source": repo,  # to default save_dir: pretrained_models/...
            "fetch_from": FetchFrom.HUGGING_FACE,  # needs to be here only bc speechbrain/asr-crdnn-rnnlm-librispeech exists
            "download_only": True,
        },
    )

    # Instantiate pretrained HF model
    pretrained_hf_asr = EncoderDecoderASR.from_hparams(
        source=repo,
        fetch_from=FetchFrom.HUGGING_FACE,
        run_opts=deepcopy(run_opts),
    )

    # From HF model card
    pretrained_hf_asr.transcribe_file(
        "speechbrain/asr-crdnn-rnnlm-librispeech/example.wav"
    )
