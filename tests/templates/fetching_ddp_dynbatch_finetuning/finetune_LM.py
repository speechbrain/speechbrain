#!/usr/bin/env/python3
"""Recipe for debugging/testing, based upon SpeechBrain's minilibrispeech ASR template.

Does the following feature set work out together on some environment?
    DDP; dynamic batching; fine-tuning; mixed pretrainer fetching & testing using pretrained interface

Authors:
    * Andreas Nautsch 2023
"""
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
from ASR_template_train import ASR, dataio_prepare


logger = logging.getLogger(__name__)


def eval_reporting(set_k, reports):
    for log_metric, specs in reports.items():
        logger.log_stats(
            stats_meta={"set": set_k, "metric": log_metric},
            test_stats=specs["tracker"].summarize(specs["field"]),
        )


def lm_compute_forward(self, batch, stage):
    """From templates/speech_recognition/ASR/train.py:compute_forward

    Uses lm_model & S2SRNNBeamSearchLM directly.
    """
    # We first move the batch to the appropriate device.
    batch = batch.to(self.device)
    feats, self.feat_lens = self.prepare_features(stage, batch.sig)
    tokens_bos, _ = self.prepare_tokens(stage, batch.tokens_bos)

    # Running the encoder (prevent propagation to feature extraction)
    encoded_signal = self.modules.encoder(feats.detach())

    # Embed tokens and pass tokens & encoded signal to decoder
    embedded_tokens = self.modules.embedding(tokens_bos)
    decoder_outputs, _ = self.modules.decoder(
        embedded_tokens, encoded_signal, self.feat_lens
    )

    # Output layer for seq2seq log-probabilities
    predictions = {}
    if self.is_ctc_active(stage):
        ctc_logits = self.modules.ctc_lin(encoded_signal)
        predictions["ctc_logprobs"] = self.hparams.log_softmax(ctc_logits)
        # TODO check if ctc_lin isn't triggered twice ... if so, freeze & unfreeze it's parameters?
    (
        predictions["tokens"],
        _,
        predictions["seq_logprobs"],
    ) = self.hparams.train_valid_test_search(encoded_signal, self.feat_lens)

    return predictions


if __name__ == "__main__":
    # CLI parsing
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print(hparams_file)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # DDP init
    sb.utils.distributed.ddp_init_group(run_opts)

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
            length_func=lambda x: x["duration"],
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
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        ),
        "num_workers": valid_dataloader_opts["num_workers"],
    }

    # We download:
    # * the pretrained ASR from the local template checkpoint - local: speechbrain/asr-crdnn-rnnlm-librispeech
    # * the pretrained LM from HuggingFace - HF: speechbrain/asr-crdnn-rnnlm-librispeech
    # * the tokenizer from URL - https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech/resolve/main/...
    run_on_main(
        hparams["pretrainer_tokenizer"].collect_files,
        kwargs={"fetch_from": FetchFrom.LOCAL},
    )
    run_on_main(
        hparams["pretrainer_LM"].collect_files,
        kwargs={"fetch_from": FetchFrom.HUGGING_FACE},
    )
    run_on_main(
        hparams["pretrainer_ASR"].collect_files,
        kwargs={"fetch_from": FetchFrom.ONLINE},
    )
    hparams["pretrainer_tokenizer"].load_collected(run_opts["device"])
    hparams["pretrainer_LM"].load_collected(run_opts["device"])
    hparams["pretrainer_ASR"].load_collected(run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Substitute: compute_forward & ensure return_log_probs for train_valid_test_search decoder
    setattr(asr_brain, "compute_forward", lm_compute_forward)
    asr_brain.hparams.train_valid_test_search.return_log_probs = True

    # Freeze all but LM
    for mod in [
        "encoder",
        "embedding",
        "decoder",
    ]:  # ctc_lin & seq_lin are part of the asr_model, but cheap to train
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
    test_stats = asr_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    # Clean-up memory (for using EncoderDecoderASR interface only) but preserve testing-relevant objects
    test_datasets = deepcopy(datasets["test"])
    reporting = {
        "WER": {
            "field": "error_rate",
            "tracker": deepcopy(hparams["error_rate_computer"]),
        },
        "CER": {
            "field": "error_rate",
            "tracker": deepcopy(hparams["cer_computer"]),
        },
    }
    test_loader_kwargs = deepcopy(hparams["test_dataloader_opts"])
    del asr_brain, hparams, datasets

    # Pre-trained interface
    pretrained_asr = EncoderDecoderASR.from_hparams(
        source="pretrained.yaml",
        savedir="pretrained_models",
        run_opts={"device": run_opts["device"]},
    )

    # Re:testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        test_set = test_datasets[k]
        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_set = make_dataloader(test_set, **test_loader_kwargs)

        with torch.no_grad():
            for batch in tqdm(test_set, dynamic_ncols=True, disable=False):
                batch = batch.to(pretrained_asr.device)
                # instead of using batch.sig, we desire to see pretrained_asr.load_audio in action
                wavs = []
                for audio in batch.wav:
                    wavs.append(pretrained_asr.load_audio(path=audio))
                wavs, wav_lens = batch_pad_right(wavs)
                wavs, wav_lens = (
                    wavs.to(pretrained_asr.device),
                    wav_lens.to(pretrained_asr.device),
                )
                predictions = pretrained_asr.classify_batch(wavs, wav_lens)
                predicted = [wrd.split(" ") for wrd in predictions[0]]
                targeted = [wrd.split(" ") for wrd in batch.wrd]
                ids = batch.id
                for metric in reporting.keys():
                    reporting[metric]["tracker"].append(
                        ids=ids, predict=predicted, target=targeted
                    )

            # Reporting summary on main process
            run_on_main(
                eval_reporting, kwargs={"set_k": k, "reports": reporting}
            )
