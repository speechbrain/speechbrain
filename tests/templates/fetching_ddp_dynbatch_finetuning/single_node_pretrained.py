#!/usr/bin/env/python3
"""Recipe for debugging/testing, based upon SpeechBrain's minilibrispeech ASR template.

Does the following feature set work out together on some environment?
    DDP; dynamic batching; fine-tuning; mixed pretrainer fetching & testing using pretrained interface

Authors:
    * Andreas Nautsch 2023
"""
import logging
import speechbrain as sb
from copy import deepcopy
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.pretrained.fetching import FetchFrom, FetchSource


logger = logging.getLogger(__name__)


def run_hf_repo_single_node(run_options):
    """Test function for basic HF model use.

    Arguments
    ---------

    run_options: dict
        A set of options to change the runtime environment.
    """
    # Test on both nodes w/ pretrained HF model; ensure first it's fetched
    repo = FetchSource(
        FetchFrom.HUGGING_FACE,  # needs to be here only bc local path "speechbrain/asr-crdnn-rnnlm-librispeech" exists
        "speechbrain/asr-crdnn-rnnlm-librispeech",
    )
    EncoderDecoderASR.from_hparams(
        source=repo,  # to default save_dir: pretrained_models/...
        download_only=True,
    )

    # Instantiate pretrained HF model
    pretrained_hf_asr = EncoderDecoderASR.from_hparams(
        source=repo, run_opts=deepcopy(run_options),
    )

    # From HF model card
    pred = pretrained_hf_asr.transcribe_file(
        FetchSource(
            FetchFrom.HUGGING_FACE,  # bc HF-alike local path structure exists
            "speechbrain/asr-crdnn-rnnlm-librispeech/example.wav",
        )
    )

    print(f"prediction: {pred}")


def run_hf_other_repo_single_node(run_options):
    """Test function for basic HF model use.

    Arguments
    ---------

    run_options: dict
        A set of options to change the runtime environment.
    """
    # Test on both nodes w/ pretrained HF model; ensure first it's fetched
    repo = "speechbrain/asr-crdnn-commonvoice-fr"
    EncoderDecoderASR.from_hparams(
        source=repo,
        savedir="pretrained_models/asr-crdnn-commonvoice-fr",
        download_only=True,
    )

    # Instantiate pretrained HF model
    pretrained_hf_asr = EncoderDecoderASR.from_hparams(
        source=repo,
        savedir="pretrained_models/asr-crdnn-commonvoice-fr",
        run_opts=deepcopy(run_options),
    )

    # From HF model card
    pred = pretrained_hf_asr.transcribe_file(
        "speechbrain/asr-crdnn-commonvoice-fr/example-fr.wav"
    )

    print(f"prediction: {pred}")


if __name__ == "__main__":
    # hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    _, run_opts, _ = sb.parse_arguments(
        ["finetune.yaml", "--device", "cpu"]
    )  # just some yaml, so we get device into run_opts
    run_hf_repo_single_node(run_opts)
    run_hf_other_repo_single_node(run_opts)
