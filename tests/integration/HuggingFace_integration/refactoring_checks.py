#!/usr/bin/env/python3
"""This is a test script for creating a list of expected outcomes (before refactoring);
then, manual editing might change YAMLs and/or code; another test runs to compare results
(after refactoring to before). The target is a list of known HF repos.

The goal is to identify to which extent changes break existing functionality.
Then, larger changes to code base can be rolled out more assured.

Tested with dependencies:
pip install huggingface_hub==0.10.1 datasets==2.6.1 transformers==4.23.1

dependencies as of: Oct 11, 2022; Oct 14, 2022; Oct 11, 2022.
"""

import os
import yaml
import torch  # noqa
import importlib  # noqa
import speechbrain  # noqa
from speechbrain.pretrained.interfaces import foreign_class  # noqa


# TODO fix
"""
    "speechbrain/ssl-wav2vec2-base-librispeech": {
        "sample": "example.wav",
        "cls": "WaveformEncoder",
        "fnx": "encode_file",
    },
    "speechbrain/asr-wav2vec2-commonvoice-fr": {
        "sample": "example-fr.wav",
        "cls": "EncoderASR",
        "fnx": "transcribe_batch",
    },
    "speechbrain/asr-wav2vec2-commonvoice-rw": {
        "sample": "example.mp3",
        "cls": "EncoderASR",
        "fnx": "transcribe_batch",
    },
    "speechbrain/emotion-recognition-wav2vec2-IEMOCAP": {
        "sample": "anger.wav",
        "cls": None,
        "fnx": "classify_file",
        "foreign": 'foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")',
        "prediction": 'model.classify_file("speechbrain/emotion-recognition-wav2vec2-IEMOCAP/anger.wav")',
    },
"""

WAV2_VEC2_HF_REPOS = {
    "speechbrain/asr-wav2vec2-transformer-aishell": {
        "sample": "example_mandarin.wav",
        "cls": "EncoderDecoderASR",
        "fnx": "transcribe_batch",
    },
    "speechbrain/asr-wav2vec2-ctc-aishell": {
        "sample": "anger.wav",
        "cls": None,
        "fnx": "classify_file",
        "foreign": 'foreign_class(source="speechbrain/asr-wav2vec2-ctc-aishell",  pymodule_file="custom_interface.py", classname="CustomEncoderDecoderASR")',
        "prediction": 'model.transcribe_file("speechbrain/asr-wav2vec2-ctc-aishell/example.wav")',
    },
    "speechbrain/asr-wav2vec2-librispeech": {
        "sample": "example.wav",
        "cls": "EncoderASR",
        "fnx": "transcribe_batch",
    },
    "speechbrain/asr-wav2vec2-commonvoice-en": {
        "sample": "example.wav",
        "cls": "EncoderDecoderASR",
        "fnx": "transcribe_batch",
    },
    "speechbrain/asr-wav2vec2-commonvoice-it": {
        "sample": "example-it.wav",
        "cls": "EncoderDecoderASR",
        "fnx": "transcribe_batch",
    },
    "speechbrain/asr-wav2vec2-dvoice-amharic": {
        "sample": "example_amharic.wav",
        "cls": "EncoderASR",
        "fnx": "transcribe_batch",
    },
    "speechbrain/asr-wav2vec2-dvoice-darija": {
        "sample": "example_darija.wav",
        "cls": "EncoderASR",
        "fnx": "transcribe_batch",
    },
    "speechbrain/asr-wav2vec2-dvoice-fongbe": {
        "sample": "example_fongbe.wav",
        "cls": "EncoderASR",
        "fnx": "transcribe_batch",
    },
    "speechbrain/asr-wav2vec2-dvoice-swahili": {
        "sample": "example_swahili.wav",
        "cls": "EncoderASR",
        "fnx": "transcribe_batch",
    },
    "speechbrain/asr-wav2vec2-dvoice-wolof": {
        "sample": "example_wolof.wav",
        "cls": "EncoderASR",
        "fnx": "transcribe_batch",
    },
}


def get_prediction(repo, values):
    # get the pretrained class; model & predictions
    if values["cls"] is not None:
        obj = eval(
            f'importlib.import_module("speechbrain.pretrained").{values["cls"]}'
        )
        model = obj.from_hparams(  # noqa
            source=repo,
            savedir=repo.replace("speechbrain", "pretrained_models"),
        )
        prediction = eval(
            f'model.{values["fnx"]}(model.load_audio("{repo}/{values["sample"]}").unsqueeze(0), torch.tensor([1.0]))'
        )
    else:
        model = eval(values["foreign"])  # noqa
        prediction = eval(values["prediction"])

    return [x[0] for x in prediction]


def gather_expected_results(
    yaml_path="tests/tmp/refactoring_wav2vec2_results.yaml",
):
    """Before refactoring HF YAMLs and/or code (regarding wav2vec2), gather prediction results.

    Parameters
    ----------
    yaml_path : str
        Path where to store/load refactoring testing results for later comparison.

    """
    # load results, if existing -or- new from scratch
    if os.path.exists(yaml_path):
        with open(yaml_path) as yaml_in:
            results = yaml.safe_load(yaml_in)
    else:
        results = {}

    # go through each repo
    for repo, values in WAV2_VEC2_HF_REPOS.items():
        # skip if results are there
        if repo not in results.keys():
            # continue when values are there
            assert type(values) == dict
            if all([k in values.keys() for k in ["sample", "cls", "fnx"]]):
                print(f"Collecting results for: {repo} w/ values={values}")
                prediction = get_prediction(repo, values)

                # extend the results
                results[repo] = {"before": prediction}
                with open(yaml_path, "w") as yaml_out:
                    yaml.dump(results, yaml_out, default_flow_style=None)


def gather_refactoring_results(
    yaml_path="tests/tmp/refactoring_wav2vec2_results.yaml",
):
    # expected results need to exist
    if os.path.exists(yaml_path):
        with open(yaml_path) as yaml_in:
            results = yaml.safe_load(yaml_in)

        # go through each repo
        for repo, values in WAV2_VEC2_HF_REPOS.items():
            # skip if results are there
            if repo in results.keys():
                if "after" not in results[repo]:
                    # continue when values are there
                    assert type(values) == dict
                    if all(
                        [k in values.keys() for k in ["sample", "cls", "fnx"]]
                    ):
                        print(
                            f"Collecting refactoring results for: {repo} w/ values={values}"
                        )

                        # extend the results
                        results[repo]["after"] = get_prediction(repo, values)
                        results[repo]["same"] = (
                            results[repo]["before"] == results[repo]["after"]
                        )

                        # update
                        with open(yaml_path, "w") as yaml_out:
                            yaml.dump(
                                results, yaml_out, default_flow_style=None
                            )

                        print(f"\tsame: {results[repo]['same'] }")
