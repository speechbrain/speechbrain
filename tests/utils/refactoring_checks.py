#!/usr/bin/env/python3
"""This is a test script for creating a list of expected outcomes (before refactoring);
then, manual editing might change YAMLs and/or code; another test runs to compare results
(after refactoring to before). The target is a list of known HF repos.

The goal is to identify to which extent changes break existing functionality.
Then, larger changes to code base can be rolled out more assured.

Authors
 * Andreas Nautsch, 2022, 2023
"""

import os
import sys
from tqdm import tqdm
import yaml
import torch  # noqa
import importlib  # noqa
import subprocess
import speechbrain  # noqa
from glob import glob
from copy import deepcopy
from torch.utils.data import DataLoader
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main  # noqa
from speechbrain.utils.train_logger import FileTrainLogger
from speechbrain.pretrained.interfaces import foreign_class  # noqa
from speechbrain.dataio.dataloader import LoopedLoader, make_dataloader


def init(new_interfaces_git, new_interfaces_branch, new_interfaces_local_dir):
    """Initialises a PR branch to: https://github.com/speechbrain/speechbrain/tree/hf-interface-testing

    Skip if the path as of `new_interfaces_local_dir` exists (e.g. by DIY init instead of via this script).

    Parameters
    ----------
    new_interfaces_git: str
        Your git repo (or default: `https://github.com/speechbrain/speechbrain`);
        can be specified in tests/utils/overrides.yaml
    new_interfaces_branch: str
        Default is `hf-interface-testing` (a git branch); can be specified in tests/utils/overrides.yaml
    new_interfaces_local_dir: str
        Default is `tests/tmp/hf_interfaces` (a local path); can be specified in tests/utils/overrides.yaml

    Returns
    -------
    str
        Local path of `updates_pretrained_models` where the update HF yaml/interface files can be found.
    """
    # set up git etc
    if not os.path.exists(new_interfaces_local_dir):
        # note: not checking for anything, whether it exists or not - or if there is a previous one already
        # clone repo with PR on updates_pretrained_models into local folder
        cmd_out_clone = subprocess.run(
            ["git", "clone", new_interfaces_git, new_interfaces_local_dir],
            capture_output=True,
        )
        print(f"\tgit clone log: {cmd_out_clone}")

        # cd into that local folder, switch branch to the one containing updates_pretrained_models & cd back
        cwd = os.getcwd()
        os.chdir(new_interfaces_local_dir)
        cmd_out_co = subprocess.run(
            ["git", "checkout", new_interfaces_branch], capture_output=True
        )
        print(f"\tgit checkout log: {cmd_out_co}")
        os.chdir(cwd)

    # return the valid local path with updates_pretrained_models
    updates_dir = f"{new_interfaces_local_dir}/updates_pretrained_models"
    return updates_dir


def get_model(repo, values, updates_dir=None, run_opts=None):
    """Fetches a pretrained model with the option the re-specify its hyperparameters & interface.

    Parameters
    ----------
    repo: str
        Source of pretrained model (assuming its within the HF speechbrain collection).
    values: dict
        Interface specification.
        Example: speechbrain:hf-interface-testing/updates_pretrained_models/ssl-wav2vec2-base-librispeech/test.yaml
    updates_dir: str
        Local folder with yaml:interface updates; None (default) = take original yaml/interface specification.
    run_opts: dict
        Run options, such as device

    Returns
    -------
    A pretrained model with a speechbrain.pretrained.interface or a custom interface.
    """
    # get the pretrained class; model & predictions
    kwargs = {
        "source": f"speechbrain/{repo}",
        "savedir": f"pretrained_models/{repo}",
    }

    # adjust symlinks
    hparams = f"pretrained_models/{repo}/hyperparams.yaml"
    if (
        "foreign" in values.keys()
    ):  # it's a custom model which has its own Python filename
        custom = f'pretrained_models/{repo}/{values["foreign"]}'
    # prepare model loading: is it the old -or- the new yaml/interface?
    if updates_dir is not None:
        # testing the refactoring; assuming all model data has been loaded already
        kwargs["source"] = f"{updates_dir}/{repo}"
        os.unlink(hparams)
        os.symlink(f"{updates_dir}/{repo}/hyperparams.yaml", hparams)
        if "foreign" in values.keys():
            os.unlink(custom)
            os.symlink(
                f'{updates_dir}/{repo}/{values["foreign"]}', custom,
            )
    else:
        # re:testing on develop? => simply unlink anything before and re:link from cached HF hub
        if os.path.exists(hparams):
            os.unlink(hparams)
        if "foreign" in values.keys():
            if os.path.exists(custom):
                os.unlink(custom)

    if run_opts is not None:
        kwargs["run_opts"] = run_opts

    print(f"\trepo: {repo}")
    # load pretrained model either via specified pretrained class or custom interface
    if "foreign" not in values.keys():
        print(f'\tspeechbrain.pretrained.{values["cls"]}')
        print(f"\tobj.from_hparams({kwargs})")
        obj = eval(f'speechbrain.pretrained.{values["cls"]}')
        model = obj.from_hparams(**kwargs)
    else:
        kwargs["pymodule_file"] = values["foreign"]
        kwargs["classname"] = values["cls"]
        model = foreign_class(**kwargs)

    return model


def get_prediction(repo, values, updates_dir=None):
    """Gets the prediction for one predefined audio example, pattern: {repo}/{values["sample"]} (see HF model card).

    Parameters
    ----------
    repo: str
        Source of pretrained model (assuming its within the HF speechbrain collection).
    values: dict
        Interface specification.
        Examples: speechbrain:hf-interface-testing/updates_pretrained_models/ssl-wav2vec2-base-librispeech/test.yaml
                  speechbrain:hf-interface-testing/updates_pretrained_models/asr-wav2vec2-librispeech/test.yaml
    updates_dir: str
        Controls whether/not we are in the refactored results (None: expected results; before refactoring).

    Returns
    -------
    Cleaned-up prediction results for yaml output (result logging & comparison through yaml de/serialization).
    """

    def sanitize(data):
        # cleanup data for yaml output (w/o this, yaml will make attempts to save torch/numpy arrays in their format)
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            if data.ndim:
                data = list(data)
        return data

    # get the pretrained model (before/after yaml/interface update)
    model = get_model(repo=repo, values=values, updates_dir=updates_dir)  # noqa

    try:
        # simulate batch from single file
        prediction = eval(
            f'model.{values["fnx"]}(model.load_audio("{repo}/{values["sample"]}", savedir="pretrained_models/{repo}").unsqueeze(0), torch.tensor([1.0]))'
        )

    except Exception:
        # use an example audio if no audio can be loaded
        print(f'\tWARNING - no audio found on HF: {repo}/{values["sample"]}')
        prediction = eval(
            f'model.{values["fnx"]}(model.load_audio("tests/samples/single-mic/example1.wav", savedir="pretrained_models/{repo}").unsqueeze(0), torch.tensor([1.0]))'
        )

    finally:
        del model

    return [sanitize(x[0]) for x in prediction]


def gather_expected_results(
    new_interfaces_git="https://github.com/speechbrain/speechbrain",
    new_interfaces_branch="testing-refactoring",
    new_interfaces_local_dir="tests/tmp/hf_interfaces",
    yaml_path="tests/tmp/refactoring_results.yaml",
):
    """Before refactoring HF YAMLs and/or code, gather prediction results.

    Parameters
    ----------
    new_interfaces_git: str
        Your git repo (or default: `https://github.com/speechbrain/speechbrain`);
        can be specified in tests/utils/overrides.yaml
    new_interfaces_branch: str
        Default is `hf-interface-testing` (a git branch); can be specified in tests/utils/overrides.yaml
    new_interfaces_local_dir: str
        Default is `tests/tmp/hf_interfaces` (a local path); can be specified in tests/utils/overrides.yaml
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
    updates_dir = init(
        new_interfaces_git, new_interfaces_branch, new_interfaces_local_dir
    )
    repos = map(os.path.basename, glob(f"{updates_dir}/*"))
    for repo in repos:
        # skip if results are there
        if repo not in results.keys():
            # get values
            with open(f"{updates_dir}/{repo}/test.yaml") as yaml_test:
                values = load_hyperpyyaml(yaml_test)

            print(f"Collecting results for: {repo} w/ values={values}")
            prediction = get_prediction(repo, values)

            # extend the results
            results[repo] = {"before": prediction}
            with open(yaml_path, "w") as yaml_out:
                yaml.dump(results, yaml_out, default_flow_style=None)


def gather_refactoring_results(
    new_interfaces_git="https://github.com/speechbrain/speechbrain",
    new_interfaces_branch="testing-refactoring",
    new_interfaces_local_dir="tests/tmp/hf_interfaces",
    yaml_path="tests/tmp/refactoring_results.yaml",
):
    """After refactoring HF YAMLs and/or code, gather prediction results.

    Parameters
    ----------
    new_interfaces_git: str
        Your git repo (or default: `https://github.com/speechbrain/speechbrain`);
        can be specified in tests/utils/overrides.yaml
    new_interfaces_branch: str
        Default is `hf-interface-testing` (a git branch); can be specified in tests/utils/overrides.yaml
    new_interfaces_local_dir: str
        Default is `tests/tmp/hf_interfaces` (a local path); can be specified in tests/utils/overrides.yaml
    yaml_path: str
        Path where to store/load refactoring testing results for later comparison.
    """
    # expected results need to exist
    if os.path.exists(yaml_path):
        with open(yaml_path) as yaml_in:
            results = yaml.safe_load(yaml_in)

    # go through each repo
    updates_dir = init(
        new_interfaces_git, new_interfaces_branch, new_interfaces_local_dir
    )
    repos = map(os.path.basename, glob(f"{updates_dir}/*"))
    for repo in repos:
        # skip if results are there
        if "after" not in results[repo].keys():
            # get values
            with open(f"{updates_dir}/{repo}/test.yaml") as yaml_test:
                values = load_hyperpyyaml(yaml_test)

            print(
                f"Collecting refactoring results for: {repo} w/ values={values}"
            )

            # extend the results
            results[repo]["after"] = get_prediction(repo, values, updates_dir)
            results[repo]["same"] = (
                results[repo]["before"] == results[repo]["after"]
            )

            # update
            with open(yaml_path, "w") as yaml_out:
                yaml.dump(results, yaml_out, default_flow_style=None)

            print(f"\tsame: {results[repo]['same'] }")


def test_performance(
    repo, values, run_opts, updates_dir=None, recipe_overrides={}
):
    """

    Parameters
    ----------
    repo: str
        Source of pretrained model (assuming its within the HF speechbrain collection).
    values: dict
        Interface specification.
        Examples: speechbrain:hf-interface-testing/updates_pretrained_models/ssl-wav2vec2-base-librispeech/test.yaml
                  speechbrain:hf-interface-testing/updates_pretrained_models/asr-wav2vec2-librispeech/test.yaml
    run_opts: dict
        Run options, such as device
    updates_dir: str
        Controls whether/not we are in the refactored results (None: expected results; before refactoring).
    recipe_overrides: dict
        Recipe YAMLs contain placeholders and flags which need to be overwritten (e.g. data_folder & skip_prep).
        See: overrides.yaml

    Returns
    -------
    Dict for export to yaml with performance statistics, as specified in the test.yaml files.
    """
    # Dataset depending file structure
    tmp_dir = f'tests/tmp/{values["dataset"]}'
    speechbrain.create_experiment_directory(experiment_directory=tmp_dir)
    stats_meta = {
        f'[{values["dataset"]}] - {"BEFORE" if updates_dir is None else "AFTER"}': repo
    }

    # Load pretrained
    model = get_model(
        repo=repo, values=values, updates_dir=updates_dir, run_opts=run_opts
    )  # noqa

    # Dataio preparation; we need the test sets only
    with open(values["recipe_yaml"]) as fin:
        recipe_hparams = load_hyperpyyaml(
            fin, values["overrides"] | recipe_overrides
        )

    # Dataset preparation is assumed to be done through recipes; before running this.
    exec(values["dataio"])
    test_datasets = deepcopy(eval(values["test_datasets"]))

    # harmonise
    if type(test_datasets) is not dict:
        tmp = {}
        if type(test_datasets) is list:
            for i, x in enumerate(test_datasets):
                tmp[i] = x
        else:
            tmp[0] = test_datasets
        test_datasets = tmp

    # prepare testing
    logger = FileTrainLogger(save_file=f"{tmp_dir}/{repo}.log")
    reporting = deepcopy(values["performance"])
    for metric, specs in reporting.items():
        reporting[metric]["tracker"] = deepcopy(
            recipe_hparams[specs["handler"]]()
        )

    test_loader_kwargs = deepcopy(recipe_hparams[values["test_loader"]])
    del recipe_hparams

    stats = {}
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        test_set = test_datasets[k]
        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_set = make_dataloader(test_set, **test_loader_kwargs)

        with torch.no_grad():
            for batch in tqdm(test_set, dynamic_ncols=True, disable=False):
                batch = batch.to(model.device)
                wavs, wav_lens = batch.sig
                wavs, wav_lens = (  # noqa
                    wavs.to(model.device),
                    wav_lens.to(model.device),
                )
                predictions = eval(  # noqa
                    f'model.{values["fnx"]}(wavs, wav_lens)'
                )
                predicted = eval(values["predicted"])  # noqa
                targeted = eval(values["targeted"])  # noqa
                ids = batch.id  # noqa
                for metric in reporting.keys():
                    reporting[metric]["tracker"].append(*eval(values["append"]))

        stats[k] = {}
        for metric, specs in reporting.items():
            stats[k][metric] = specs["tracker"].summarize(specs["field"])
        logger.log_stats(
            stats_meta=stats_meta | {"set": k}, test_stats=stats[k],
        )

    return stats


# run first w/ "--after=False" on latest develop, then checkout the refactoring branch and run w/ "--after=True"
# PYTHONPATH=`realpath .` python tests/utils/refactoring_checks.py tests/utils/overrides.yaml --LibriSpeech_data="" --CommonVoice_EN_data="" --CommonVoice_FR_data="" --IEMOCAP_data="" --after=False
if __name__ == "__main__":
    hparams_file, run_opts, overrides = speechbrain.parse_arguments(
        sys.argv[1:]
    )

    with open(hparams_file) as fin:
        dataset_overrides = load_hyperpyyaml(fin, overrides)

    # go through each repo
    updates_dir = init(
        dataset_overrides["new_interfaces_git"],
        dataset_overrides["new_interfaces_branch"],
        dataset_overrides["new_interfaces_local_dir"],
    )

    # load results, if existing -or- new from scratch
    yaml_path = f'{dataset_overrides["new_interfaces_local_dir"]}.yaml'
    if os.path.exists(yaml_path):
        with open(yaml_path) as yaml_in:
            results = yaml.safe_load(yaml_in)
    else:
        results = {}

    repos = map(
        os.path.basename,
        glob(f'{updates_dir}/{dataset_overrides["glob_filter"]}'),
    )
    for repo in repos:
        # get values
        with open(f"{updates_dir}/{repo}/test.yaml") as yaml_test:
            values = load_hyperpyyaml(yaml_test)

        # for this testing, some fields need to exist; skip otherwise
        if any(
            [
                entry not in values
                for entry in [
                    "dataset",
                    "overrides",
                    "dataio",
                    "test_datasets",
                    "test_loader",
                    "performance",
                    "predicted",
                ]
            ]
        ):
            continue

        # skip if datasets is not given
        if not dataset_overrides[f'{values["dataset"]}_data']:
            continue

        print(f"Run tests on: {repo}")
        if repo not in results.keys():
            results[repo] = {}

        # Before refactoring
        if "before" not in results[repo].keys():
            results[repo]["before"] = test_performance(
                repo,
                values,
                updates_dir=None,
                run_opts=run_opts,
                recipe_overrides=dataset_overrides[values["dataset"]],
            )

            # update
            with open(yaml_path, "w") as yaml_out:
                yaml.dump(results, yaml_out, default_flow_style=None)

        # After refactoring
        if (
            "after" not in results[repo].keys()
            and dataset_overrides["after"] is True
        ):
            results[repo]["after"] = test_performance(
                repo,
                values,
                run_opts=run_opts,
                updates_dir=updates_dir,
                recipe_overrides=dataset_overrides[values["dataset"]],
            )

            results[repo]["same"] = (
                results[repo]["before"] == results[repo]["after"]
            )
            print(f'\tbefore: {results[repo]["before"]}')
            print(f'\t after: {results[repo]["after"]}')
            print(f'\t  same: {results[repo]["same"]}')

            # update
            with open(yaml_path, "w") as yaml_out:
                yaml.dump(results, yaml_out, default_flow_style=None)

    # update
    with open(yaml_path, "w") as yaml_out:
        yaml.dump(results, yaml_out, default_flow_style=None)
