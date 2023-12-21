"""Library for running recipe tests.

Authors
 * Mirco Ravanelli 2022
 * Andreas Nautsch 2022, 2023
"""
import os
import re
import csv
import sys
import pydoc
from time import time
import subprocess as sp
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import download_file  # noqa: F401

__skip_list = ["README.md", "setup"]


def check_row_for_test(row, filters_fields, filters, test_field):
    """Checks if the current row of the csv recipe file has a test to run.

    Arguments
    ---------
    row: dict
        Line of the csv file (in dict from).
    filters_fields: list
        This can be used with the "filter" variable
        to run only some tests. For instance, filters_fields=['Task'] and filters=['ASR'])
        will only run tests for ASR recipes.
    filters: list
        See above.
    test_field: string
        Key of the input dictionary that contains the test flags.

    Returns
    ---------
    test: bool
        True if the line must be tested, False otherwise.
    """
    test = True
    for i, field in enumerate(filters_fields):
        field_values = filters[i]
        if type(field_values) == str:
            # ... AND ... filter
            if not (field_values == row[field]):
                test = False
        elif type(field_values) == list:  # type(field) == list
            # ... AND (... OR ...) ... filter; at least one entry of the list matches
            test_flag = False
            for filt in field_values:
                if filt == row[field]:
                    test_flag = True
            test = test and test_flag
        else:
            print("\tError in filters_fields and filters definition.")
            test = False

    if test:
        test_flag = row[test_field].strip()
        if len(test_flag) == 0:
            test = False
    return test


def prepare_test(
    recipe_folder="tests/recipes",
    script_field="Script_file",
    hparam_field="Hparam_file",
    test_field="test_debug_flags",
    check_field="test_debug_checks",
    download_field="test_download",
    message_field="test_message",
    filters_fields=[],
    filters=[],
):
    """Extracts all the needed information to run the recipe test.

    Arguments
    ---------
    recipe_folder: path
        Path of the folder containing csv recipe files summarizing all the recipes in the repo.
    script_field: str
        Field of the csv recipe file containing the path of the script to run.
    hparam_field: str
        Field of the csv recipe file containing the path of the hparam file.
    test_field: string
        Field of the csv recipe file containing the test flags.
    check_field: string
        Field of the csv recipe file containing the checks to perform.
    download_field: string
        Field of the csv recipe file containing files or folders to download for
        each test (optional).
    message_field: string
        Field of the csv recipe file containing optional messages to show before running the test.
    filters_fields: list
        This can be used with the "filter" variable
        to run only some tests. For instance, filters_fileds=['Task'] and filters=['ASR'])
        will only run tests for ASR recipes.
    filters: list
        See above.

    Returns
    ---------
    test_script: dict
        A Dictionary containing recipe IDs as keys and test_scripts as values.
    test_hparam: dict
        A dictionary containing recipe IDs as keys and hparams as values.
    test_flag: dict
        A dictionary containing recipe IDs as keys and the test flags as values.
    test_check: dict
        A dictionary containing recipe IDs as keys and the checks as values.
    test_download: dict
        A dictionary containing recipe IDs as keys and the checks as values.
    test_message: dict
        A dictionary containing recipe IDs as keys and the checks as values.
    """

    # Dictionary initialization
    test_script = {}
    test_hparam = {}
    test_flag = {}
    test_check = {}
    test_download = {}
    test_message = {}

    # Loop over all recipe CSVs
    print(f"\tfilters_fields={filters_fields} => filters={filters}")
    for recipe_csvfile in os.listdir(recipe_folder):
        # skip setup scripts; consider CSV files only
        if recipe_csvfile in __skip_list:
            continue

        print(f"Loading recipes from: {recipe_csvfile}")
        # Detect needed information for the recipe tests
        with open(
            os.path.join(recipe_folder, recipe_csvfile), newline=""
        ) as csvf:
            reader = csv.DictReader(csvf, delimiter=",", skipinitialspace=True)
            for row_id, row in enumerate(reader):
                recipe_id = f"{recipe_csvfile[:-4]}_row_{row_id+2:02d}"
                if not (
                    check_row_for_test(row, filters_fields, filters, test_field)
                ):
                    print(f"\tSkipped {recipe_id}")
                    continue
                test_script[recipe_id] = row[script_field].strip()
                test_hparam[recipe_id] = row[hparam_field].strip()
                test_flag[recipe_id] = row[test_field].strip()
                test_check[recipe_id] = row[check_field].strip()

                # Manage test_download (optional field)
                if download_field in row:
                    test_download[recipe_id] = row[download_field].strip()
                if message_field in row:
                    test_message[recipe_id] = row[message_field].strip()

    return (
        test_script,
        test_hparam,
        test_flag,
        test_check,
        test_download,
        test_message,
    )


def check_files(
    check_str, output_folder, recipe_id, pattern=r"file_exists=\[(.*?)\]"
):
    """Checks if the output folder created by the test has the expected files.

    Arguments
    ---------
    check_str: str
        String summarizing the checks to perform.
    output_folder: path
        The path where to check the files.
    recipe_id: str
        Unique ID of the recipe.
    pattern: str
        The pattern used to extract the list of files to check from check_str.

    Returns
    ---------
    check: bool
        True if all the files are found, False otherwise.
    """

    check = True
    files_to_check = re.search(pattern, check_str)
    files_to_check = files_to_check.group(1).split(",")

    for file_to_check in files_to_check:
        check_path = os.path.join(output_folder, file_to_check)
        if not (os.path.exists(check_path)):
            print(
                "\tERROR: The recipe %s does not contain the expected file %s"
                % (recipe_id, check_path)
            )
            check = False
    return check


def check_performance(
    check_str, output_folder, recipe_id, pattern=r"performance_check=\[(.*?)\]"
):
    """Checks if the performance achieved by the recipe matches with the
    expectations. This is done by adding a performance_check entry in the recipe
    check field of the csv recipe file
    For instance: performance_check=[train_log.txt, train loss, <=15, epoch: 2]),
    will check the variable "train_loss" in the train_log.txt at epoch 2. It will
    raise an error if the train_loss is >15. If epoch is -1, we check the last
    line of the file.

    Arguments
    ---------
    check_str: str
        String summarizing the checks to perform.
    output_folder: path
        The path where the recipe files are stored.
    recipe_id: str
        Unique ID of the recipe.
    pattern: str
        The pattern used to extract the list of files to check from check_str.

    Returns
    ---------
    check: bool
        True if all the files are found, False otherwise.
    """

    check = True
    performance_to_check = re.search(pattern, check_str)
    if performance_to_check is None:
        return check

    # Getting the needed information from the "performance_check" entry
    performance_to_check = performance_to_check.group(1).split(",")
    filename = performance_to_check[0].strip()
    filename = os.path.join(output_folder, filename)
    variable = performance_to_check[1].strip()
    threshold = performance_to_check[2].strip()
    epoch = performance_to_check[3].strip()

    if not (os.path.exists(filename)):
        print(
            "\tERROR: The file %s of recipe %s does not exist (needed for performance checks)"
            % (filename, recipe_id)
        )

        return False

    # Real all the lines of the performance file
    with open(filename) as file:
        lines = file.readlines()

    # Fitler the lines
    lines_filt = []
    last_line = ""
    for line in lines:
        if len(line.strip()) > 0:
            last_line = line
        if epoch in line:
            lines_filt.append(line)

    # If epoch: -1, we take the last line of the file
    epoch_id = int(epoch.split(":")[-1].strip())
    if epoch_id == -1:
        lines_filt.append(last_line)

    # Raising an error if there are no lines after applying the filter
    if len(lines_filt) == 0:
        print(
            "\tERROR: No entries %s in %s (recipe %s). See performance_check entry."
            % (epoch, filename, recipe_id)
        )
        return False

    for line in lines_filt:

        # Search variable value
        var_value = extract_value(line, variable)

        if var_value is None:
            print(
                "\tERROR: The file %s of recipe %s does not contain the variable %s (needed for performance checks)"
                % (filename, recipe_id, variable)
            )
            return False
        var_value = float(var_value)
        check = check_threshold(threshold, var_value)

        if not (check):
            print(
                "\tERROR: The variable %s of file %s (recipe %s) violated the specified threshold (%s %s)"
                % (variable, filename, recipe_id, var_value, threshold)
            )

        break

    return check


def extract_value(string, key):
    """Extracts from the input string the value given a key. For instance:
    input_string = "Epoch loaded: 49 - test loss: 4.71e-01, test PER: 14.21"
    print(extract_value(input_string, "test loss"))    # Output: 0.471
    print(extract_value(input_string, "test PER"))     # Output: 14.21

    Arguments
    ---------
    string: str
        The input string. It should be in the format mentioned above.
    key: str
        The key argument to extract.

    Returns
    ---------
    value: float or str
        The value corresponding to the specified key.
    """
    escaped_key = re.escape(key)

    # Create the regular expression pattern to match the argument and its corresponding value
    pattern = r"(?P<key>{})\s*:\s*(?P<value>[-+]?\d*\.\d+([eE][-+]?\d+)?)".format(
        escaped_key
    )

    # Search for the pattern in the input string
    match = re.search(pattern, string)

    if match:
        value = match.group("value")
        return value
    else:
        return None


def check_threshold(threshold, value):
    """Checks if the value satisfied the threshold constraints.

    Arguments
    ---------
    threshold: str
        String that contains the contains. E.g, ">=10" or "==15" or "<5".
    value: float
        Float corresponding to the value to test

    Returns
    ---------
    bool
        True if the constraint is satisfied, False otherwise.
    """

    # Get threshold value:
    th_value = float(
        threshold.strip().replace("=", "").replace(">", "").replace("<", "")
    )

    # Check Threshold
    if "==" in threshold:
        return value == th_value

    elif ">=" in threshold:
        return value >= th_value

    elif ">" in threshold:
        return value > th_value

    elif "<=" in threshold:
        return value <= th_value

    elif "<" in threshold:
        return value < th_value
    else:
        return False


def run_test_cmd(cmd, stdout_file, stderr_file):
    """Runs the command corresponding to a recipe test. The standard output and
    the standard error is saved in the specified paths.

    Arguments
    ---------
    cmd: str
        String corresponding to the command to run.
    stdout_file: path
        File where standard output is stored.
    stderr_file: path
        File where standard error is stored.

    Returns
    ---------
    rc: bool
        The return code obtained after running the command. If 0, the test is
        run without errors. If >0 the execution failed.
    """
    f_stdout = open(stdout_file, "w")
    f_stderr = open(stderr_file, "w")
    child = sp.Popen([cmd], stdout=f_stdout, stderr=f_stderr, shell=True)
    child.communicate()[0]
    rc = child.returncode
    f_stdout.close()
    f_stderr.close()
    return rc


def run_recipe_tests(
    recipe_folder="tests/recipes",
    script_field="Script_file",
    hparam_field="Hparam_file",
    test_field="test_debug_flags",
    check_field="test_debug_checks",
    run_opts="--device=cpu",
    output_folder="tests/tmp/",
    filters_fields=[],
    filters=[],
    do_checks=True,
    download_only=False,
    run_tests_with_checks_only=False,
):
    """Runs the recipes tests.

    Arguments
    ---------
    recipe_folder: path
        Path of the folder containing csv recipe files summarizing all the recipes in the repo.
    script_field: str
        Field of the csv recipe file containing the path of the script to run.
    hparam_field: str
        Field of the csv recipe file containing the path of the hparam file.
    test_field: string
        Field of the csv recipe file containing the test flags.
    check_field: string
        Field of the csv recipe file containing the checks to perform.
    run_opts: string
        Additional flags to add for the tests (see run_opts of speechbrain/core.py).
    output_folder: string
        Folder where the output of the tests are saved.
    filters_fields: list
        This can be used with the "filter" variable
        to run only some tests. For instance, filters_fileds=['Task'] and filters=['ASR'])
        will only run tests for ASR recipes.
    filters: list
        See above.
    do_checks: bool (default: True)
        If True performs the checks on the output folder (when the check_field is not empty).
    download_only: bool (default: False)
        If True skips running/checking tests after downloading relevant pre-trained data (prepare for offline testing).
    run_tests_with_checks_only: bool (default: False)
        If True skips all tests that do not have performance check criteria defined.

    Returns
    -------
    check: True
        True if all the recipe tests pass, False otherwise.

    Example
    -------
    python -c 'from speechbrain.utils.recipe_tests import run_recipe_tests; print("TEST FAILED!") if not(run_recipe_tests(filters_fields=["Dataset", "Task"], filters=[["AISHELL-1", "CommonVoice"], "SSL"])) else print("TEST PASSED")'
    """
    # Create the output folder (where the tests results will be saved)
    os.makedirs(output_folder, exist_ok=True)
    print("Test ouputs will be put in %s" % (output_folder))

    # Read the csv recipe file and detect which tests we have to run
    (
        test_script,
        test_hparam,
        test_flag,
        test_check,
        test_download,
        test_message,
    ) = prepare_test(
        recipe_folder,
        script_field,
        hparam_field,
        test_field=test_field,
        check_field=check_field,
        filters_fields=filters_fields,
        filters=filters,
    )

    # Early stop if there are no recipes to test
    if len(test_script) == 0:
        print("No recipes found for testing (please check recipe filters).")
        return False

    # Download all upfront
    if download_only:
        download_only_test(
            test_script,
            test_hparam,
            test_flag,
            test_check,
            run_opts,
            run_tests_with_checks_only,
            output_folder,
        )
        return False

    # Run  script (check how to get std out, std err and save them in files)
    check = True
    for i, recipe_id in enumerate(sorted(test_script.keys())):

        # Check if the output folder is specified in test_field
        spec_outfold = False
        if "--output_folder" in test_flag[recipe_id]:
            pattern = r"--output_folder\s*=?\s*([^\s']+|'[^']*')"
            match = re.search(pattern, test_flag[recipe_id])
            output_fold = match.group(1).strip("'")
            spec_outfold = True
        else:
            output_fold = os.path.join(output_folder, recipe_id)
            os.makedirs(output_fold, exist_ok=True)

        # Create files for storing standard input and standard output
        stdout_file = os.path.join(output_fold, "stdout.txt")
        stderr_file = os.path.join(output_fold, "stderr.txt")

        # If we are interested in performance checks only, skip
        check_str = test_check[recipe_id].strip()
        if run_tests_with_checks_only:
            if len(check_str) == 0:
                continue

        print(
            "(%i/%i) Running test for %s..."
            % (i + 1, len(test_script.keys()), recipe_id)
        )

        if recipe_id in test_download:
            download_cmds = test_download[recipe_id].split(";")
            for download_cmd in download_cmds:
                print("\t" + download_cmd)
                eval(download_cmd)

        # Check for setup scripts
        setup_script = os.path.join(
            "tests/recipes/setup",
            test_script[recipe_id][:-3].replace("/", "_"),
            test_hparam[recipe_id]
            .replace(os.path.dirname(test_script[recipe_id]), "")[1:-5]
            .replace("/", "_"),
        )
        if os.path.exists(setup_script):
            os.system(setup_script)

        # Composing command to run
        cmd = (
            f"PYTHONPATH={os.getcwd() + '/' + os.path.dirname(test_script[recipe_id])} python "
            + test_script[recipe_id]
            + " "
            + test_hparam[recipe_id]
            + " "
            + test_flag[recipe_id]
            + " "
            + run_opts
        )

        if not spec_outfold:
            cmd = cmd + " --output_folder=" + output_fold

        # add --debug if no do_checks to save testing time
        if not do_checks:
            cmd += " --debug --debug_persistently"

        # Print message (if any)
        if recipe_id in test_message:
            print("\t\t" + test_message[recipe_id])

        # Running the test
        time_start = time()
        return_code = run_test_cmd(cmd, stdout_file, stderr_file)
        test_duration = time() - time_start
        print("\t... %.2fs" % test_duration)

        # Tear down
        td_script = os.path.join(os.path.dirname(setup_script), "tear_down")
        if os.path.exists(td_script):
            os.system(td_script)

        # Check return code
        if return_code != 0:
            print(
                "\tERROR: Error in %s (%s). Check %s and %s for more info."
                % (recipe_id, test_hparam[recipe_id], stderr_file, stdout_file)
            )
            check = False

        # Checks
        if do_checks and len(check_str) > 0:
            print("\t...checking files & performance...")

            # Check if the expected files exist
            check &= check_files(check_str, output_fold, recipe_id)
            check &= check_performance(check_str, output_fold, recipe_id)

    return check


def download_only_test(
    test_script,
    test_hparam,
    test_flag,
    test_check,
    run_opts,
    run_tests_with_checks_only,
    output_folder,
):
    """Downloads only the needed data (useful for off-line tests).

    Arguments
    ---------
    test_script: dict
        A Dictionary containing recipe IDs as keys and test_scripts as values.
    test_hparam: dict
        A dictionary containing recipe IDs as keys and hparams as values.
    test_flag: dict
        A dictionary containing recipe IDs as keys and the test flags as values.
    test_check: dict
        A dictionary containing recipe IDs as keys and the checks as values.
    run_opts: str
        Running options to append to each test.
    run_tests_with_checks_only: str
            Running options to append to each test.
    run_tests_with_checks_only: bool
        If True skips all tests that do not have performance check criteria defined.
    output_folder: path
        The output folder where to store all the test outputs.
    """

    for i, recipe_id in enumerate(test_script.keys()):
        # If we are interested in performance checks only, skip
        check_str = test_check[recipe_id].strip()
        if run_tests_with_checks_only:
            if len(check_str) == 0:
                continue

        print(
            "(%i/%i) Collecting pretrained models for %s..."
            % (i + 1, len(test_script.keys()), recipe_id)
        )

        output_fold = os.path.join(output_folder, recipe_id)
        os.makedirs(output_fold, exist_ok=True)
        stdout_file = os.path.join(output_fold, "stdout.txt")
        stderr_file = os.path.join(output_fold, "stderr.txt")

        cmd = (
            "python -c 'import sys;from hyperpyyaml import load_hyperpyyaml;import speechbrain;"
            "hparams_file, run_opts, overrides = speechbrain.parse_arguments(sys.argv[1:]);"
            "fin=open(hparams_file);hparams = load_hyperpyyaml(fin, overrides);fin.close();"
            # 'speechbrain.create_experiment_directory(experiment_directory=hparams["output_folder"],'
            # 'hyperparams_to_save=hparams_file,overrides=overrides,);'
        )
        with open(test_hparam[recipe_id]) as hparam_file:
            for line in hparam_file:
                if "pretrainer" in line:
                    cmd += 'hparams["pretrainer"].collect_files();hparams["pretrainer"].load_collected(device="cpu");'
                elif "from_pretrained" in line:
                    field = line.split(":")[0].strip()
                    cmd += f'hparams["{field}"]'
        cmd += (
            "' "
            + test_hparam[recipe_id]
            + " --output_folder="
            + output_fold
            + " "
            + test_flag[recipe_id]
            + " "
            + run_opts
        )

        # Prepare the test
        run_test_cmd(cmd, stdout_file, stderr_file)


def load_yaml_test(
    recipe_folder="tests/recipes",
    script_field="Script_file",
    hparam_field="Hparam_file",
    test_field="test_debug_flags",
    filters_fields=[],
    filters=[],
    avoid_list=[],
    rir_folder="tests/tmp/rir",
    data_folder="tests/tmp/yaml",
    output_folder="tests/tmp/",
):
    """Tests if the yaml files can be loaded without errors.

    Arguments
    ---------
    recipe_folder: path
        Path of the folder containing csv recipe files summarizing all the recipes in the repo.
    script_field: str
        Field of the csv recipe file containing the path of the script to run.
    hparam_field: str
        Field of the csv recipe file containing the path of the hparam file.
    test_field: string
        Field of the csv recipe file containing the test flags.
    filters_fields: list
        This can be used with the "filter" variable
        to run only some tests. For instance, filters_fileds=['Task'] and filters=['ASR'])
        will only run tests for ASR recipes.
    filters: list
        See above.
    avoid_list: list
        List of hparam file not to check.
    rir_folder:
        This overrides the rir_folder; rir_path, and openrir_folder usually specified in the hparam files.
    data_folder:
        This overrides the data_folder usually specified in the hparam files.
    output_folder:
        This overrides the output_folder usually specified in the hparam files.

    Returns
    ---------
    check: True
        True if all the hparam files are loaded correctly, False otherwise.
    """

    # Get current working directory
    cwd = os.getcwd()

    # Set data_foler and output folder
    data_folder = os.path.join(cwd, data_folder)
    output_folder = os.path.join(cwd, output_folder)
    rir_folder = os.path.join(cwd, rir_folder)

    # Additional overrides
    add_overrides = {
        "manual_annot_folder": data_folder,
        "musan_folder": data_folder,
        "tea_models_dir": data_folder,
        "wsj_root": data_folder,
        "tokenizer_file": data_folder,
        "commonlanguage_folder": data_folder,
        "tea_infer_dir": data_folder,
        "original_data_folder": data_folder,
        "pretrain_st_dir": data_folder,
        # RIR folder specifications -> all point to the same zip file: one download destination
        "rir_path": rir_folder,
        "rir_folder": rir_folder,
        "openrir_folder": rir_folder,
        "open_rir_folder": rir_folder,
        "data_folder_rirs": rir_folder,
    }

    # Additional overrides when extra !PLACEHOLDER are encountered (not: data_folder - output_folder)
    add_placeholder_overrides = {
        "wav2vec2_hub": "facebook/wav2vec2-large-960h-lv60-self",  # this might not hold for all set-ups
        "root_data_folder": data_folder,
        "wav2vec2_folder": f"{output_folder}/wav2vec2_checkpoint",
        # these will need refactoring at some point (recipe-depending values)
        "pretrained_tokenizer_path": "speechbrain/asr-wav2vec2-switchboard",
        "pretrained_lm_tokenizer_path": "speechbrain/asr-transformer-switchboard",
        "channels_path": None,
        "concepts_path": None,
    }

    # Read the csv recipe file and detect which tests we have to run
    test_script, test_hparam, test_flag, test_check = prepare_test(
        recipe_folder,
        script_field,
        hparam_field,
        test_field=test_field,
        filters_fields=filters_fields,
        filters=filters,
    )

    check = True
    for i, recipe_id in enumerate(test_script.keys()):
        hparam_file = test_hparam[recipe_id]
        script_file = test_script[recipe_id]

        # Changing working folder to recipe folder (as 'run_folder' to avoid name conflict with arg 'recipe_folder')
        script_folder = os.path.dirname(script_file)
        run_folder = os.path.join(cwd, script_folder)
        os.chdir(run_folder)

        # Avoid files lister in avoid_list
        if hparam_file in avoid_list:
            print(
                f"\t({i + 1}/{len(test_script.keys())}) Skipped: {hparam_file}! (check avoid_list for details)"
            )
            continue

        print(
            "(%i/%i) Checking %s..."
            % (i + 1, len(test_script.keys()), hparam_file)
        )

        # Get absolute path to the hparam file
        hparam_file = os.path.join(cwd, hparam_file)

        # Load hyperparameters file with command-line overrides
        overrides = {"data_folder": data_folder, "output_folder": output_folder}

        tag_custom_model = None
        # Append additional overrides when needed
        with open(hparam_file) as f:
            for line in f:
                # os.chdir(run_folder) is not changing sys.module, and pydoc.locate (in load_hyperpyyaml) fails
                if "new:custom_model" in line:
                    tag_custom_model = "custom_model"
                    custom_model_from_root = f"{script_folder.replace(os.sep, '.')}.{tag_custom_model}"
                    if pydoc.locate(custom_model_from_root) is not None:
                        sys.modules[tag_custom_model] = sys.modules[
                            custom_model_from_root
                        ]
                # check for !PLACEHOLDER overrides
                flag_continue = False
                for key, value in add_placeholder_overrides.items():
                    placeholder_pattern = key + ": !PLACEHOLDER"
                    if (
                        placeholder_pattern in line
                        and line.find(placeholder_pattern) == 0
                    ):
                        overrides.update({key: value})
                        flag_continue = True

                # if !PLACEHOLDER was substituted already, skip further pattern overrides for this line
                if flag_continue:
                    continue

                # check for additional overrides
                for key, value in add_overrides.items():
                    pattern = key + ":"
                    if pattern in line and line.find(pattern) == 0:
                        overrides.update({key: value})

        with open(hparam_file) as fin:
            try:
                _ = load_hyperpyyaml(fin, overrides)
            except Exception as e:
                print("\t" + str(e))
                check = False
                print("\tERROR: cannot load %s" % (hparam_file))
        if tag_custom_model is not None:
            if tag_custom_model in sys.modules:
                del sys.modules[tag_custom_model]
            for tcm_key in list(sys.modules.keys()):
                if tcm_key.startswith("recipes"):
                    del sys.modules[tcm_key]
    return check
