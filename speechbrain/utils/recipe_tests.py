"""Library for running recipe tests.

Authors
 * Mirco Ravanelli 2022
 * Andreas Nautsch 2022
"""
import os
import re
import csv
import subprocess as sp
from hyperpyyaml import load_hyperpyyaml


def check_row_for_test(row, filter_fields, filters, test_field):
    """Checks if the current row of the csv recipe file has a test to run.

    Arguments
    ---------
    row: dict
        Line of the csv file (in dict from).
    filter_fields: list
        This can be used with the "filter" variable
        to run only some tests. For instance, filter_fileds=['Task'] and filters=['ASR'])
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
    for field in filter_fields:
        for filt in filters:
            if not (filt in row[field]):
                test = False
    if test:
        test_flag = row[test_field].strip()
        if len(test_flag) == 0:
            test = False
    return test


def prepare_test(
    recipe_csvfile="tests/recipes.csv",
    script_field="Script_file",
    hparam_field="Hparam_file",
    recipe_id_field="RecipeID",
    test_field="test_debug_flags",
    check_field="test_debug_checks",
    filters_fields=[],
    filters=[],
):
    """Extracts all the needed information to run the recipe test.

    Arguments
    ---------
    recipe_csvfile: path
        Path of the csv recipe file summarizing all the recipes in the repo.
    script_field: str
        Field of the csv recipe file containing the path of the script to run.
    hparam_field: str
        Field of the csv recipe file containing the path of the hparam file.
    recipe_id_field: str
        Field of the csv recipe file containing the unique recipe ID.
    test_field: string
        Field of the csv recipe file containing the test flags.
    check_field: string
        Field of the csv recipe file containing the checks to perform.
    filter_fields: list
        This can be used with the "filter" variable
        to run only some tests. For instance, filter_fileds=['Task'] and filters=['ASR'])
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
    """

    # Dictionary initialization
    test_script = {}
    test_hparam = {}
    test_flag = {}
    test_check = {}

    # Detect needed information for the recipe tests
    with open(recipe_csvfile, newline="") as csvf:
        reader = csv.DictReader(csvf, delimiter=",", skipinitialspace=True)
        for row in reader:
            if not (
                check_row_for_test(row, filters_fields, filters, test_field)
            ):
                continue
            recipe_id = row[recipe_id_field].strip()
            test_script[recipe_id] = row[script_field].strip()
            test_hparam[recipe_id] = row[hparam_field].strip()
            test_flag[recipe_id] = row[test_field].strip()
            test_check[recipe_id] = row[check_field].strip()

    return test_script, test_hparam, test_flag, test_check


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
    raise an error if the train_loss is >15.

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
    for line in lines:
        if epoch in line:
            lines_filt.append(line)

    # Raising an error if there are no lines after applying the filter
    if len(lines_filt) == 0:
        print(
            "\tERROR: No entries %s in %s (recipe %s). See performance_check entry."
            % (epoch, filename, recipe_id)
        )
        return False

    for line in lines_filt:

        # Search variable value
        pattern = variable + ": " + "(.*?) "
        var_value = re.search(pattern, line)

        if var_value is None:
            print(
                "\tERROR: The file %s of recipe %s does not contain the variable %s (needed for performance checks)"
                % (filename, recipe_id, variable)
            )
            return False
        var_value = float(var_value.group(1))
        check = check_threshold(threshold, var_value)

        if not (check):
            print(
                "\tERROR: The variable %s of file %s (recipe %s) violated the specified threshold (%s %s)"
                % (variable, filename, recipe_id, var_value, threshold)
            )

        break

    return check


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
    recipe_csvfile="tests/recipes.csv",
    script_field="Script_file",
    hparam_field="Hparam_file",
    recipe_id_field="RecipeID",
    test_field="test_debug_flags",
    check_field="test_debug_checks",
    run_opts="--device=cpu",
    output_folder="tests/tmp/recipes/",
    filters_fields=[],
    filters=[],
    do_checks=True,
):
    """Runs the recipes tests.

    Arguments
    ---------
    recipe_csvfile: path
        Path of the csv recipe file summarizing all the recipes in the repo.
    script_field: str
        Field of the csv recipe file containing the path of the script to run.
    hparam_field: str
        Field of the csv recipe file containing the path of the hparam file.
    recipe_id_field: str
        Field of the csv recipe file containing the unique recipe ID.
    test_field: string
        Field of the csv recipe file containing the test flags.
    check_field: string
        Field of the csv recipe file containing the checks to perform.
    run_opts: string
        Additional flags to add for the tests (see run_opts of speechbrain/core.py).
    output_folder: string
        Folder where the output of the tests are saved.
    filter_fields: list
        This can be used with the "filter" variable
        to run only some tests. For instance, filter_fileds=['Task'] and filters=['ASR'])
        will only run tests for ASR recipes.
    filters: list
        See above.
    do_checks:
        If True performs the checks on the output folder (when the check_field is not empty).

    Returns
    ---------
    check: True
        True if all the recipe tests pass, False otherwise.
    """
    # Create the output folder (where the tests results will be saved)
    os.makedirs(output_folder, exist_ok=True)
    print("Test ouputs will be put in %s" % (output_folder))

    # Read the csv recipe file and detect which tests we have to run
    test_script, test_hparam, test_flag, test_check = prepare_test(
        recipe_csvfile,
        script_field,
        hparam_field,
        filters_fields=filters_fields,
        filters=filters,
    )

    # Run  script (check how to get std out, std err and save them in files)
    check = True
    for i, recipe_id in enumerate(test_script.keys()):
        print(
            "(%i/%i) Running test for %s..."
            % (i + 1, len(test_script.keys()), recipe_id)
        )

        output_fold = os.path.join(output_folder, recipe_id)
        os.makedirs(output_fold, exist_ok=True)
        stdout_file = os.path.join(output_fold, "stdout.txt")
        stderr_file = os.path.join(output_fold, "stderr.txt")

        # Composing command to run
        cmd = (
            "python "
            + test_script[recipe_id]
            + " "
            + test_hparam[recipe_id]
            + " --output_folder="
            + output_fold
            + " "
            + test_flag[recipe_id]
            + " "
            + run_opts
        )

        # Running the test
        return_code = run_test_cmd(cmd, stdout_file, stderr_file)

        # Check return code
        if return_code != 0:
            print(
                "\tERROR: Error in %s. Check %s and %s for more info."
                % (recipe_id, stderr_file, stdout_file)
            )
            check = False

        # Checks
        check_str = test_check[recipe_id].strip()
        if do_checks and len(check_str) > 0:

            # Check if the expected files exist
            check &= check_files(check_str, output_fold, recipe_id)
            check &= check_performance(check_str, output_fold, recipe_id)

    return check


def load_yaml_test(
    recipe_csvfile="tests/recipes.csv",
    script_field="Script_file",
    hparam_field="Hparam_file",
    test_field="Hparam_file",
    filters_fields=[],
    filters=[],
    avoid_list=[
        "templates/hyperparameter_optimization_speaker_id/train.yaml",
        "templates/speaker_id/train.yaml",
        # recipes creating errors if NVIDIA driver is not on one's system
        "recipes/timers-and-such/multistage/hparams/train_LS_LM.yaml",
        "recipes/timers-and-such/multistage/hparams/train_TAS_LM.yaml",
        "recipes/timers-and-such/direct/hparams/train.yaml",
        "recipes/timers-and-such/decoupled/hparams/train_LS_LM.yaml",
        "recipes/timers-and-such/decoupled/hparams/train_TAS_LM.yaml",
        "recipes/fluent-speech-commands/direct/hparams/train.yaml",
        "recipes/CommonLanguage/lang_id/hparams/train_ecapa_tdnn.yaml",
        "recipes/SLURP/direct/hparams/train.yaml",
    ],
    rir_folder="tests/tmp/rir",
    data_folder="tests/tmp/yaml",
    output_folder="tests/tmp/yaml",
):
    """Tests if the yaml files can be loaded without errors.

    Arguments
    ---------
    recipe_csvfile: path
        Path of the csv recipe file summarizing all the recipes in the repo.
    script_field: str
        Field of the csv recipe file containing the path of the script to run.
    hparam_field: str
        Field of the csv recipe file containing the path of the hparam file.
    test_field: string
        Field of the csv recipe file containing the test flags.
    filter_fields: list
        This can be used with the "filter" variable
        to run only some tests. For instance, filter_fileds=['Task'] and filters=['ASR'])
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

    # Read the csv recipe file and detect which tests we have to run
    test_script, test_hparam, test_flag, test_check = prepare_test(
        recipe_csvfile,
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

        # Changing working folder to recipe folder
        recipe_folder = os.path.dirname(script_file)
        recipe_folder = os.path.join(cwd, recipe_folder)
        os.chdir(recipe_folder)

        # Avoid files lister in avoid_list
        if hparam_file in avoid_list:
            continue

        print(
            "(%i/%i) Checking %s..."
            % (i + 1, len(test_script.keys()), hparam_file)
        )

        # Get absolute path to the hparam file
        hparam_file = os.path.join(cwd, hparam_file)

        # Load hyperparameters file with command-line overrides
        overrides = {"data_folder": data_folder, "output_folder": output_folder}

        # Append additional overrides when needed
        with open(hparam_file) as f:
            for line in f:
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
    return check
