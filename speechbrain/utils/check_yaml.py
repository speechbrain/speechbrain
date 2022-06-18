"""Tests for checking consistency between yaml files and their corresponding training scripts.

Authors
 * Mirco Ravanelli 2022
 * Andreas Nautsch 2022
"""

import os
import re


def get_yaml_var(hparam_file):
    """Extracts from the input yaml file (hparams_file) the list of variables that
    should be used in the script file.

    Arguments
    ---------
    hparam_file : path
        Path of the yaml file containing the hyperparameters.

    Returns
    -------
    var_list: list
        list of the variables declared in the yaml file (sub-variables are not
        included).
    """
    var_lst = []
    with open(hparam_file) as f:
        for line in f:
            # Avoid empty lists or comments - skip pretrainer definitions
            if (
                len(line) <= 1
                or line.startswith("#")
                or "speechbrain.utils.parameter_transfer.Pretrainer" in line
            ):
                continue

            # Remove trailing characters
            line = line.rstrip()

            # Check for variables (e.g., 'key:' or '- !ref')
            if line.find(":") != -1 or line.find("- !ref") != -1:
                var_name = line[: line.find(":")]
                # The variables to check are like "key:" (we do not need to check
                # subvariavles as " key:")
                if not (
                    var_name[0] == " "
                    or var_name[0] == "\t"
                    or "!!" in line
                    or "!apply" in line
                ):
                    var_lst.append(var_name)
                # Check for the reference pattern
                # note: output_folder: !ref results/<experiment_name>/<seed> pattern
                if line.find("!ref") != -1:
                    # Check for 'annotation_list_to_check: [!ref <train_csv>, !ref <valid_csv>]' pattern
                    for subline in line.split("<"):
                        sub_var = subline[: subline.find(">")]
                        # Check for 'models[generator]' pattern (dictionary reference)
                        dict_pos = sub_var.find("[")
                        if dict_pos != -1:
                            sub_var = sub_var[:dict_pos]
                        # Remove variables already used in yaml
                        if sub_var in var_lst:
                            var_lst.remove(sub_var)
    return var_lst


def detect_script_vars(script_file, var_lst):
    """Detects from the input script file (script_file) which of given variables (var_lst) are demanded.

    Arguments
    ---------
    script_file : path
        Path of the script file needing the hyperparameters.
    var_lst : list
        list of the variables declared in the yaml file.

    Returns
    -------
    detected_var: list
        list of the variables detected in the script file.
    """
    var_types = [
        "hparams.",
        "modules.",
        'attr(self.hparams, "',
        'hparams.get("',
    ]
    detected_var = []
    with open(script_file) as f:
        for line in f:
            for var in var_lst:
                # The pattern can be ["key"] or ".key"
                if '["' + var + '"]' in line:
                    if var not in detected_var:
                        detected_var.append(var)
                        continue  # no need to go through the other cases for this var
                # case: hparams[f"{dataset}_annotation"] - only that structure at the moment
                re_match = re.search(r"\[f.\{.*\}(.*).\]", line)
                if re_match is not None:
                    if re_match.group(1) in var:
                        print(
                            "\t\tWARNING: potential inconsistency %s maybe used in %s (or not)."
                            % (var, re_match.group(0))
                        )
                        if var not in detected_var:
                            detected_var.append(var)
                            continue

                # Chek var types
                for var_type in var_types:
                    if var_type + var in line:
                        if var not in detected_var:
                            detected_var.append(var)
                            continue
                # case: tea_enc_list.append(hparams['tea{}_enc'])
                re_var = re.search(r"\[.(.*){}(.*).\]", line)
                if re_var is not None:
                    re_var_pattern = re_var.group(1) + ".*" + re_var.group(2)
                    re_pattern = re.search(re_var_pattern, var)
                    if re_pattern is not None:
                        if re_pattern.group() == var:
                            detected_var.append(var)
                            continue
    return detected_var


def check_yaml_vs_script(hparam_file, script_file):
    """Checks consistency between the given yaml file (hparams_file) and the
    script file. The function detects if there are variables declared in the yaml
    file, but not used in the script file.

    Arguments
    ---------
    hparam_file : path
        Path of the yaml file containing the hyperparameters.
    script_file : path
        Path of the script file (.py) containing the training recipe.

    Returns
    -------
    Bool
        This function returns False if some mismatch is detected and True otherwise.
        An error is raised to inform about which variable has been declared but
        not used.
    """
    print("Checking %s..." % (hparam_file))

    # Check if files exist
    if not (os.path.exists(hparam_file)):
        print("File %s not found!" % (hparam_file,))
        return False

    if not (os.path.exists(script_file)):
        print("File %s not found!" % (script_file,))
        return False

    # Getting list of variables declared in yaml
    var_lst = get_yaml_var(hparam_file)

    # Detect which of these variables are used in the script file
    detected_vars_train = detect_script_vars(script_file, var_lst)

    # Check which variables are declared but not used
    default_run_opt_keys = [
        "debug",
        "debug_batches",
        "debug_epochs",
        "device",
        "cpu",
        "data_parallel_backend",
        "distributed_launch",
        "distributed_backend",
        "find_unused_parameters",
        "jit_module_keys",
        "auto_mix_prec",
        "max_grad_norm",
        "nonfinite_patience",
        "noprogressbar",
        "ckpt_interval_minutes",
        "grad_accumulation_factor",
        "optimizer_step_limit",
    ]
    unused_vars = list(
        set(var_lst) - set(detected_vars_train) - set(default_run_opt_keys)
    )
    for unused_var in unused_vars:
        print(
            '\tERROR: variable "%s" not used in %s!' % (unused_var, script_file)
        )

    return len(unused_vars) == 0


def extract_patterns(lines, start_pattern, end_pattern):
    """Extracts a variables from start_pattern to end_pattern.

    Arguments
    ---------
    lines: list
        List of strings to parse.
    start_pattern: string
        String that indicated the start of the pattern.
    end_pattern: string
        String that indicated the end of the pattern.

    Returns
    -------
    var_lst: list
        List of variables detected.
    """
    var_lst = []

    for line in lines:
        start_indexes = [
            index
            for index in range(len(line))
            if line.startswith(start_pattern, index)
        ]
        for index in start_indexes:
            start_var = index + len(start_pattern)
            line_src = line[start_var:]
            var_name = ""
            for char in line_src:
                if char in end_pattern:
                    break
                var_name = var_name + char
            var_lst.append(var_name)
    return var_lst


def check_module_vars(
    hparam_file, script_file, module_key="modules:", module_var="self.modules."
):
    """Checks if the variables self.moduled.var are properly declared in the
    hparam file.

    Arguments
    ---------
    hparam_file : path
        Path of the yaml file containing the hyperparameters.
    script_file : path
        Path of the script file (.py) containing the training recipe.
    module_key: string
        String that denoted the start of the module in the hparam file.
    module_var: string
        String that denoted the start of the module in the script file.
    Returns
    -------
    Bool
        This function returns False if some mismatch is detected and True otherwise.
        An error is raised to inform about which variable has been used but
        not declared.
    """
    stop_char = [" ", ",", "(", ")", "[", "]", "{", "}", ".", ":", "\n"]
    module_block = False
    end_block = [" ", "\t"]
    avoid_lst = ["parameters", "keys", "eval", "train"]

    # Extract Modules variables from the hparam file
    module_vars_hparams = []
    with open(hparam_file) as f:
        for line in f:
            if module_key in line:
                module_block = True
                continue
            if line[0] not in end_block:
                module_block = False

            if module_block:
                var = line.strip().split(":")[0]
                module_vars_hparams.append(var)

    # Extract Modules variables from the script file
    with open(script_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    module_var_script = extract_patterns(lines, module_var, stop_char)
    module_var_script = set(module_var_script)
    for avoid in avoid_lst:
        if avoid in module_var_script:
            module_var_script.remove(avoid)

    # Remove optional variables "if hasattr(self.modules, "env_corrupt"):"
    stop_char.append('"')
    opt_vars = extract_patterns(lines, 'if hasattr(self.modules, "', stop_char)
    opt_vars.extend(
        extract_patterns(lines, 'if hasattr(self.hparams, "', stop_char)
    )

    # Remove optional
    for avoid in set(opt_vars):
        if avoid in module_var_script:
            module_var_script.remove(avoid)

    # Check Module variavles
    unused_vars = list(set(module_var_script) - set(module_vars_hparams))

    for unused_var in unused_vars:
        print(
            '\tERROR: variable "self.modules.%s" used in %s, but not listed in the "modules:" section of %s'
            % (unused_var, script_file, hparam_file)
        )
    return len(unused_vars) == 0
