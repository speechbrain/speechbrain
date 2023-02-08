"""Consistency check between yaml files and script files.

Authors
 * Mirco Ravanelli 2022
"""
import os
import csv
from tests.consistency.test_recipe import __skip_list
from tests.utils.check_yaml import check_yaml_vs_script


def test_yaml_script_consistency(recipe_folder="tests/recipes"):
    """This test checks the consistency between yaml files (used to specify
    hyperparameters) and script files (that implement the training recipe).

    Arguments
    ---------
    recipe_folder : path
        Path of the folder with csv files containing the training scripts with their coupled
        yaml files (with colums called 'Hparam_file', 'Script_file', 'Data_prep_file')
    """

    # Use this list to itemize special yaml for which we do not have to test
    avoid_check = []

    # Loop over all recipe CSVs
    for recipe_csvfile in os.listdir(recipe_folder):
        if recipe_csvfile in __skip_list:
            continue
        with open(
            os.path.join(recipe_folder, recipe_csvfile), newline=""
        ) as csvfile:
            check = True
            reader = csv.DictReader(
                csvfile, delimiter=",", skipinitialspace=True
            )
            for row in reader:

                # Avoid checks
                if row["Hparam_file"] in avoid_check:
                    continue

                # Check yaml-script consistency
                if not (
                    check_yaml_vs_script(row["Hparam_file"], row["Script_file"])
                ):
                    check = False

                # Check module variables
                # if not (check_module_vars(row["Hparam_file"], row["Script_file"])):
                #    check = False

    assert check
