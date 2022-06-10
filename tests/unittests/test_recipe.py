"""Tests for checking the recipes and their files.

Authors
 * Mirco Ravanelli 2022
"""
import os
from speechbrain.utils.data_utils import get_all_files, get_list_from_csv


def test_recipe_list(
    search_folder="recipes",
    hparam_ext=[".yaml"],
    hparam_field="Hparam_file",
    recipe_csvfile="tests/recipes.csv",
    avoid_list=[
        "/models/",
        "recipes/Voicebank/MTL/CoopNet/hparams/logger.yaml",
        "recipes/LibriParty/generate_dataset/dataset.yaml",
    ],
):
    """This test checks if all the all hparam file of all the recipes are listed
    in the csv recipe file.

    Arguments
    ---------
    search_folder: path
        The path where to search the hparam files.
    hparam_ext: list
        The list containing the extensions of hparam files.
    hparam_field: str
        Field of the csv file where the path of the hparam file is reported.
    recipe_csvfile: path
        Path of the csv recipe file.
    avoid_list: list
        List of files for which this check must be avoided.
    """
    hparam_lst = get_all_files(
        search_folder, match_and=hparam_ext, exclude_or=avoid_list
    )
    recipe_lst = get_list_from_csv(recipe_csvfile, field=hparam_field)
    diff_lst = list(set(hparam_lst) - set(recipe_lst))

    for file in diff_lst:
        print(
            "\tWARNING: The file %s is not listed in %s. Please add it!"
            % (file, recipe_csvfile)
        )

    assert len(diff_lst) == 0


def test_recipe_files(
    recipe_csvfile="tests/recipes.csv",
    fields=["Script_file", "Hparam_file", "Data_prep_file", "Readme_file"],
):
    """This test checks if the files listed in the recipe csv file exist.

    Arguments
    ---------.
    recipe_csvfile: path
        Path of the csv recipe file.
    fields: list
        Fields of the csv recipe file to check.
    """
    check = True
    for field in fields:
        lst = get_list_from_csv(recipe_csvfile, field=field)
        lst = filter(None, lst)
        for file in lst:
            if not (os.path.exists(file.strip())):
                print(
                    "\tWARNING: The file %s listed in %s does not exist!"
                    % (file, recipe_csvfile)
                )
                check = False
    assert check
