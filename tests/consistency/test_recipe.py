"""Tests for checking the recipes and their files.

Authors
 * Mirco Ravanelli 2022
"""
import os
import csv
from speechbrain.utils.data_utils import get_all_files, get_list_from_csv


__skip_list = ["README.md", "setup"]


def test_recipe_list(
    search_folder="recipes",
    hparam_ext=[".yaml"],
    hparam_field="Hparam_file",
    recipe_folder="tests/recipes",
    flags_field="test_debug_flags",
    avoid_list=[
        "/models/",
        "/results/",
        "recipes/Voicebank/MTL/CoopNet/hparams/logger.yaml",
        "recipes/LibriParty/generate_dataset/dataset.yaml",
        "hpopt.yaml",
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
    recipe_folder: path
        Path of the folder containing csv recipe files.
    flags_field: str
        Field of the csv file where the debug flags are stated (for data flow testing).
    avoid_list: list
        List of files for which this check must be avoided.

    Returns
    ---------
    bool:
        True if the test passes, False otherwise.

    """
    all_diffs_zero = True
    all_with_flags = True
    for recipe_csvfile in os.listdir(recipe_folder):
        if recipe_csvfile in __skip_list:
            continue
        dataset = os.path.splitext(recipe_csvfile)[0]
        hparam_lst = get_all_files(
            os.path.join(search_folder, dataset),
            match_and=hparam_ext,
            exclude_or=avoid_list,
        )
        recipe_lst = get_list_from_csv(
            os.path.join(recipe_folder, recipe_csvfile), field=hparam_field
        )
        diff_lst = list(set(hparam_lst) - set(recipe_lst))

        for file in diff_lst:
            print(
                "\tERROR: The file %s is not listed in %s. Please add it. \
                    For more info see tests/consistency/README.md"
                % (file, recipe_csvfile)
            )

        all_diffs_zero &= len(diff_lst) == 0

        flags_lst = get_list_from_csv(
            os.path.join(recipe_folder, recipe_csvfile), flags_field
        )
        for flags in flags_lst:
            if not flags:
                all_with_flags = False
                print(f"\tERROR: {flags_field} are missing in {recipe_csvfile}")

    assert all_diffs_zero
    assert all_with_flags


def test_recipe_files(
    recipe_folder="tests/recipes",
    fields=["Script_file", "Hparam_file", "Data_prep_file", "Readme_file"],
):
    """This test checks if the files listed in the recipe csv file exist.

    Arguments
    ---------.
    recipe_folder: path
        Path of the folder containing csv recipe files.
    fields: list
        Fields of the csv recipe file to check.

    Returns
    ---------
    check: bool
        True if the test passes, False otherwise.
    """
    check = True
    # Loop over all recipe CSVs
    for recipe_csvfile in os.listdir(recipe_folder):
        if recipe_csvfile in __skip_list:
            continue
        for field in fields:
            lst = get_list_from_csv(
                os.path.join(recipe_folder, recipe_csvfile), field=field
            )
            lst = filter(None, lst)
            for files in lst:
                files = files.split(" ")
                files = filter(None, files)
                for file in files:
                    if not (os.path.exists(file.strip())):
                        print(
                            "\tERROR: The file %s listed in %s does not exist!"
                            % (file, recipe_csvfile)
                        )
                        check = False
    assert check


def test_mandatory_files(
    recipe_folder="tests/recipes",
    must_exist=["Script_file", "Hparam_file", "Readme_file"],
):
    """This test checks if all the recipes contain the specified mandatory files.

    Arguments
    ---------.
    recipe_folder: path
        Path of the folder containing csv recipe files.
    must_exist: list
        List of the fields of the csv recipe file that must contain valid paths.

    Returns
    ---------
    check: bool
        True if the test passes, False otherwise.
    """

    check = True

    # Loop over all recipe CSVs
    for recipe_csvfile in os.listdir(recipe_folder):
        if recipe_csvfile in __skip_list:
            continue
        with open(
            os.path.join(recipe_folder, recipe_csvfile), newline=""
        ) as csvf:
            reader = csv.DictReader(csvf, delimiter=",", skipinitialspace=True)
            for row_id, row in enumerate(reader):
                for field in must_exist:
                    if not (os.path.exists(row[field].strip())):
                        print(
                            "\tERROR: The recipe %s does not contain a %s. Please add it to %s!"
                            % (row_id, field, recipe_csvfile)
                        )
                        check = False
    assert check


def test_README_links(
    recipe_folder="tests/recipes",
    readme_field="Readme_file",
    must_link=["Result_url", "HF_repo"],
):
    """This test checks if the README file contains the correct GDRIVE and HF repositories.

    Arguments
    ---------.
    recipe_folder: path
        Path of the folder containing csv recipe files.
    readme_field: string
        Field of the csv recipe file that contains the path to the readme file.
    must_link : list
        Fields that contains the paths that must be linked in the readme file.

    Returns
    ---------
    check: bool
        True if the test passes, False otherwise.
    """
    check = True

    # Loop over all recipe CSVs
    for recipe_csvfile in os.listdir(recipe_folder):
        if recipe_csvfile in __skip_list:
            continue
        with open(
            os.path.join(recipe_folder, recipe_csvfile), newline=""
        ) as csvf:
            reader = csv.DictReader(csvf, delimiter=",", skipinitialspace=True)
            for row in reader:
                with open(row[readme_field].strip()) as readmefile:
                    content = readmefile.read()
                    for field in must_link:
                        links = row[field].strip().split(" ")
                        for link in links:
                            if len(link) == 0:
                                continue
                            if not (link in content):
                                print(
                                    "\tERROR: The link to %s does not exist in %s. Please add it to %s!"
                                    % (link, row[readme_field], recipe_csvfile)
                                )
                                check = False
    assert check
