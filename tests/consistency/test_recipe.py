"""Tests for checking the recipes and their files.

Authors
 * Mirco Ravanelli 2022
"""
import os
import csv
from speechbrain.utils.data_utils import get_all_files, get_list_from_csv


def test_recipe_list(
    search_folder="recipes",
    hparam_ext=[".yaml"],
    hparam_field="Hparam_file",
    recipe_csvfile="tests/recipes.csv",
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
    recipe_csvfile: path
        Path of the csv recipe file.
    avoid_list: list
        List of files for which this check must be avoided.

    Returns
    ---------
    bool:
        True if the test passes, False otherwise.

    """
    hparam_lst = get_all_files(
        search_folder, match_and=hparam_ext, exclude_or=avoid_list
    )
    recipe_lst = get_list_from_csv(recipe_csvfile, field=hparam_field)
    diff_lst = list(set(hparam_lst) - set(recipe_lst))

    for file in diff_lst:
        print(
            "\tERROR: The file %s is not listed in %s. Please add it. \
                For more info see tests/consistency/README.md"
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

    Returns
    ---------
    check: bool
        True if the test passes, False otherwise.
    """
    check = True
    for field in fields:
        lst = get_list_from_csv(recipe_csvfile, field=field)
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
    recipe_csvfile="tests/recipes.csv",
    must_exist=["Script_file", "Hparam_file", "Readme_file"],
    recipe_id_field="RecipeID",
):
    """This test checks if all the recipes contain the specified mandatory files.

    Arguments
    ---------.
    recipe_csvfile: path
        Path of the csv recipe file.
    must_exist: list
        List of the fields of the csv recipe file that must contain valid paths.
    recipe_id_field: str
        Field of the csv file containing a unique recipe ID.

    Returns
    ---------
    check: bool
        True if the test passes, False otherwise.
    """

    check = True
    with open(recipe_csvfile, newline="") as csvf:
        reader = csv.DictReader(csvf, delimiter=",", skipinitialspace=True)
        for row in reader:
            for field in must_exist:
                if not (os.path.exists(row[field].strip())):
                    print(
                        "\tERROR: The recipe %s does not contain a %s. Please add it!"
                        % (row[recipe_id_field], field)
                    )
                    check = False
    assert check


def test_README_links(
    recipe_csvfile="tests/recipes.csv",
    readme_field="Readme_file",
    must_link=["Result_url", "HF_repo"],
):
    """This test checks if the README file contains the correct GDRIVE and HF repositories.

    Arguments
    ---------.
    recipe_csvfile: path
        Path of the csv recipe file.
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
    with open(recipe_csvfile, newline="") as csvf:
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
                                "\tERROR: The link to %s does not exist in %s. Please add it!"
                                % (link, row[readme_field])
                            )
                            check = False
    assert check
