"""Tests for checking the recipes and their files.

Authors
 * Mirco Ravanelli 2022
"""

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
