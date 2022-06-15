import csv
from speechbrain.utils.check_yaml import check_yaml_vs_script


def test_yaml_script_consistency(recipe_list="tests/recipes.csv"):
    """This test checks the consistency between yaml files (used to specify
    hyperparameters) and script files (that implement the training recipe).

    Arguments
    ---------
    recipe_list : path
        Path of the a csv file containing the training scripts with their coupled
        yaml files (with colums called 'Hparam_file', 'Script_file', 'Data_prep_file')
    """
    with open(recipe_list, newline="") as csvfile:
        check = True
        reader = csv.DictReader(csvfile, delimiter=",", skipinitialspace=True)
        for row in reader:
            if not (
                check_yaml_vs_script(
                    row["Hparam_file"],
                    row["Script_file"],
                    row["Data_prep_file"].split(" "),
                )
            ):
                check = False
    assert check
