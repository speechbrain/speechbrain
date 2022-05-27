from speechbrain.utils.check_yaml import check_yaml_vs_script


def test_yaml_script_consistency(recipe_list="tests/unittests/recipe_list.txt"):
    """This test checks the consistency between yaml files (used to specify
    hyperparameters) and script files (that implement the training recipe).

    Arguments
    ---------
    recipe_list : path
        Path of the a file containing the training scripts with their coupled
        yaml files (every line contains <path_to_training_script> <path_to_yaml_file>)
                                    (or <path_to_training_script> <path_to_yaml_file> <path_to_prepare_script>)
    """
    check = True
    with open(recipe_list) as f:
        for line in f:
            entry_items = (
                line.rstrip().replace(" \n", "\n").replace("  ", " ").split(" ")
            )
            if len(entry_items) >= 2:
                script_file, hparam_file = entry_items[:2]
                prepare_files = entry_items[2:]
                if not (
                    check_yaml_vs_script(
                        hparam_file, script_file, prepare_files
                    )
                ):
                    check = False
            else:
                print("Error processing line: %s" % line)
    assert check
