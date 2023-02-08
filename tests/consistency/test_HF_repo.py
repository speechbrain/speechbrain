"""Library for the HuggingFace (HF) repositories.

Authors
 * Mirco Ravanelli 2022
 * Andreas Nautsch 2022
"""
import os
import csv
from speechbrain.utils.data_utils import download_file


def run_HF_check(
    recipe_folder="tests/recipes", field="HF_repo", output_folder="HF_repos",
):
    """Checks if the code reported in the readme files of the HF repository is
    runnable. Note: the tests run the code marked as python in the readme file.

    Arguments
    ---------
    recipe_folder: path
        Path of the folder with csv recipe files summarizing all the recipes in the repo.
    field: string
        Field of the csv recipe file containing the links to HF repos.
    output_folder: path
        Where to download the HF readme files.

    Returns
    ---------
    check: True
        True if all the code runs, False otherwise.
    """
    check = True
    for recipe_csvfile in os.listdir(recipe_folder):
        # Detect list of HF repositories
        HF_repos = repo_list(os.path.join(recipe_folder, recipe_csvfile), field)

        # Set up output folder
        os.makedirs(output_folder, exist_ok=True)
        os.chdir(output_folder)

        # Checking all detected repos
        for repo in HF_repos:
            if not (check_repo(repo)):
                check = False
    return check


def repo_list(recipe_folder="tests/recipes", field="HF_repo"):
    """Get the list of HF recipes in the csv recipe file.

    Arguments
    ---------
    recipe_folder: path
        Path of the fodler with csv recipe files summarizing all the recipes in the repo.
    field: string
        Field of the csv recipe file containing the links to HF repos.

    Returns
    ---------
    HF_repos: list
        List of the detected HF repos.
    """
    HF_repos = []

    # Loop over all recipe CSVs
    for recipe_csvfile in os.listdir(recipe_folder):
        with open(
            os.path.join(recipe_folder, recipe_csvfile), newline=""
        ) as csvf:
            reader = csv.DictReader(csvf, delimiter=",", skipinitialspace=True)
            for row in reader:
                if len(row[field]) > 0:
                    repos = row[field].split(" ")
                    for repo in repos:
                        HF_repos.append(repo)
    HF_repos = set(HF_repos)
    return HF_repos


def check_repo(HF_repo):
    """Runs the code reported in the README file of the given HF_repo. It checks
    if the code runs without errors.

    Arguments
    ---------
    HF_repo: string
        URL of the HF repository to check.

    Returns
    ---------
    check: bool
        True if all the code runs, False otherwise.
    """
    exp_name = os.path.basename(HF_repo)
    readme_file = HF_repo + "/raw/main/README.md"

    dest_file = exp_name + ".md"
    download_file(readme_file, dest_file)

    code_snippets = []
    code = []
    flag = False
    with open(dest_file, "r") as f:
        for line in f:
            if "```python" in line:
                flag = True
                code = []
            elif "```\n" in line and flag:
                flag = False
                code_snippets.append(code)
            elif flag:
                if len(line.strip()) > 0:
                    code.append(line)
                    print(line)

    for code in code_snippets:
        try:
            exec("\n".join(code))
        except Exception as e:
            print("\t" + str(e))
            check = False
            print("\tERROR: cannot run code snippet in %s" % (HF_repo))
    return check
