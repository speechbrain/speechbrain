"""Library for the HuggingFace (HF) repositories.

Authors
 * Mirco Ravanelli 2022
 * Andreas Nautsch 2022
"""
import os
import csv
from speechbrain.utils.data_utils import download_file


def run_HF_check(
    recipe_folder="tests/recipes", field="HF_repo", output_folder="tests/tmp",
):
    """Checks if the code reported in the readme files of the HF repository is
    runnable. Note: the tests run the code marked as python in the readme file.

    Arguments
    ---------
    recipe_folder: path
        Path of the folder containing csv recipe files summarizing all the recipes in the repo.
    field: string
        Field of the csv recipe file containing the links to HF repos.
    output_folder: path
        Where to download the HF readme files.

    Returns
    ---------
    check: True
        True if all the code runs, False otherwise.
    """
    # Detect list of HF repositories
    HF_repos = repo_list(recipe_folder, field)

    # Set up output folder
    os.makedirs(output_folder, exist_ok=True)
    os.chdir(output_folder)

    # Download all first
    for i, repo in enumerate(HF_repos):
        check_repo(repo, skip_exec=True)

    # Checking all detected repos
    check = True
    for i, repo in enumerate(HF_repos):
        print("(%i/%i) Checking %s..." % (i + 1, len(HF_repos), repo))
        if not (check_repo(repo, skip_download=True)):
            check = False
    return check


def repo_list(recipe_folder="tests/recipes", field="HF_repo"):
    """Get the list of HF recipes in the csv recipe file.

    Arguments
    ---------
    recipe_folder: path
        Path of the folder containing csv recipe files summarizing all the recipes in the repo.
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


def check_repo(HF_repo, skip_download=False, skip_exec=False):
    """Runs the code reported in the README file of the given HF_repo. It checks
    if the code runs without errors.

    Arguments
    ---------
    HF_repo: string
        URL of the HF repository to check.
    skip_download: bool
        Flag whether/not to skip download part of check.
    skip_exec: bool
        Flag whether/not to README execution part of check.

    Returns
    ---------
    check: bool
        True if all the code runs, False otherwise.
    """
    exp_name = os.path.basename(HF_repo)
    if HF_repo[-1] == "/":
        readme_file = HF_repo + "raw/main/README.md"
    else:
        readme_file = HF_repo + "/raw/main/README.md"

    dest_file = exp_name + ".md"
    if not skip_download:
        download_file(readme_file, dest_file)

    code_snippets = []
    code = []
    flag = False
    check = True
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
                    if not skip_exec:
                        code.append(line)
                    elif not skip_download:
                        if (
                            ("from_hparams" in line)
                            or ("foreign_class" in line)
                            or ("import" in line)
                        ):
                            code.append(
                                line.replace(")", ", download_only=True)")
                            )

    for code in code_snippets:
        try:
            exec("\n".join(code))
        except Exception as e:
            print("\t" + str(e))
            check = False
            print("\tERROR: cannot run code snippet in %s" % (HF_repo))
    return check
