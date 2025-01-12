#!/usr/bin/env python3
"""
Script: readme_builder.py

Description:
    This script creates the PERFORMANCE.md file, containing tables summarizing the performance
    of models and tasks available in SpeechBrain. It fetches performance data from
    the tests/recipes/*.csv files, where a special field called "performance" (e.g., Accuracy=85.7%)
    is expected.

Usage:
    python readme_builder.py

Authors:
    - Mirco Ravanelli 2023
"""

import argparse
import csv
import re

from speechbrain.utils.data_utils import get_all_files


def create_table(fid_w, csv_file):
    """
    Reads the input CSV file and adds performance tables to the output file.

    Args:
        fid_w (file pointer): Pointer to the output performance file.
        csv_file (str): Path to the recipe CSV file containing recipe information
                        (e.g., 'tests/recipes/LibriSpeech.csv').

    Returns
    -------
        None
    """
    # Read CSV file into a list of dictionaries
    with open(csv_file, "r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        recipes_lst = [row for row in csv_reader]

        dataset = recipes_lst[0].get("Dataset", "")
        if not recipes_lst or "performance" not in recipes_lst[0]:
            return

        print(f"## {dataset} Dataset\n", file=fid_w)

    # Filter recipes
    recipes = {task: [] for task in set(row["Task"] for row in recipes_lst)}

    for recipe_line in recipes_lst:
        got_performance = len(recipe_line["performance"].strip()) > 0

        if not got_performance:
            continue

        task = recipe_line["Task"]
        recipes[task].append(recipe_line)

    # Creating performance tables for each task
    for task, recipes_task in recipes.items():
        if not recipes_task:
            continue  # Skip empty task

        print(f"### {task}\n", file=fid_w)

        performance_dict = extract_name_value_pairs(
            recipes_task[0]["performance"]
        )
        performance_metrics = performance_dict.keys()
        performance_metrics = " | ".join(performance_metrics) + " |"

        print(
            f"| Model | Checkpoints | HuggingFace | {performance_metrics}",
            file=fid_w,
        )
        print(
            "".join(["| --------"] * (3 + len(performance_dict))) + "|",
            file=fid_w,
        )

        for recipe in recipes_task:
            performance_dict = extract_name_value_pairs(recipe["performance"])
            performance_values = " | ".join(performance_dict.values()) + " |"

            str_res = (
                f'[here]({recipe["Result_url"]})'
                if recipe["Result_url"]
                else "-"
            )
            hf_repo = (
                f'[here]({recipe["HF_repo"]})' if recipe["HF_repo"] else "-"
            )

            performance_line = f' | [`{recipe["Hparam_file"]}`]({recipe["Hparam_file"]}) | {str_res} | {hf_repo} | {performance_values}'
            print(performance_line, file=fid_w)

        print("\n", file=fid_w)


def extract_name_value_pairs(input_string):
    """
    Extracts performance metrics and their values from the performance line.

    Args:
        input_string (str): The string containing the performance.

    Returns
    -------
        dict: A dictionary containing the detected performance metrics and their values.
    """
    pattern = re.compile(r"(\w+(?:-\w+)?)=(\S+)")
    matches = pattern.findall(input_string)
    result = {name: value for name, value in matches}
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Create the performance file from the recipe info csv files."
        ),
    )

    parser.add_argument(
        "--recipe_info_dir",
        help="The directory where all the csv files containing the recipe info are stored. "
        "E.g., tests/recipes/",
    )
    parser.add_argument(
        "--output_file",
        help="The path to the output performance file to create",
    )

    args = parser.parse_args()

    file_w = open(args.output_file, "w", encoding="utf-8")

    # List of recipe files
    recipe_files = get_all_files(
        args.recipe_info_dir, match_and=[".csv"], exclude_or=["~"]
    )

    header = """\
# SpeechBrain Performance Report
This document provides an overview of the performance achieved on key datasets and tasks supported by SpeechBrain.
"""

    print(header, file=file_w)

    for csv_file in sorted(recipe_files):
        create_table(file_w, csv_file)

    file_w.close()

    print(args.output_file + " CREATED!")
