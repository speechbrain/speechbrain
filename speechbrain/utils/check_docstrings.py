"""This library contains functions that checks the dosctrings

Authors
 * Mirco Ravanelli 2022
"""

import re
from speechbrain.utils.data_utils import get_all_files


def extractName(s, search_class=False):
    """Extracts the names of the function or classes in the input string.

    Arguments
    ---------
    s: string
        Input string where to search for function or class names.
    search_clas: bool
        If True, searches for class names.

    Returns
    -------
    string: string
        Name of the function or class detected.
    """
    string = ""
    if search_class:
        regexp = re.compile(r"(class)\s(.*)\:")
    else:
        regexp = re.compile(r"(def)\s(.*)\(.*\)\:")
    for m in regexp.finditer(s):
        string += m.group(2)
    return string


def check_docstrings(
    base_folder=".", check_folders=["speechbrain", "tools", "templates"]
):
    """Checks if all the functions or classes have a docstring.

    Arguments
    ---------
    base_folder: path
        The main folder of speechbrain.
    check_folders: list
        List of subfolders to check.

    Returns
    -------
    check: bool
        True if all the functions/classes have a docstring, False otherwise.
    """
    # Search all python libraries in the folder of interest
    lib_lst = get_all_files(
        base_folder,
        match_and=[".py"],
        match_or=check_folders,
        exclude_or=[".pyc"],
    )
    check = True
    # Loop over the detected libraries
    for libpath in lib_lst:

        if "__" in libpath:
            continue
        print("Checking %s..." % (libpath))

        # Support variable initalization
        fun_name = libpath
        class_name = libpath
        check_line = True
        is_class = False
        first_line = True
        with open(libpath) as f:
            for line in f:

                # Remove spaces or tabs
                line = line.strip()

                # Avoid processing lines with the following patterns
                if ">>>" in line:
                    continue
                if "..." in line:
                    continue
                if len(line) == 0:
                    continue
                if line[0] == "#":
                    continue

                # Check if the docstring is written after the class/funct declaration
                if check_line:
                    if line[0] != '"' and not (is_class):
                        if line[0:2] == 'r"':
                            check_line = False
                            continue
                        check = False
                        if first_line:
                            print(
                                "\tERROR: The library %s must start with a docstring. "
                                % (libpath)
                                + "Please write it. For more info, see tests/consistency/DOCSTRINGS.md"
                            )
                        else:
                            print(
                                "\tERROR: The function %s in %s has no docstring. "
                                % (fun_name, libpath)
                                + "Please write it. For more info, see tests/consistency/DOCSTRINGS.md"
                            )
                    if line[0] != '"' and is_class:
                        if line[0:2] == 'r"':
                            check_line = False
                            continue

                        check = False
                        print(
                            "\tERROR: The class %s in %s has no docstring. "
                            % (class_name, libpath)
                            + "Please write it. For more info, see tests/consistency/DOCSTRINGS.md"
                        )

                    # Update support variables
                    check_line = False
                    is_class = False
                    first_line = False
                    continue

                # Extract function name (if any)
                fun_name = extractName(line)
                if len(fun_name) > 0 and line[0:3] == "def":
                    if fun_name[0] == "_":
                        continue
                    check_line = True

                # Extract class name (if any)
                class_name = extractName(line, search_class=True)
                if len(class_name) > 0 and line[0:5] == "class":
                    check_line = True
                    is_class = True
    return check
