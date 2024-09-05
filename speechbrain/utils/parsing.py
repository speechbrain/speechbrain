"""General file parsing utilities

Authors
 * Sylvain de Langen 2024
 * Pierre Champion 2023
"""

import warnings


def expect_line_count_in_file(fname, expected=None, offset=0):
    """Count the number of lines in a file.

    One trailing newline at the end of the file will be ignored and won't count
    as an empty line (common with editors or generated files).
    Other empty lines are counted.

    If `expected` is specified, then:

    - If the file is missing, a warning will be emitted and `expected` will be
      returned as a default.
    - If the file is present, but the line count is not what was expected, then
      a warning will be emitted but the real count will be returned.

    Otherwise, if the file is missing, an error will be raised.

    Arguments
    ---------
    fname : str | pathlib.Path
        Path to the file to count the lines of.
    default : int
        Value to return when the file is not found. Other errors are stillAlso used to warn if the
        file is present the line count doesn't match this value.
    offset : int
        Value to add to the number of line (not to `default`).

    Returns
    -------
    int
        Line count of the file (or default if not found).

    Raises
    ------
    FileNotFoundError
        File was not found **and** no `default` value was supplied.
    OSError
        Another IO error has occurred. Non-`FileNotFoundError`s errors don't
        result in `expected` being returned.

    Example in yaml
    ---------------
        output_neurons: !apply:speechbrain.utils.parsing.expect_line_count_in_file
           file: !ref <lang_dir>/tokens.txt
           expected: 21
    """
    try:
        with open(fname, "r", encoding="utf-8") as fp:
            line_count = len(fp.readlines())
            if line_count != expected:
                warnings.warn(
                    f"Number of line of '{fname}' ({line_count}) differs from"
                    f" default ({expected})! Using '{line_count}' as value."
                    " Please update your configuration."
                )

        return line_count + offset
    except FileNotFoundError:
        if expected is not None:
            warnings.warn(
                f"File {fname} not found, using default value={expected}. "
                "On the next run, the value may differ!"
            )
            return expected

        raise
