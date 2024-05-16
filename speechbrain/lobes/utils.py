"""Utils for pipeline

Authors
 * Pierre Champion 2023
"""

import os
import warnings


def NumberOfLines(file, default, offset=-1):
    """Reads the number of line of a file or use default.
    Handy safeguard to make sure a default value equal file.readlines
    value which can be obtained after dataprep

    Arguments
    ---------
    file : file
        File to read from
    default : int
        Default value when file not found (before dataprep)
    offset : int
        Value to add to the number of line.
        Default to -1 to consider that the file to contain one line per output class

    Returns
    -------
        Returns int

    Raises
    ---------
        UserWarning: If file not found or if default differ from actual number of line

    Example in yaml
    ---------------
        output_neurons: !apply:speechbrain.lobes.utils.NumberOfLines
           file: !ref <lang_dir>/tokens.txt
    """
    if not os.path.isfile(file):
        warnings.warn(
            f"File {file} not found, using default value={default}, on the next"
            + " run, the value may differ!"
        )
        number_of_line = default
    else:
        with open(file, "r") as fp:
            number_of_line = len(fp.readlines())
            if number_of_line != default:
                warnings.warn(
                    f"Number of line of '{file}' ({number_of_line}) differ from"
                    + f" default ({default})! Using '{number_of_line}' as value."
                    + " Please update your configuration."
                )
    return number_of_line + offset
