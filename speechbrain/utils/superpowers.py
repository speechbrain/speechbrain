"""Superpowers which should be rarely used.

This library contains functions for importing python classes and
for running shell commands. Remember, with great power comes great
responsibility.

Authors
 * Mirco Ravanelli 2020
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


def run_shell(cmd):
    r"""This function can be used to run a command in the bash shell.

    Arguments
    ---------
    cmd : str
        Shell command to run.

    Returns
    -------
    bytes
        The captured standard output.
    bytes
        The captured standard error.
    int
        The returncode.

    Raises
    ------
    OSError
        If returncode is not 0, i.e., command failed.

    Example
    -------
    >>> out, err, code = run_shell("echo 'hello world'")
    >>> out.decode(errors="ignore")
    'hello world\n'
    """

    # Executing the command
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )

    # Capturing standard output and error
    (output, err) = p.communicate()

    if p.returncode != 0:
        raise OSError(err.decode(errors="replace"))

    # Adding information in the logger
    msg = output.decode(errors="replace") + "\n" + err.decode(errors="replace")
    logger.debug(msg)

    return output, err, p.returncode
