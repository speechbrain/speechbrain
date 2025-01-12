"""
Source: https://github.com/microsoft/DNS-Challenge
Ownership: Microsoft

* Author
    rocheng
"""

import csv
import glob
import os
from shutil import copyfile


def get_dir(cfg, param_name, new_dir_name):
    """Helper function to retrieve directory name if it exists,
    create it if it doesn't exist"""

    if param_name in cfg:
        dir_name = cfg[param_name]
    else:
        dir_name = os.path.join(os.path.dirname(__file__), new_dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def write_log_file(log_dir, log_filename, data):
    """Helper function to write log file"""
    # data = zip(*data)
    with open(
        os.path.join(log_dir, log_filename),
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in data:
            csvwriter.writerow([row])


def str2bool(string):
    """Convert a string to a boolean value."""
    return string.lower() in ("yes", "true", "t", "1")


def rename_copyfile(src_path, dest_dir, prefix="", ext="*.wav"):
    """Copy and rename files from a source directory to a destination directory."""
    srcfiles = glob.glob(f"{src_path}/" + ext)
    for i in range(len(srcfiles)):
        dest_path = os.path.join(
            dest_dir, prefix + "_" + os.path.basename(srcfiles[i])
        )
        copyfile(srcfiles[i], dest_path)
