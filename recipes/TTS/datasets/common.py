"""Helper functions for datasets

Authors
* Artem Ploujnikov 2021
"""

import os


def filename_to_id(file_name):
    """
    Returns the provided file name without the extension
    and the directory part. Based on the convention of
    the dataset, it can be used as an ID

    Arguments
    ---------
    file_name: str
        the file name (of the .txt or .wav file)

    Returns
    -------
    item_id: str
        the ID part of the filename
    """
    base_name = os.path.basename(file_name)
    item_id, _ = os.path.splitext(base_name)
    return item_id
