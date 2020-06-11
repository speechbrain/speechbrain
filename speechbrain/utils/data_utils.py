"""This library gathers utilities for data io operation.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
"""

import os
import collections.abc


def get_all_files(
    dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None
):
    """Returns a list of files within found within a folder.

    Different options can be used to restrict the search to some specific
    patterns.

    Arguments
    ---------
    dirName : str
        the directory to search
    match_and : list
        a list that contains patterns to match. The file is
        returned if it matches all the entries in `match_and`.
    match_or : list
        a list that contains patterns to match. The file is
        returned if it matches one or more of the entries in `match_or`.
    exclude_and : list
        a list that contains patterns to match. The file is
        returned if it matches none of the entries in `exclude_and`.
    exclude_or : list
        a list that contains pattern to match. The file is
        returned if it fails to match one of the entries in `exclude_or`.

    Example
    -------
    >>> get_all_files('samples/rir_samples', match_and=['3.wav'])
    ['samples/rir_samples/rir3.wav']
    """

    # Match/exclude variable initialization
    match_and_entry = True
    match_or_entry = True
    exclude_or_entry = False
    exclude_and_entry = False

    # Create a list of file and sub directories
    listOfFile = os.listdir(dirName)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:

        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files(
                fullPath,
                match_and=match_and,
                match_or=match_or,
                exclude_and=exclude_and,
                exclude_or=exclude_or,
            )
        else:

            # Check match_and case
            if match_and is not None:
                match_and_entry = False
                match_found = 0

                for ele in match_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(match_and):
                    match_and_entry = True

            # Check match_or case
            if match_or is not None:
                match_or_entry = False
                for ele in match_or:
                    if ele in fullPath:
                        match_or_entry = True
                        break

            # Check exclude_and case
            if exclude_and is not None:
                match_found = 0

                for ele in exclude_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(exclude_and):
                    exclude_and_entry = True

            # Check exclude_and case
            if exclude_or is not None:
                exclude_or_entry = False
                for ele in exclude_or:
                    if ele in fullPath:
                        exclude_or_entry = True
                        break

            # If needed, append the current file to the output list
            if (
                match_and_entry
                and match_or_entry
                and not (exclude_and_entry)
                and not (exclude_or_entry)
            ):
                allFiles.append(fullPath)

    return allFiles


def split_list(seq, num):
    """Returns a list of splits in the sequence.

    Arguments
    ---------
    seq : iterable
        the input list, to be split.
    num : int
        the number of chunks to produce.

    Example
    -------
    >>> split_list([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
    [[1, 2], [3, 4], [5, 6], [7, 8, 9]]
    """
    # Average length of the chunk
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    # Creating the chunks
    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg

    return out


def recursive_items(dictionary):
    """Yield each (key, value) of a nested dictionary

    Arguments
    ---------
    dictionary : dict
        the nested dictionary to list.

    Yields
    ------
    `(key, value)` tuples from the dictionary.

    Example
    -------
    >>> rec_dict={'lev1': {'lev2': {'lev3': 'current_val'}}}
    >>> [item for item in recursive_items(rec_dict)]
    [('lev3', 'current_val')]
    """
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)


def recursive_update(d, u, must_match=False):
    """Similar function to `dict.update`, but for a nested `dict`.

    From: https://stackoverflow.com/a/3233356

    If you have to a nested mapping structure, for example:

        {"a": 1, "b": {"c": 2}}

    Say you want to update the above structure with:

        {"b": {"d": 3}}

    This function will produce:

        {"a": 1, "b": {"c": 2, "d": 3}}

    Instead of:

        {"a": 1, "b": {"d": 3}}

    Arguments
    ---------
    d : dict
        mapping to be updated
    u : dict
        mapping to update with
    must_match : bool
        Whether to throw an error if the key in `u` does not exist in `d`.

    Example
    -------
    >>> d = {'a': 1, 'b': {'c': 2}}
    >>> recursive_update(d, {'b': {'d': 3}})
    >>> d
    {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    # TODO: Consider cases where u has branch off k, but d does not.
    # e.g. d = {"a":1}, u = {"a": {"b": 2 }}
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping) and k in d:
            recursive_update(d.get(k, {}), v)
        elif must_match and k not in d:
            raise KeyError(
                f"Override '{k}' not found in: {[key for key in d.keys()]}"
            )
        else:
            d[k] = v
