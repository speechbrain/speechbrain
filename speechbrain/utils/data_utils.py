"""
 -----------------------------------------------------------------------------
 data_utils.py

 Description: This library gathers utils for data io operation.
 -----------------------------------------------------------------------------
"""

import os
import collections.abc


def get_all_files(
    dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None
):
    """
     -------------------------------------------------------------------------
     speechbrain.utils.data_utils.get_all_files (author: Mirco Ravanelli)

     Description: This function get a list of files within found within a
                  folder. Different options can be used to restrict the search
                  to some specific patterns.

     Input (call):     - dirName (type: directory, mandatory):
                           it is the configuration dictionary.

                       - match_and (type: list, optional, default:None):
                           it is a list that contains pattern to match. The
                           file is returned if all the entries in match_and
                           are founded.

                       - match_or (type: list, optional, default:None):
                           it is a list that contains pattern to match. The
                           file is returned if one the entries in match_or are
                           founded.

                       - exclude_and (type: list, optional, default:None):
                           it is a list that contains pattern to match. The
                           file is returned if all the entries in match_or are
                            not founded.

                       - exclude_or (type: list, optional, default:None):
                           it is a list that contains pattern to match. The
                           file is returned if one of the entries in match_or
                           is not founded.

                      - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.


     Output (call):  - allFiles(type:list):
                       it is the output list of files.


     Example:   from speechbrain.utils.data_utils import get_all_files

                # List of wav files
                print(get_all_files('samples',match_and=['.wav']))

                # List of cfg files
                print(get_all_files('exp',match_and=['.cfg']))

     -------------------------------------------------------------------------
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
    """
     -------------------------------------------------------------------------
     speechbrain.utils.data_utils.split_list (author: Mirco Ravanelli)

     Description: This function splits the input list in N parts.

     Input (call):    - seq (type: list, mandatory):
                           it is the input list

                      - nums (type: int(1,inf), mandatory):
                           it is the number of chunks to produce

     Output (call):  out (type: list):
                       it is a list containing all chunks created.


     Example:  from speechbrain.utils.data_utils import split_list

               print(split_list([1,2,3,4,5,6,7,8,9],4))

     -------------------------------------------------------------------------
     """

    # Average length of the chunk
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    # Creating the chunks
    while last < len(seq):
        out.append(seq[int(last): int(last + avg)])
        last += avg

    return out


def recursive_items(dictionary):
    """
     -------------------------------------------------------------------------
     speechbrain.utils.data_utils.recursive_items (author: Mirco Ravanelli)

     Description: This function output the key, value of a recursive
                  dictionary (i.e, a dictionary that might contain other
                  dictionaries).

     Input (call):    - dictionary (type: dict, mandatory):
                           the dictionary (or dictionary of dictionaries)
                           in input.

     Output (call):   - (key,valies): key value tuples on the
                       recursive dictionary.


     Example:   from speechbrain.utils.data_utils import recursive_items

                rec_dict={}
                rec_dict['lev1']={}
                rec_dict['lev1']['lev2']={}
                rec_dict['lev1']['lev2']['lev3']='current_val'

                print(list(recursive_items(rec_dict)))

     -------------------------------------------------------------------------
     """
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)


def recursive_update(d, u):
    """
    Description:
        This function performs what dict.update does, but for any
        nested structure.
        If you have to a nested mapping structure, for example:
        { "a": 1, "b": { "c": 2 } }
         So say you want to update the above structure
        with:
        { "b": { "d": 3 } }
        This function will produce:
        { "a": 1, "b": { "c": 2, "d": 3 } }
        Instead of
        { "a": 1, "b": { "d": 3 } }
    Input:
        d - mapping to be updated
        u - mapping to update with
    Output:
        d - the updated mapping,
            note that the mapping is updated in place
    Author:
        Alex Martelli, with possibly other editors
        From: https://stackoverflow.com/a/3233356
    """
    # TODO: Consider cases where u has branch off k, but d does not.
    # e.g. d = {"a":1}, u = {"a": {"b": 2 }}
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
