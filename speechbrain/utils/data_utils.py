"""
 -----------------------------------------------------------------------------
 data_utils.py

 Description: This library gathers utils for data io operation.
 -----------------------------------------------------------------------------
"""

import os
import re
import copy
import yaml
import logging
import inspect
import ruamel.yaml
import collections.abc
from io import StringIO
from pydoc import locate
logger = logging.getLogger(__name__)


def get_all_files(
    dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None
):
    """
     -------------------------------------------------------------------------
     speechbrain.utils.data_utils.get_all_files (author: Mirco Ravanelli)

     Description: This function get a list of files within found within a
                  folder. Different options can be used to restrict the search
                  to some specific patterns.

     Input (call):
        - dirName (type: directory, mandatory):
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

     Output (call):
        - allFiles(type:list):
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


# NOTE: Empty dict as default parameter is fine here since overrides are never
# modified
def load_extended_yaml(
    yaml_string,
    overrides={},
):
    """
    Description:
        This function implements the SpeechBrain extended YAML syntax
        by providing parser for it. The purpose for this syntax is a compact,
        structured hyperparameter and computation block definition. This
        function implements two extensions to the yaml syntax, $-reference
        and object instantiation.

        $-reference substitution:
            Allows internal references to any other node in the file. Any
            tag (starting with '!') that contains $<key> will have all
            referrences replaced by the corresponding value, :

            ```
            constants:
                output_folder: exp/asr
            alignment_saver: !asr.ali.hmm.save
                save_dir: !$constants.output_folder # replaced with exp/asr
            ```

            Strings values are handled specially: $-strings are substituted but
            the rest of the string is left in place, allowing filepaths to be
            easily extended:

            ```
            constants:
                output_folder: exp/asr
            alignment_saver: !asr.ali.hmm.save
                save_dir: !$constants.output_folder/ali # exp/asr/ali
            ```

        object instantiation:
            If a '!' tag is used, the node is interpreted as the parameters
            for instantiating the named class. In the previous example,
            the alignment_saver will be an instance of the asr.ali.hmm.save
            class, with 'exp/asr/ali' passed to the __init__() method as
            a keyword argument. This is equivalent to:

            ```
            import asr.ali.hmm
            alignment_saver = asr.ali.hmm.save(save_dir='exp/asr/ali')
            ```

    Input:
        yaml_string:
            A file-like object or string from which to read.

        overrides:
            mapping with which to override the values read the string.
            As yaml implements a nested structure, so can the overrides.
            See speechbrain.utils.data_utils.recursive_update

    Output:
        A dictionary reflecting the structure of `yaml_string`.

    Authors:
        Aku Rouhe and Peter Plantinga 2020
    """

    # Load once to store references and apply overrides
    # using ruamel.yaml to preserve the tags
    ruamel_yaml = ruamel.yaml.YAML()
    preview = ruamel_yaml.load(yaml_string)
    preview = recursive_update(preview, overrides)

    # Dump back to string so we can load with bells and whistles
    yaml_string = StringIO()
    ruamel_yaml.dump(preview, yaml_string)
    yaml_string.seek(0)

    # NOTE: obj_and_ref_constructor needs to be defined in this scope to have
    # the correct version of preview
    def obj_and_ref_constructor(loader, tag_suffix, node):
        nonlocal preview  # Not needed, but let's be explicit
        if '$' in tag_suffix:
            # Check that the node is a scalar
            loader.construct_scalar(node)
            return _recursive_resolve(tag_suffix, [], preview)
        else:
            return object_constructor(loader, tag_suffix, node)

    # We also need a PyYAML Loader that is specific to this context
    # PyYAML syntax requires defining a new class to get a new loader
    class CustomLoader(yaml.SafeLoader):
        pass
    CustomLoader.add_multi_constructor('!', obj_and_ref_constructor)
    return yaml.load(yaml_string, Loader=CustomLoader)


def object_constructor(loader, tag_suffix, node):
    """
    Description:
        A constructor method for a '!' tag with a class name. The class
        is instantiated, and the sub-tree is passed as arguments.

    Inputs:
        - loader: loader
            The loader used to call this constructor (e.g. yaml.SafeLoader)

        - tag_suffix: string
            The rest of the tag (after the '!' in this case)

        - node: node
            The sub-tree belonging to the tagged node

    Outputs:
        The instantiated class

    Author:
        Peter Plantinga 2020
    """
    class_ = locate(tag_suffix)
    if class_ is None:
        raise ValueError('There is no such class as %s' % tag_suffix)

    # Parse arguments from the node
    kwargs = {}
    if isinstance(node, yaml.MappingNode):
        kwargs = loader.construct_mapping(node, deep=True)
    elif isinstance(node, yaml.SequenceNode):
        args = loader.construct_sequence(node, deep=True)
        signature = inspect.signature(class_)
        kwargs = signature.bind(*args).arguments

    return class_(**kwargs)


def deref(ref, preview):
    """
    Description:
        Find the value referred to by a reference in dot-notation

    Inputs:
        - ref: string
            The location of the requested value, e.g. 'constants.param'

        - preview: dict
            The dictionary to use for finding values

    Author:
        Peter Plantinga 2020
    """

    # Follow references in dot notation
    for part in ref[1:].split('.'):
        if part not in preview:
            error_msg = 'The reference "%s" does not exist' % ref
            logger.error(error_msg, exc_info=True)
        preview = preview[part]

    # For ruamel.yaml classes, the value is in the tag attribute
    try:
        preview = preview.tag.value[1:]
    except AttributeError:
        pass

    return preview


def _recursive_resolve(reference, reference_list, preview):
    reference_finder = re.compile(r'\$[\w.]+')
    if len(reference_list) > 1 and reference in reference_list[1:]:
        raise ValueError("Circular reference detected: ", reference_list)

    # Base case, no '$' present
    if '$' not in str(reference):
        return reference

    # First check for a full match. These replacements preserve type
    if reference_finder.fullmatch(reference):
        value = deref(reference, preview)
        reference_list += [reference]
        return _recursive_resolve(value, reference_list, preview)

    # Next, do replacements within the string (interpolation)
    matches = reference_finder.findall(reference)
    reference_list += [match[0] for match in matches]

    def replace_fn(x):
        return str(deref(x[0], preview))

    sub = reference_finder.sub(replace_fn, reference)
    return _recursive_resolve(sub, reference_list, preview)
