"""This library gathers utilities for data io operation.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Samuele Cornell 2020
"""

import os
import shutil
import urllib.request
import collections.abc
import torch
import tqdm


def undo_padding(batch, lengths):
    """Produces Python lists given a batch of sentences with
    their corresponding relative lenghts.

    Arguments
    ---------
    batch : tensor
        Batch of sentences gathered in a batch.
    lenght: tensor
        Relative length of each sentence in the batch.

    Example
    -------
    >>> batch=torch.rand([4,100])
    >>> lengths=torch.tensor([0.5,0.6,0.7,1.0])
    >>> snt_list=undo_padding(batch, lengths)
    >>> len(snt_list)
    4
    """
    batch_max_len = batch.shape[1]
    as_list = []
    for seq, seq_length in zip(batch, lengths):
        actual_size = int(torch.round(seq_length * batch_max_len))
        seq_true = seq.narrow(0, 0, actual_size)
        as_list.append(seq_true.tolist())
    return as_list


def get_all_files(
    dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None
):
    """Returns a list of files found within a folder.

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


def download_file(
    source, dest, unpack=False, dest_unpack=None, replace_existing=False
):
    class DownloadProgressBar(tqdm.tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if "http" not in source:
        shutil.copyfile(source, dest)

    elif not os.path.isfile(dest) or (
        os.path.isfile(dest) and replace_existing
    ):
        print(f"Downloading {source} to {dest}")
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=source.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(
                source, filename=dest, reporthook=t.update_to
            )
    else:
        print("Destination path is not empty. Skipping download")

    # Unpack if necessary
    if unpack:
        if dest_unpack is None:
            dest_unpack = os.path.dirname(dest)
        print(f"Extracting {dest} to {dest_unpack}")
        shutil.unpack_archive(dest, dest_unpack)


class FuncPipeline:
    """
    Chain together functions.

    The class instances are callable, and will apply all given functions to the input,
    in the given order.

    Arguments
    ---------
    *funcs : list, optional
        Any number of functions, given in order of execution.

    Returns
    -------
    Any
        The input as processed by each function. If no functions were given, simply returns the input.
    """

    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, x):
        if not self.funcs:
            return x
        for func in self.funcs:
            x = func(x)
        return x

    def __str__(self):
        if self.funcs:
            return "FuncPipeline:\n" + "\n".join(str(f) for f in self.funcs)
        else:
            return "Empty FuncPipeline"


def pad_right_to(
    tensor: torch.Tensor, target_shape: (list, tuple), mode="constant", value=0,
):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Parameters
    ----------
    tensor : input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values

    """
    assert len(target_shape) == tensor.ndim

    pads = []
    valid_vals = []
    i = len(target_shape) - 1
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1

    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals


def batch_pad_right(tensors: list, mode="constant", value=0):
    """
    Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Parameters
    ----------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values

    """

    if not len(tensors):
        raise IndexError("Tensors list must not be empty")

    if len(tensors) == 1:
        return tensors[0].unsqueeze(0), torch.tensor([1.0])

    if not (
        any(
            [tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))]
        )
    ):
        raise IndexError("All tensors must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the last dimension

    max_shape = []
    for dim in range(tensors[0].ndim):
        if (dim < tensors[0].ndim - 1) and not all(
            [x.shape[dim] != tensors[0][dim] for x in tensors]
        ):
            raise EnvironmentError(
                "Tensors should have same dimensions except for last one"
            )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent[0])

    batched = torch.stack(batched)

    return batched, torch.tensor(valid)


def split_by_whitespace(text):
    """A very basic functional version of str.split"""
    return text.split()
