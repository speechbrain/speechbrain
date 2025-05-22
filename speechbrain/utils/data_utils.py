"""This library gathers utilities for data io operation.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Samuele Cornell 2020
 * Adel Moumen 2024
 * Pierre Champion 2023
"""

import collections.abc
import csv
import gzip
import math
import os
import pathlib
import re
import shutil
import urllib.request
from numbers import Number

import torch
import tqdm

import speechbrain as sb


def undo_padding(batch, lengths):
    """Produces Python lists given a batch of sentences with
    their corresponding relative lengths.

    Arguments
    ---------
    batch : torch.Tensor
        Batch of sentences gathered in a batch.
    lengths : torch.Tensor
        Relative length of each sentence in the batch.

    Returns
    -------
    as_list : list
        A python list of the corresponding input tensor.

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
        The directory to search.
    match_and : list
        A list that contains patterns to match. The file is
        returned if it matches all the entries in `match_and`.
    match_or : list
        A list that contains patterns to match. The file is
        returned if it matches one or more of the entries in `match_or`.
    exclude_and : list
        A list that contains patterns to match. The file is
        returned if it matches none of the entries in `exclude_and`.
    exclude_or : list
        A list that contains pattern to match. The file is
        returned if it fails to match one of the entries in `exclude_or`.

    Returns
    -------
    allFiles : list
        The list of files matching the patterns.

    Example
    -------
    >>> get_all_files('tests/samples/RIRs', match_and=['3.wav'])
    ['tests/samples/RIRs/rir3.wav']
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

            # Check exclude_or case
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


def get_list_from_csv(csvfile, field, delimiter=",", skipinitialspace=True):
    """Gets a list from the selected field of the input csv file.

    Arguments
    ---------
    csvfile: path
        Path to the csv file.
    field: str
        Field of the csv file used to create the list.
    delimiter: str
        Delimiter of the csv file.
    skipinitialspace: bool
        Set it to true to skip initial spaces in the entries.

    Returns
    -------
    The list of files in the given field of a csv
    """
    lst = []
    with open(csvfile, newline="", encoding="utf-8") as csvf:
        reader = csv.DictReader(
            csvf, delimiter=delimiter, skipinitialspace=skipinitialspace
        )
        for row in reader:
            lst.append(row[field])
    return lst


def split_list(seq, num):
    """Returns a list of splits in the sequence.

    Arguments
    ---------
    seq : iterable
        The input list, to be split.
    num : int
        The number of chunks to produce.

    Returns
    -------
    A list of lists, length num and containing all elements of seq.

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
    """Yield each (key, value) of a nested dictionary.

    Arguments
    ---------
    dictionary : dict
        The nested dictionary to list.

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
        Mapping to be updated.
    u : dict
        Mapping to update with.
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
    source,
    dest,
    unpack=False,
    dest_unpack=None,
    replace_existing=False,
    write_permissions=False,
):
    """Downloads the file from the given source and saves it in the given
    destination path.

     Arguments
    ---------
    source : path or url
        Path of the source file. If the source is an URL, it downloads it from
        the web.
    dest : path
        Destination path.
    unpack : bool
        If True, it unpacks the data in the dest folder.
        The archive is preserved.

        File formats supported for unpacking/decompression are:

        - any format enumerated by `shutil.get_archive_formats()`, usually
          including `.tar`, `.tar.gz`, `.zip`.
        - plain `.gz` file (when not a `.tar` archive)

        Note that you should ALWAYS trust an archive you are extracting, for
        security reasons.
    dest_unpack: path
        Path where to store the unpacked dataset
    replace_existing : bool
        If True, replaces the existing files.
    write_permissions: bool
        When set to True, all the files in the dest_unpack directory will be granted write permissions.
        This option is active only when unpack=True.
    """
    try:
        # make sure all processing reached here before main process create dest_dir
        sb.utils.distributed.ddp_barrier()
        if sb.utils.distributed.if_main_process():

            class DownloadProgressBar(tqdm.tqdm):
                """DownloadProgressBar class."""

                def update_to(self, b=1, bsize=1, tsize=None):
                    """Needed to support multigpu training."""
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            # Create the destination directory if it doesn't exist
            dest_dir = pathlib.Path(dest).resolve().parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            if "http" not in source:
                shutil.copyfile(source, dest)

            elif not os.path.isfile(dest) or (
                os.path.isfile(dest) and replace_existing
            ):
                print(f"Downloading {source} to {dest}")
                with DownloadProgressBar(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    desc=source.split("/")[-1],
                ) as t:
                    urllib.request.urlretrieve(
                        source, filename=dest, reporthook=t.update_to
                    )
            else:
                print(f"{dest} exists. Skipping download")

            # Unpack if necessary
            if unpack:
                if dest_unpack is None:
                    dest_unpack = os.path.dirname(dest)
                print(f"Extracting {dest} to {dest_unpack}")

                if dest.endswith(".gz") and not dest.endswith(".tar.gz"):
                    # just a gzip'd file, but not an actual archive.
                    # merely uncompress it and remove the `.gz`.
                    with gzip.open(dest, "rb") as f_in:
                        with open(dest[:-3], "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.unpack_archive(dest, dest_unpack)

                if write_permissions:
                    set_writing_permissions(dest_unpack)

    finally:
        sb.utils.distributed.ddp_barrier()


def set_writing_permissions(folder_path):
    """
    This function sets user writing permissions to all the files in the given folder.

    Arguments
    ---------
    folder_path : folder
        Folder whose files will be granted write permissions.
    """
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Set writing permissions (mode 0o666) to the file
            os.chmod(file_path, 0o666)


def pad_right_to(tensor, target_shape, mode="constant", value=0):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Arguments
    ---------
    tensor : torch.Tensor
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
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == tensor.ndim
    pads = []  # this contains the abs length of the padding for each dimension.
    valid_vals = []  # this contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
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
    """Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Arguments
    ---------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.

    """
    if not len(tensors):
        raise IndexError("Tensors list must not be empty")

    if len(tensors) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return tensors[0].unsqueeze(0), torch.tensor([1.0])

    if not (
        all(
            [tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))]
        )
    ):
        raise IndexError("All tensors must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the first dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(tensors[0].ndim):
        if dim != 0:
            if not all(
                [x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]
            ):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for the first one"
                )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
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


def recursive_to(data, *args, **kwargs):
    """Moves data to device, or other type, and handles containers.

    Very similar to torch.utils.data._utils.pin_memory.pin_memory,
    but applies .to() instead.
    """
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, collections.abc.Mapping):
        return {
            k: recursive_to(sample, *args, **kwargs)
            for k, sample in data.items()
        }
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(
            *(recursive_to(sample, *args, **kwargs) for sample in data)
        )
    elif isinstance(data, collections.abc.Sequence):
        return [recursive_to(sample, *args, **kwargs) for sample in data]
    elif hasattr(data, "to"):
        return data.to(*args, **kwargs)
    # What should be done with unknown data?
    # For now, just return as they are
    else:
        return data


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def mod_default_collate(batch):
    """Makes a tensor from list of batch values.

    Note that this doesn't need to zip(*) values together
    as PaddedBatch connects them already (by key).

    Here the idea is not to error out.

    This is modified from:
    https://github.com/pytorch/pytorch/blob/c0deb231db76dbea8a9d326401417f7d1ce96ed5/torch/utils/data/_utils/collate.py#L42
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        try:
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        except RuntimeError:  # Unequal size:
            return batch
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        try:
            if (
                elem_type.__name__ == "ndarray"
                or elem_type.__name__ == "memmap"
            ):
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    return batch
                return mod_default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        except RuntimeError:  # Unequal size
            return batch
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    else:
        return batch


def split_path(path):
    """Splits a path to source and filename

    This also handles URLs and Huggingface hub paths, in addition to
    regular paths.

    Arguments
    ---------
    path : str or FetchSource

    Returns
    -------
    str
        Source
    str
        Filename
    """

    def split(src):
        """Core function to split path."""
        if "/" in src:
            return src.rsplit("/", maxsplit=1)
        else:
            # Interpret as path to file in current directory.
            return "./", src

    if isinstance(path, sb.utils.fetching.FetchSource):
        fetch_from, fetch_path = path
        source, filename = split(fetch_path)
        return sb.utils.fetching.FetchSource(fetch_from, source), filename
    else:
        return split(path)


def scalarize(value):
    """Converts a namedtuple or dictionary containing tensors
    to their scalar value

    Arguments
    ---------
    value: dict or namedtuple
        a dictionary or named tuple of tensors

    Returns
    -------
    result: dict
        a result dictionary
    """
    if hasattr(value, "_asdict"):
        value_dict = value._asdict()
    else:
        value_dict = value
    return {key: item_value.item() for key, item_value in value_dict.items()}


def unsqueeze_as(x, target):
    """Reshape the tensor to be of a shape compatible with the target
    tensor, only valid if x.dim() <= y.dim()

    Arguments
    ---------
    x: torch.Tensor
        the original tensor
    target: torch.Tensor
        the tensor whose shape

    Returns
    -------
    result: torch.Tensor
        a view of tensor x reshaped to a shape compatible with y
    """
    return x.view(x.shape + (1,) * (target.dim() - x.dim()))


def pad_divisible(tensor, length=None, factor=2, len_dim=1, pad_value=0):
    """Adds extra padding to the specified dimension of a tensor to make
    it divisible  by the specified factor. This is useful when passing
    variable-length sequences to downsampling UNets or other similar
    architectures in which inputs are expected to be divisible by the
    downsampling factor

    Arguments
    ---------
    tensor: torch.Tensor
        the tensor to be padded, of arbitrary dimension

    length: torch.Tensor
        a 1-D tensor of relative lengths

    factor: int
        the divisibility factor

    len_dim: int
        the index of the dimension used as the length

    pad_value: int
        the value with which outputs will be padded

    Returns
    -------
    tensor_padded: torch.Tensor
        the tensor, with additional padding if required
    length: torch.Tensor
        the adjusted length tensor, if provided

    Example
    -------
    >>> x = torch.tensor([[1, 2, 3, 4],
    ...                   [5, 6, 0, 0]])
    >>> lens = torch.tensor([1., .5])
    >>> x_pad, lens_pad = pad_divisible(x, length=lens, factor=5)
    >>> x_pad
    tensor([[1, 2, 3, 4, 0],
            [5, 6, 0, 0, 0]])
    >>> lens_pad
    tensor([0.8000, 0.4000])
    """
    time_dim = tensor.size(len_dim)

    desired_time_dim = time_dim
    gap = time_dim % factor
    if gap > 0:
        desired_time_dim += factor - gap

    new_shape = list(tensor.shape)
    new_shape[len_dim] = desired_time_dim

    tensor_padded, _ = pad_right_to(tensor, new_shape, value=pad_value)

    # Adjust lengths to the new dimension, post-padding
    if length is not None:
        length = length * (time_dim / desired_time_dim)

    return tensor_padded, length


def trim_to_shape(tensor, shape):
    """Trims the specified tensor to match the specified shape

    Arguments
    ---------
    tensor: torch.Tensor
        a tensor
    shape: enumerable
        the desired shape

    Returns
    -------
    tensor: torch.Tensor
        the trimmed tensor
    """
    for dim, size in enumerate(shape):
        tensor = tensor.narrow(dim, 0, size)
    return tensor


def trim_as(tensor, other):
    """Trims the specified tensor to match the shape of another
    tensor (at most)

    Arguments
    ---------
    tensor: torch.Tensor:
        a tensor
    other: torch.Tensor
        the tensor whose shape to match

    Returns
    -------
    tensor: torch.Tensor
        the trimmed tensor
    """
    return trim_to_shape(tensor, other.shape)


def match_shape(tensor, other):
    """A swiss-army-knife helper function to match the shape of a tensor to
    match that of another tensor - useful for masks, etc.

    Arguments
    ---------
    tensor: torch.Tensor:
        a tensor
    other: torch.Tensor
        the tensor whose shape to match

    Returns
    -------
    tensor: torch.Tensor
        the tensor with matching shape
    """
    result = unsqueeze_as(tensor, other)
    result = result.expand_as(other)
    result = trim_as(result, other)
    return result


def batch_shuffle(items, batch_size):
    """Shuffles batches of fixed size within a sequence

    Arguments
    ---------
    items: sequence
        a tensor or an indexable sequence, such as a list
    batch_size: int
        the batch size

    Returns
    -------
    items: sequence
        the original items. If a tensor was passed, a tensor
        will be returned. Otherwise, it will return a list
    """
    batch_count = math.floor(len(items) / batch_size)
    batches = torch.randperm(batch_count)
    batch_idx = (
        batches.unsqueeze(-1).expand(batch_count, batch_size) * batch_size
    )
    batch_offset = torch.arange(batch_size).unsqueeze(0)
    batch_idx += batch_offset
    tail = torch.arange(batch_count * batch_size, len(items))
    batch_idx = torch.concat((batch_idx.flatten(), tail))
    if torch.is_tensor(items):
        result = items[batch_idx]
    else:
        result = [items[idx] for idx in batch_idx]
    return result


def concat_padded_features(
    feats, lens, dim=1, feats_slice_start=None, feats_slice_end=None
):
    """Concatenates multiple padded feature tensors into a single
    padded tensor in a vectorized manner without including the
    padding in the final tensor, adding padding only at the end.
    The function supports optional relative sicing of the tensors.

    One possible use case is to concatenate batches of spectrograms
    or audio.

    Arguments
    ---------
    feats: list
        a list of padded tensors
    lens: list
        a list of length tensors
    dim: int
        The dimension on which to perform concatenation
    feats_slice_start: list
        offsets, relative to the beginning of the sequence, for each
        of the tensors being concatenated. This is useful if only
        a subsequence of some slices is included
    feats_slice_end: list
        offsets, relative to the end of the sequence, for each
        of the tensors being concatenated. This is useful if only
        a subsequence of some slices is included

    Returns
    -------
    out: torch.Tensor
        a concatenated tensor
    """
    first_item = feats[0]
    item_lengths = torch.tensor([item.size(dim) for item in feats]).to(
        first_item.device
    )
    lens = torch.concat([len_rel.unsqueeze(0) for len_rel in lens])
    lens_abs = (lens * item_lengths.unsqueeze(-1)).int()

    feats_slice_start = _offset_to_tensor(feats_slice_start, lens_abs)
    feats_slice_end = _offset_to_tensor(feats_slice_end, lens_abs)

    out_start, out_end = _lens_to_boundaries(
        lens_abs, feats_slice_start, feats_slice_end, cumulative=True
    )
    in_start, in_end = _lens_to_boundaries(
        lens_abs, feats_slice_start, feats_slice_end, cumulative=False
    )
    total_length = out_end.max().int().item()

    out_shape = list(first_item.shape)
    out_shape[dim] = total_length
    out = torch.zeros(out_shape).to(first_item.device)
    for item, item_in_start, item_in_end, item_out_start, item_out_end in zip(
        feats, in_start, in_end, out_start, out_end
    ):
        in_mask = _boundaries_to_mask(item, item_in_start, item_in_end, dim)
        out_mask = _boundaries_to_mask(out, item_out_start, item_out_end, dim)
        out[out_mask] = item[in_mask]

    out_lens = out_end[-1, :].float() / total_length

    return out, out_lens


def _offset_to_tensor(offset, lengths):
    """Converts a variety of offset representations to a component x batch tensor,
    used by concat_padded_features. offset can be a tensor, a list of tensors (where
    each element is a tensor of relative offsets similar to lengths), a list of floats
    (in which case all batch elements are presumed to have the same offset)

    Arguments
    ---------
    offset: list|Tensor
        a list or tensor of offsets
    lengths: torch.Tensor
        a length tensor

    Returns
    -------
    result: torch.Tensor
        a tensor of offsets
    """
    if offset is None:
        result = None
    elif torch.is_tensor(offset):
        result = offset
    elif isinstance(offset, Number):
        result = torch.ones_like(lengths) * offset
    elif isinstance(offset, list):
        if isinstance(offset[0], Number):
            result = torch.tensor(offset).unsqueeze(-1).to(lengths.device)
        else:
            result = torch.concat([item.unsqueeze(0) for item in offset])
    else:
        raise ValueError(
            "The offset must be a number, a tensor or a list of tensors"
        )
    return result


def _lens_to_boundaries(
    lengths, slice_start=None, slice_end=None, cumulative=True
):
    """Converts a tensor of lengths to a tensor of start and end
    boundaries, used for concat_padded_features

    Arguments
    ---------
    lengths: torch.Tensor
        a (component x batch) tensor of absolute lengths
    slice_start: torch.Tensor
        a (component x batch) tensor of relative start offsets
    slice_end: torch.Tensor
        a (component x batch) tensor of relative end offsets
    cumulative: True
        if true, the start of a given component is assumed to
        be at the end of the previous component.
        if false, all components start at the beginning of the
        length dimension

    Returns
    -------
    start: torch.Tensor
        the starting boundary
    end: torch.Tensor
        the ending boundary
    """
    batch_size = lengths.size(-1)
    batch_padding = torch.zeros((1, batch_size)).int().to(lengths.device)

    if slice_start is None:
        start_offset = torch.tensor(0).to(lengths.device)
    else:
        start_offset = (lengths * slice_start).floor().int()

    if slice_end is None:
        end_offset = torch.tensor(0).to(lengths.device)
    else:
        end_offset = (lengths * slice_end).floor().int()

    if cumulative:
        effective_lengths = lengths - start_offset - end_offset
        effective_lengths_zpad = torch.concat(
            [batch_padding, effective_lengths], dim=0
        )

        start = effective_lengths_zpad.cumsum(dim=0)[:-1, :]
    else:
        start = torch.zeros(*lengths.shape).to(lengths.device)
    start += start_offset
    end = start + lengths - end_offset
    return start, end


def _boundaries_to_mask(target, start, end, len_dim=1):
    """For a given features tensor and tensors of start and end indexes,
    computes the corresponding Boolean mask

    Arguments
    ---------
    target: torch.Tensor
        the target tensor
    start: torch.Tensor
        the tensor indicating the starting positions along the length
        dimension within each batch
    end: torch.Tensor
        the tensor indicating the final positions within each batch
    len_dim: int
        the dimension used as the length

    Returns
    -------
    mask: torch.Tensor
        a Boolean mask of the same shape as target
    """
    out_range = length_range(target, len_dim)
    feats_dim = target.dim()
    item_start = unsqueeze_1d(start, feats_dim, 0)
    item_end = unsqueeze_1d(end, feats_dim, 0)
    mask = (item_start <= out_range) & (out_range < item_end)
    return mask


def unsqueeze_1d(value, dim, value_dim):
    """Unsqueezes a 1-D tensor to the specified number of
    dimension preserving one dimension and creating "dummy" dimensions
    elsewhere

    Arguments
    ---------
    value: torch.Tensor
        A 1-D tensor
    dim: int
        the number of dimension
    value_dim: int
        the dimension that the value tensor represents

    Returns
    -------
    result: torch.Tensor
        a dim-dimensional tensor
    """
    unsqueeze_dim = [None] * dim
    unsqueeze_dim[value_dim] = ...
    return value[unsqueeze_dim]


def length_range(feats, len_dim):
    """Creates a tensor with a range in a single dimension to one matching the shape
    of a its tensor

    Arguments
    ---------
    feats: torch.Tensor
        a features tensor of arbitrary shape
    len_dim: torch.Tensor
        the dimension used as length

    Returns
    -------
    result: torch.Tensor
        a tensor matching the shape of feats with an 0 to max-length range along
        the length dimension repeated across other dimensions
    """
    max_len = feats.size(len_dim)
    feats_range = torch.arange(max_len).to(feats.device)
    out = unsqueeze_1d(feats_range, feats.dim(), len_dim)
    repeat_dim = [
        feats_size // out_size
        for feats_size, out_size in zip(feats.shape, out.shape)
    ]
    return out.repeat(*repeat_dim)


def non_batch_dims(sample):
    """Returns all dimensions of the specified tensor
    except the batch dimension

    Arguments
    ---------
    sample: torch.Tensor
        an arbitrary tensor

    Returns
    -------
    dims: list
        a list of dimensions
    """
    return list(range(1, sample.dim()))


def masked_mean(sample, mask=None):
    """A metric function that computes the mean of each sample, excluding
    padding

    Arguments
    ---------
    sample: torch.Tensor
        a tensor of spectrograms
    mask: torch.Tensor
        a length mask

    Returns
    -------
    result: torch.Tensor
        a tensor fo means
    """
    if mask is None:
        mask = torch.ones_like(sample).bool()
    dims = non_batch_dims(sample)
    return (sample * mask).sum(dim=dims) / mask.expand_as(sample).sum(dim=dims)


def masked_std(sample, mask=None):
    """A metric function that computes the standard deviation of each
    sample, excluding padding

    Arguments
    ---------
    sample: torch.Tensor
        a tensor of spectrograms
    mask: torch.Tensor
        a length mask

    Returns
    -------
    result: torch.Tensor
        a tensor fo means
    """
    if mask is None:
        mask = torch.ones_like(sample).bool()
    dims = non_batch_dims(sample)
    mean = unsqueeze_as(masked_mean(sample, mask), sample)
    diff_sq = ((sample - mean) * mask) ** 2
    return (
        diff_sq.sum(dim=dims) / (mask.expand_as(diff_sq).sum(dim=dims) - 1)
    ).sqrt()


def masked_min(sample, mask=None):
    """A metric function that computes the minimum of each sample

    Arguments
    ---------
    sample: torch.Tensor
        a tensor of spectrograms
    mask: torch.Tensor
        a length mask

    Returns
    -------
    result: torch.Tensor
        a tensor fo means
    """
    if mask is None:
        mask = torch.ones_like(sample).bool()
    dims = non_batch_dims(sample)
    return sample.masked_fill(~mask.bool(), torch.inf).amin(dim=dims)


def masked_max(sample, mask=None):
    """A metric function that computes the minimum of each sample

    Arguments
    ---------
    sample: torch.Tensor
        a tensor of spectrograms
    mask: torch.Tensor
        a length mask

    Returns
    -------
    result: torch.Tensor
        a tensor fo means
    """
    if mask is None:
        mask = torch.ones_like(sample).bool()
    dims = non_batch_dims(sample)
    return sample.masked_fill(~mask.bool(), -torch.inf).amax(dim=dims)


def dist_stats(sample, mask=None):
    """Returns standard distribution statistics (mean, std, min, max)

    Arguments
    ---------
    sample: torch.Tensor
        a tensor of spectrograms
    mask: torch.Tensor
        a length mask

    Returns
    -------
    result: torch.Tensor
        a tensor fo means
    """
    return {
        "mean": masked_mean(sample, mask),
        "std": masked_std(sample, mask),
        "min": masked_min(sample, mask),
        "max": masked_max(sample, mask),
    }


def dict_value_combinations(values):
    """Returns all possible key-value combinations from
    the given dictionary

    Arguments
    ---------
    values: dict
        A dictionary with lists of values as values
        Example:
        {
            "digit": [1,2,3],
            "speaker": [10, 20]
        }

    Returns
    -------
    result: list
        a list of dictionaries in which each dictionary
        is a possible permutations
    """
    return [
        item
        for item in dict_value_combinations_gen(values, values.keys())
        if len(item) == len(values)
    ]


def dict_value_combinations_gen(values, keys):
    """Returns a generation of permutations of the specified
    values dictionary

    Arguments
    ---------
    values: dict
        A dictionary with lists of values as values
        Example:
        {
            "digit": [1,2,3],
            "speaker": [10, 20]
        }
    keys: list
        the keys to consider

    Returns
    -------
    result: generator
        a generator of dictionaries in which each dictionary
        is a possible permutation
    """
    if not keys:
        return
    key, *rest = keys
    key_values = values[key]
    for value in key_values:
        curr = {key: value}
        for sub in dict_value_combinations_gen(values, rest):
            item = dict(curr)
            item.update(sub)
            yield item
        else:
            yield curr
