"""
Data reading and writing

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Ju-Chieh Chou 2020
 * Samuele Cornell 2020
"""
import os
import csv
import torch
import logging
import pickle
import hashlib
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)


def read_wav(waveforms_obj):
    files = waveforms_obj["files"]
    if not isinstance(files, list):
        files = [files]

    waveforms = []
    for f in files:
        if (
            "start" not in waveforms_obj.keys()
            or "stop" not in waveforms_obj.keys()
        ):
            tmp, fs = torchaudio.load(f)
            waveforms.append(tmp)
        else:
            num_frames = waveforms_obj["stop"] - waveforms_obj["start"]
            offset = waveforms_obj["start"]
            tmp, fs = torchaudio.load(f, num_frames=num_frames, offset=offset)
            waveforms.append(tmp)

    out = torch.cat(waveforms, 0)
    if out.size(0) != 1:
        logger.error(
            "Multichannel audio currently not supported", exc_info=True
        )
        raise NotImplementedError

    return out.squeeze(0)


def load_pickle(pickle_path):
    with open(pickle_path, "r") as f:
        out = pickle.load(f)
    return out


def to_floatTensor(x: (list, tuple, np.ndarray)):
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    else:
        return torch.tensor(x, dtype=torch.float)


def to_doubleTensor(x: (list, tuple, np.ndarray)):
    if isinstance(x, torch.Tensor):
        return x.double()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).double()
    else:
        return torch.tensor(x, dtype=torch.double)


def to_longTensor(x: (list, tuple, np.ndarray)):
    if isinstance(x, torch.Tensor):
        return x.long()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).long()
    else:
        return torch.tensor(x, dtype=torch.long)


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """
    Creates a binary mask for each sequence.
    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3
    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.
    Returns
    -------
    mask : The binary mask
    Example:
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


def get_md5(file):
    """
    Get the md5 checksum of an input file
    Arguments
    ---------
    file : str
        Path to file for which compute the checksum
    Returns
    -------
    md5
        Checksum for the given filepath
    Example
    -------
    >>> get_md5('samples/audio_samples/example1.wav')
    'c482d0081ca35302d30d12f1136c34e5'
    """
    # Lets read stuff in 64kb chunks!
    BUF_SIZE = 65536
    md5 = hashlib.md5()
    # Computing md5
    with open(file, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def save_md5(files, out_file):
    """
    Saves the md5 of a list of input files as a pickled dict into a file.
    Arguments
    ---------
    files : list
        List of input files from which we will compute the md5.
    outfile : str
        The path where to store the output pkl file.
    Returns
    -------
    None
    Example:
    >>> files = ['samples/audio_samples/example1.wav']
    >>> tmpdir = getfixture('tmpdir')
    >>> save_md5(files, os.path.join(tmpdir, "md5.pkl"))
    """
    # Initialization of the dictionary
    md5_dict = {}
    # Computing md5 for all the files in the list
    for file in files:
        md5_dict[file] = get_md5(file)
    # Saving dictionary in pkl format
    save_pkl(md5_dict, out_file)


def save_pkl(obj, file):
    """
    Save an object in pkl format.
    Arguments
    ---------
    obj : object
        Object to save in pkl format
    file : str
        Path to the output file
    sampling_rate : int
        Sampling rate of the audio file, TODO: this is not used?
    Example:
    >>> tmpfile = os.path.join(getfixture('tmpdir'), "example.pkl")
    >>> save_pkl([1, 2, 3, 4, 5], tmpfile)
    >>> load_pkl(tmpfile)
    [1, 2, 3, 4, 5]
    """
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    """
    Loads a pkl file
    For an example, see `save_pkl`
    Arguments
    ---------
    file : str
        Path to the input pkl file.
    Returns
    -------
    The loaded object
    """
    with open(file, "rb") as f:
        return pickle.load(f)


def prepend_bos_token(label, bos_index):
    """Create labels with <bos> token at the beginning.
    Arguments
    ---------
    label : torch.IntTensor
        Containing the original labels. Must be of size: [batch_size, max_length]
    bos_index : int
        The index for <bos> token.
    Returns
    -------
    new_label : The new label with <bos> at the beginning.
    Example:
    >>> label=torch.LongTensor([[1,0,0], [2,3,0], [4,5,6]])
    >>> new_label=prepend_bos_token(label, bos_index=7)
    >>> new_label
    tensor([[7, 1, 0, 0],
            [7, 2, 3, 0],
            [7, 4, 5, 6]])
    """
    new_label = label.long().clone()
    batch_size = label.shape[0]

    bos = new_label.new_zeros(batch_size, 1).fill_(bos_index)
    new_label = torch.cat([bos, new_label], dim=1)
    return new_label


def append_eos_token(label, length, eos_index):
    """Create labels with <eos> token appended.
    Arguments
    ---------
    label : torch.IntTensor
        Containing the original labels. Must be of size: [batch_size, max_length]
    length : torch.LongTensor
        Cotaining the original length of each label sequences. Must be 1D.
    eos_index : int
        The index for <eos> token.
    Returns
    -------
    new_label : The new label with <eos> appended.
    Example:
    >>> label=torch.IntTensor([[1,0,0], [2,3,0], [4,5,6]])
    >>> length=torch.LongTensor([1,2,3])
    >>> new_label=append_eos_token(label, length, eos_index=7)
    >>> new_label
    tensor([[1, 7, 0, 0],
            [2, 3, 7, 0],
            [4, 5, 6, 7]], dtype=torch.int32)
    """
    new_label = label.int().clone()
    batch_size = label.shape[0]

    pad = new_label.new_zeros(batch_size, 1)
    new_label = torch.cat([new_label, pad], dim=1)
    new_label[torch.arange(batch_size), length.long()] = eos_index
    return new_label


def merge_char(sequences, space="_"):
    """Merge characters sequences into word sequences.
    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains character sequence.
    space : string
        The token represents space. Default: _
    Returns
    -------
    The list contain word sequences for each sentence.
    Example:
    >>> sequences = [["a", "b", "_", "c", "_", "d", "e"], ["e", "f", "g", "_", "h", "i"]]
    >>> results = merge_char(sequences)
    >>> results
    [['ab', 'c', 'de'], ['efg', 'hi']]
    """
    results = []
    for seq in sequences:
        words = "".join(seq).split("_")
        results.append(words)
    return results


def merge_csvs(data_folder, csv_lst, merged_csv):
    """Merging several csv files into one file.
    Arguments
    ---------
    data_folder : string
        The folder to store csv files to be merged and after merging.
    csv_lst : list
        filenames of csv file to be merged.
    merged_csv : string
        The filename to write the merged csv file.
    Example:
    >>> merge_csvs("samples/audio_samples/",
    ... ["csv_example.csv", "csv_example2.csv"],
    ... "test_csv_merge.csv")
    """
    write_path = os.path.join(data_folder, merged_csv)
    if os.path.isfile(write_path):
        logger.info("Skipping merging. Completed in previous run.")

    with open(os.path.join(data_folder, csv_lst[0])) as f:
        header = f.readline()
    lines = []
    for csv_file in csv_lst:
        with open(os.path.join(data_folder, csv_file)) as f:
            for i, line in enumerate(f):
                if i == 0:
                    # Checking header
                    if line != header:
                        raise ValueError(
                            "Different header for " f"{csv_lst[0]} and {csv}."
                        )
                    continue
                lines.append(line)
    with open(write_path, "w") as f:
        f.write(header)
        for line in lines:
            f.write(line)
    logger.info(f"{write_path} is created.")


def split_word(sequences, space="_"):
    """Split word sequences into character sequences.
    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains words sequence.
    space : string
        The token represents space. Default: _
    Returns
    -------
    The list contain word sequences for each sentence.
    Example:
    >>> sequences = [['ab', 'c', 'de'], ['efg', 'hi']]
    >>> results = split_word(sequences)
    >>> results
    [['a', 'b', '_', 'c', '_', 'd', 'e'], ['e', 'f', 'g', '_', 'h', 'i']]
    """
    results = []
    for seq in sequences:
        chars = list("_".join(seq))
        results.append(chars)
    return results
