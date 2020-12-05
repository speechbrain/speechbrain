"""
Data reading and writing

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Ju-Chieh Chou 2020
 * Samuele Cornell 2020
"""

import os
import torch
import logging
import soundfile as sf
import numpy as np
import pickle
import hashlib
import multiprocessing as mp
import csv
import time
import torchaudio
import json

logger = logging.getLogger(__name__)


def _recursive_format(data, replacements):
    # Data: dict or list, replacements : dict
    # Replaces string keys in replacements by their values
    # at all levels of data (in str values)
    # Works in-place.
    if isinstance(data, dict):
        for key, item in data.items():
            if isinstance(item, dict) or isinstance(item, list):
                _recursive_format(item, replacements)
            elif isinstance(item, str):
                data[key] = item.format_map(replacements)
            # If not dict, list or str, do nothing
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) or isinstance(item, list):
                _recursive_format(item, replacements)
            elif isinstance(item, str):
                data[i] = item.format_map(replacements)
            # If not dict, list or str, do nothing


def load_json(json_path, replacements={}):
    """Loads JSON and recursively formats string values

    Arguments
    ----------
    json_path : str
        Path to json file
    replacements : dict
        Optional dict:
        e.g. {"data_folder": "/home/speechbrain/data"}
        This is used to recursively format all string values in the data

    Returns
    -------
    dict
        JSON data with replacements applied

    Example
    -------
    >>> json_spec = '''
    ... { "ex1": {"files": ["{ROOT}/mic1/ex1.wav", "{ROOT}/mic2/ex1.wav"], "id": 1},
    ...   "ex2": {"files": [{"spk1": "{ROOT}/ex2.wav"}, {"spk2": "{ROOT}/ex2.wav"}], "id": 2}
    ... }
    ... '''
    >>> tmpfile = getfixture('tmpdir') / "test.json"
    >>> with open(tmpfile, "w") as fo:
    ...     _ = fo.write(json_spec)
    >>> data = load_json(tmpfile, {"ROOT": "/home"})
    >>> data["ex1"]["files"][0]
    '/home/mic1/ex1.wav'
    >>> data["ex2"]["files"][1]["spk2"]
    '/home/ex2.wav'

    """
    # TODO: Example / unittest
    with open(json_path, "r") as f:
        out_json = json.load(f)
    _recursive_format(out_json, replacements)
    return out_json


def read_audio(waveforms_obj):
    """General audio loading, based on custom notation

    Expected use case is specifically in conjunction with Datasets
    specified by JSON.

    The custom notation:

    The annotation can be just a path to a file:
    "/path/to/wav1.wav"

    Or can specify more options in a dict:
    {"file": "/path/to/wav2.wav",
    "start": 8000,
    "stop": 16000
    }

    Arguments
    ----------
    waveforms_obj : str, dict
        Audio reading annotation, see above for format

    Returns
    -------
    torch.Tensor
        audio tensor with shape: (samples, )

    Example
    -------
    >>> dummywav = torch.rand(16000)
    >>> import os
    >>> tmpfile = os.path.join(str(getfixture('tmpdir')),  "wave.wav")
    >>> import soundfile as sf
    >>> sf.write(tmpfile, dummywav, 16000, subtype="float")
    >>> asr_example = { "wav": tmpfile, "spk_id": "foo", "words": "foo bar"}
    >>> loaded = read_audio(asr_example["wav"])
    >>> torch.all(torch.eq(loaded, dummywav))
    tensor(True)
    """
    if isinstance(waveforms_obj, str):
        audio, _ = torchaudio.load(waveforms_obj)
        return audio.squeeze(0)

    path = waveforms_obj["file"]
    start = waveforms_obj.get("start", 0)
    # Default stop to start -> if not specified, num_frames becomes 0,
    # which is the torchaudio default
    stop = waveforms_obj.get("stop", start)
    num_frames = stop - start
    audio, fs = torchaudio.load(path, num_frames=num_frames, offset=start)
    return audio.squeeze(0)


def read_audio_multichannel(waveforms_obj):
    """General audio loading, based on custom notation

    Expected use case is specifically in conjunction with Datasets
    specified by JSON.

    The custom notation:

    The annotation can be just a path to a file:
    "/path/to/wav1.wav"

    Multiple (possibly multi-channel) files can be specified, as long as they
    have the same length:
    {"files": [
        "/path/to/wav1.wav",
        "/path/to/wav2.wav"
        ]
    }

    Or you can specify a single file more succintly:
    {"files": "/path/to/wav2.wav"}

    Offset number samples and stop number samples also can be specified to read
    only a segment within the files.
    {"files": [
        "/path/to/wav1.wav",
        "/path/to/wav2.wav"
        ]
    "start": 8000
    "stop": 16000
    }

    Arguments
    ----------
    waveforms_obj : str, dict
        Audio reading annotation, see above for format

    Returns
    -------
    torch.Tensor
        audio tensor with shape: (samples, )

    Example
    -------
    >>> dummywav = torch.rand(16000, 2)
    >>> import os
    >>> tmpfile = os.path.join(str(getfixture('tmpdir')),  "wave.wav")
    >>> import soundfile as sf
    >>> sf.write(tmpfile, dummywav, 16000, subtype="float")
    >>> asr_example = { "wav": tmpfile, "spk_id": "foo", "words": "foo bar"}
    >>> loaded = read_audio(asr_example["wav"])
    >>> torch.all(torch.eq(loaded.transpose(0, 1), dummywav))
    tensor(True)
    """
    # TODO: Example / unittest
    if isinstance(waveforms_obj, str):
        audio, _ = torchaudio.load(waveforms_obj)
        return audio

    files = waveforms_obj["files"]
    if not isinstance(files, list):
        files = [files]

    waveforms = []
    start = waveforms_obj.get("start", 0)
    # Default stop to start -> if not specified, num_frames becomes 0,
    # which is the torchaudio default
    stop = waveforms_obj.get("stop", start)
    num_frames = stop - start
    for f in files:
        audio, fs = torchaudio.load(f, num_frames=num_frames, offset=start)
        waveforms.append(audio)

    out = torch.cat(waveforms, 0)
    return out


def load_pickle(pickle_path):
    """
    Utility function for loading .pkl pickle files.

    Parameters
    ----------
    pickle_path : str
        path to pickle file

    Returns
    -------
    out : object
        python object loaded form pickle
    """
    with open(pickle_path, "r") as f:
        out = pickle.load(f)
    return out


def to_floatTensor(x: (list, tuple, np.ndarray)):
    """
    Parameters
    ----------
    x : (list, tuple, np.ndarray)
        input data to be converted to torch float.

    Returns
    -------
    tensor : torch.tensor
        data now in torch.tensor float datatype.
    """
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    else:
        return torch.tensor(x, dtype=torch.float)


def to_doubleTensor(x: (list, tuple, np.ndarray)):
    """
    Parameters
    ----------
    x : (list, tuple, np.ndarray)
        input data to be converted to torch double.

    Returns
    -------
    tensor : torch.tensor
        data now in torch.tensor double datatype.
    """
    if isinstance(x, torch.Tensor):
        return x.double()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).double()
    else:
        return torch.tensor(x, dtype=torch.double)


def to_longTensor(x: (list, tuple, np.ndarray)):
    """
    Parameters
    ----------
    x : (list, tuple, np.ndarray)
        input data to be converted to torch long.

    Returns
    -------
    tensor : torch.tensor
        data now in torch.tensor long datatype.
    """
    if isinstance(x, torch.Tensor):
        return x.long()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).long()
    else:
        return torch.tensor(x, dtype=torch.long)


def convert_index_to_lab(batch, ind2lab):
    """
    Convert a batch of integer IDs to string labels

    Arguments
    ---------
    batch : list
        List of lists, a batch of sequences
    ind2lab : dict
        Mapping from integer IDs to labels

    Returns
    -------
    list
        List of lists, same size as batch, with labels from ind2lab

    Example
    -------
    >>> ind2lab = {1: "h", 2: "e", 3: "l", 4: "o"}
    >>> out = convert_index_to_lab([[4,1], [1,2,3,3,4]], ind2lab)
    >>> for seq in out:
    ...     print("".join(seq))
    oh
    hello
    """
    return [[ind2lab[int(index)] for index in seq] for seq in batch]


def relative_time_to_absolute(batch, relative_lens, rate):
    """
    Converts SpeechBrain style relative length to absolute duration

    Operates on batch level.

    Arguments
    ---------
    batch : torch.tensor
        Sequences to determine duration for.
    relative_lens : torch.tensor
        The relative length of each sequence in batch. The longest sequence in
        the batch needs to have relative length 1.0.
    rate : float
        The rate at which sequence elements occur in real world time. Sample
        rate, if batch is raw wavs (recommended) or 1/frame_shift if batch is
        features. This has to have 1/s as the unit.

    Returns
    ------:
    torch.tensor
        Duration of each sequence in seconds.

    Example
    -------
    >>> batch = torch.ones(2, 16000)
    >>> relative_lens = torch.tensor([3./4., 1.0])
    >>> rate = 16000
    >>> print(relative_time_to_absolute(batch, relative_lens, rate))
    tensor([0.7500, 1.0000])
    """
    max_len = batch.shape[1]
    durations = torch.round(relative_lens * max_len) / rate
    return durations


class IterativeCSVWriter:
    """Write CSV files a line at a time.

    Arguments
    ---------
    outstream : file-object
        A writeable stream
    data_fields : list
        List of the optional keys to write. Each key will be expanded to the
        SpeechBrain format, producing three fields: key, key_format, key_opts

    Example
    -------
    >>> import io
    >>> f = io.StringIO()
    >>> writer = IterativeCSVWriter(f, ["phn"])
    >>> print(f.getvalue())
    ID,duration,phn,phn_format,phn_opts
    >>> writer.write("UTT1",2.5,"sil hh ee ll ll oo sil","string","")
    >>> print(f.getvalue())
    ID,duration,phn,phn_format,phn_opts
    UTT1,2.5,sil hh ee ll ll oo sil,string,
    >>> writer.write(ID="UTT2",phn="sil ww oo rr ll dd sil",phn_format="string")
    >>> print(f.getvalue())
    ID,duration,phn,phn_format,phn_opts
    UTT1,2.5,sil hh ee ll ll oo sil,string,
    UTT2,,sil ww oo rr ll dd sil,string,
    >>> writer.set_default('phn_format', 'string')
    >>> writer.write_batch(ID=["UTT3","UTT4"],phn=["ff oo oo", "bb aa rr"])
    >>> print(f.getvalue())
    ID,duration,phn,phn_format,phn_opts
    UTT1,2.5,sil hh ee ll ll oo sil,string,
    UTT2,,sil ww oo rr ll dd sil,string,
    UTT3,,ff oo oo,string,
    UTT4,,bb aa rr,string,
    """

    def __init__(self, outstream, data_fields, defaults={}):
        self._outstream = outstream
        self.fields = ["ID", "duration"] + self._expand_data_fields(data_fields)
        self.defaults = defaults
        self._outstream.write(",".join(self.fields))

    def set_default(self, field, value):
        """
        Sets a default value for the given CSV field.

        Arguments
        ---------
        field : str
            A field in the CSV
        value
            The default value
        """
        if field not in self.fields:
            raise ValueError(f"{field} is not a field in this CSV!")
        self.defaults[field] = value

    def write(self, *args, **kwargs):
        """
        Writes one data line into the CSV.

        Arguments
        ---------
        *args
            Supply every field with a value in positional form OR
        **kwargs
            Supply certain fields by key. The ID field is mandatory for all
            lines, but others can be left empty.
        """
        if args and kwargs:
            raise ValueError(
                "Use either positional fields or named fields, but not both."
            )
        if args:
            if len(args) != len(self.fields):
                raise ValueError("Need consistent fields")
            to_write = [str(arg) for arg in args]
        if kwargs:
            if "ID" not in kwargs:
                raise ValueError("I'll need to see some ID")
            full_vals = self.defaults.copy()
            full_vals.update(kwargs)
            to_write = [str(full_vals.get(field, "")) for field in self.fields]
        self._outstream.write("\n")
        self._outstream.write(",".join(to_write))

    def write_batch(self, *args, **kwargs):
        """
        Writes a batch of lines into the CSV

        Here each argument should be a list with the same length.

        Arguments
        ---------
        *args
            Supply every field with a value in positional form OR
        **kwargs
            Supply certain fields by key. The ID field is mandatory for all
            lines, but others can be left empty.
        """
        if args and kwargs:
            raise ValueError(
                "Use either positional fields or named fields, but not both."
            )
        if args:
            if len(args) != len(self.fields):
                raise ValueError("Need consistent fields")
            for arg_row in zip(*args):
                self.write(*arg_row)
        if kwargs:
            if "ID" not in kwargs:
                raise ValueError("I'll need to see some ID")
            keys = kwargs.keys()
            for value_row in zip(*kwargs.values()):
                kwarg_row = dict(zip(keys, value_row))
                self.write(**kwarg_row)

    @staticmethod
    def _expand_data_fields(data_fields):
        expanded = []
        for data_field in data_fields:
            expanded.append(data_field)
            expanded.append(data_field + "_format")
            expanded.append(data_field + "_opts")
        return expanded


# TODO: Consider making less complex
def read_wav_soundfile(file, data_options={}, lab2ind=None):  # noqa: C901
    """
    Read wav audio files with soundfile.

    Arguments
    ---------
    file : str
        The filepath to the file to read
    data_options : dict
        a dictionary containing options for the reader.
    lab2ind : dict, None
        a dictionary for converting labels to indices

    Returns
    -------
    numpy.array
        An array with the read signal

    Example
    -------
    >>> read_wav_soundfile('samples/audio_samples/example1.wav')[0:2]
    array([0.00024414, 0.00018311], dtype=float32)
    """
    # Option initialization
    start = 0
    stop = None
    endian = None
    subtype = None
    channels = None
    samplerate = None

    # List of possible options
    possible_options = [
        "start",
        "stop",
        "samplerate",
        "endian",
        "subtype",
        "channels",
    ]

    # Check if the specified options are supported
    for opt in data_options:
        if opt not in possible_options:

            err_msg = "%s is not a valid options. Valid options are %s." % (
                opt,
                possible_options,
            )

            logger.error(err_msg, exc_info=True)

    # Managing start option
    if "start" in data_options:
        try:
            start = int(data_options["start"])
        except Exception:

            err_msg = (
                "The start value for the file %s must be an integer "
                "(e.g start:405)" % (file)
            )

            logger.error(err_msg, exc_info=True)

    # Managing stop option
    if "stop" in data_options:
        try:
            stop = int(data_options["stop"])
        except Exception:

            err_msg = (
                "The stop value for the file %s must be an integer "
                "(e.g stop:405)" % (file)
            )

            logger.error(err_msg, exc_info=True)

    # Managing samplerate option
    if "samplerate" in data_options:
        try:
            samplerate = int(data_options["samplerate"])
        except Exception:

            err_msg = (
                "The sampling rate for the file %s must be an integer "
                "(e.g samplingrate:16000)" % (file)
            )

            logger.error(err_msg, exc_info=True)

    # Managing endian option
    if "endian" in data_options:
        endian = data_options["endian"]

    # Managing subtype option
    if "subtype" in data_options:
        subtype = data_options["subtype"]

    # Managing channels option
    if "channels" in data_options:
        try:
            channels = int(data_options["channels"])
        except Exception:

            err_msg = (
                "The number of channels for the file %s must be an integer "
                "(e.g channels:2)" % (file)
            )

            logger.error(err_msg, exc_info=True)

    # Reading the file with the soundfile reader
    try:
        [signal, fs] = sf.read(
            file,
            start=start,
            stop=stop,
            samplerate=samplerate,
            endian=endian,
            subtype=subtype,
            channels=channels,
        )

        signal = signal.astype("float32")

    except RuntimeError as e:
        err_msg = "cannot read the wav file %s" % (file)
        e.args = (*e.args, err_msg)
        raise

    # Set time_steps always last as last dimension
    if len(signal.shape) > 1:
        signal = signal.transpose()

    return signal


def read_pkl(file, data_options={}, lab2ind=None):
    """
    This function reads tensors store in pkl format.

    Arguments
    ---------
    file : str
        The path to file to read.
    data_options : dict, optional
        A dictionary containing options for the reader.
    lab2ind : dict, optional
        Mapping from label to integer indices.

    Returns
    -------
    numpy.array
        The array containing the read signal
    """

    # Trying to read data
    try:
        with open(file, "rb") as f:
            pkl_element = pickle.load(f)
    except pickle.UnpicklingError:
        err_msg = "cannot read the pkl file %s" % (file)
        raise ValueError(err_msg)

    type_ok = False

    if isinstance(pkl_element, list):

        if isinstance(pkl_element[0], float):
            tensor = torch.FloatTensor(pkl_element)
            type_ok = True

        if isinstance(pkl_element[0], int):
            tensor = torch.LongTensor(pkl_element)
            type_ok = True

        if isinstance(pkl_element[0], str):

            # convert string to integer as specified in self.label_dict
            if lab2ind is not None:
                for index, val in enumerate(pkl_element):
                    pkl_element[index] = lab2ind[val]

            tensor = torch.LongTensor(pkl_element)
            type_ok = True

        if not (type_ok):
            err_msg = (
                "The pkl file %s can only contain list of integers, "
                "floats, or strings. Got %s"
            ) % (file, type(pkl_element[0]))
            raise ValueError(err_msg)
    else:
        tensor = pkl_element

    tensor_type = tensor.dtype

    # Conversion to 32 bit (if needed)
    if tensor_type == "float64":
        tensor = tensor.astype("float32")

    if tensor_type == "int64":
        tensor = tensor.astype("int32")

    return tensor


def read_string(string, data_options={}, lab2ind=None):
    """
    This function reads data in string format.

    Arguments
    ---------
    string : str
        String to read
    data_options : dict, optional
        Options for the reader
    lab2ind : dict, optional
        Mapping from label to index

    Returns
    -------
    torch.LongTensor
        The read string in integer indices, if lab2ind is provided, else
    list
        The read string split at each space

    Example
    -------
    >>> read_string('hello world', lab2ind = {"hello":1, "world": 2})
    tensor([1, 2])
    """

    if callable(lab2ind):
        return lab2ind(string)

    # Try decoding string
    try:
        string = string.decode("utf-8")
    except AttributeError:
        pass

    # Splitting elements with ' '
    string = string.split(" ")

    # convert string to integer as specified in self.label_dict
    if lab2ind is not None:
        for index, val in enumerate(string):
            if val not in lab2ind:
                lab2ind[val] = len(lab2ind)

            string[index] = lab2ind[val]

        string = torch.LongTensor(string)

    return string


def read_kaldi_lab(kaldi_ali, kaldi_lab_opts):
    """
    Read labels in kaldi format

    Uses kaldi IO

    Arguments
    ---------
    kaldi_ali : str
        Path to directory where kaldi alignents are stored.
    kaldi_lab_opts : str
        A string that contains the options for reading the kaldi alignments.

    Returns
    -------
    dict
        A dictionary contaning the labels

    Note
    ----
    This depends on kaldi-io-for-python. Install it separately.
    See: https://github.com/vesis84/kaldi-io-for-python

    Example
    -------
    This example requires kaldi files
    ```
    lab_folder = '/home/kaldi/egs/TIMIT/s5/exp/dnn4_pretrain-dbn_dnn_ali'
    read_kaldi_lab(lab_folder, 'ali-to-pdf')
    ```
    """
    # EXTRA TOOLS
    try:
        import kaldi_io
    except ImportError:
        raise ImportError("Could not import kaldi_io. Install it to use this.")
    # Reading the Kaldi labels
    lab = {
        k: v
        for k, v in kaldi_io.read_vec_int_ark(
            "gunzip -c "
            + kaldi_ali
            + "/ali*.gz | "
            + kaldi_lab_opts
            + " "
            + kaldi_ali
            + "/final.mdl ark:- ark:-|",
        )
    }
    return lab


def write_wav_soundfile(data, filename, sampling_rate):
    """
    Can be used to write audio with soundfile

    Expecting data in (time, [channels]) format

    Arguments
    ---------
    data : torch.tensor
        it is the tensor to store as and audio file
    filename : str
        path to file where writing the data
    sampling_rate : int, None
        sampling rate of the audio file

    Returns
    -------
    None

    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> signal = 0.1*torch.rand([16000])
    >>> tmpfile = os.path.join(tmpdir, 'wav_example.wav')
    >>> write_wav_soundfile(signal, tmpfile, sampling_rate=16000)
    """
    if len(data.shape) > 2:
        err_msg = (
            "expected signal in the format (time, channel). Got %s "
            "dimensions instead of two for file %s"
            % (len(data.shape), filename)
        )
        raise ValueError(err_msg)
    if isinstance(data, torch.Tensor):
        if len(data.shape) == 2:
            data = data.transpose(0, 1)
        # Switching to cpu and converting to numpy
        data = data.cpu().numpy()
    # Writing the file
    sf.write(filename, data, sampling_rate)


def write_txt_file(data, filename, sampling_rate=None):
    """
    Write data in text format

    Arguments
    ---------
    data : str, list, torch.tensor, numpy.ndarray
        The data to write in the text file
    filename : str
        Path to file where to write the data
    sampling_rate : None
        Not used, just here for interface compatibility

    Returns
    -------
    None

    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> signal=torch.tensor([1,2,3,4])
    >>> write_txt_file(signal, os.path.join(tmpdir, 'example.txt'))
    """
    del sampling_rate  # Not used.
    # Check if the path of filename exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as fout:
        if isinstance(data, torch.Tensor):
            data = data.tolist()
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if isinstance(data, list):
            for line in data:
                print(line, file=fout)
        if isinstance(data, str):
            print(data, file=fout)


def write_stdout(data, filename=None, sampling_rate=None):
    """
    Write data to standard output

    Arguments
    ---------
    data : str, list, torch.tensor, numpy.ndarray
        The data to write in the text file
    filename : None
        Not used, just here for compatibility
    sampling_rate : None
        Not used, just here for compatibility

    Returns
    -------
    None


    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> signal = torch.tensor([[1,2,3,4]])
    >>> write_stdout(signal, tmpdir + '/example.txt')
    [1, 2, 3, 4]
    """
    # Managing Torch.Tensor
    if isinstance(data, torch.Tensor):
        data = data.tolist()
    # Managing np.ndarray
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, list):
        for line in data:
            print(line)
    if isinstance(data, str):
        print(data)


def save_img(data, filename, sampling_rate=None, logger=None):
    """
    Save a tensor as an image.

    Arguments
    ---------
    data : torch.tensor
        The tensor to write in the text file
    filename : str
        Path where to write the data.
    sampling_rate : int
        Sampling rate of the audio file.

    Returns
    -------
    None

    Note
    ----
    Depends on matplotlib as an extra tool. Install it separately.

    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> signal=torch.rand([100,200])
    >>> try:
    ...     save_img(signal, tmpdir + '/example.png')
    ... except ImportError:
    ...     pass
    """
    # EXTRA TOOLS
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        err_msg = "Cannot import matplotlib. To use this, install matplotlib"
        raise ImportError(err_msg)
    # Checking tensor dimensionality
    if len(data.shape) < 2 or len(data.shape) > 3:
        err_msg = (
            "cannot save image  %s. Save in png format supports 2-D or "
            "3-D tensors only (e.g. [x,y] or [channel,x,y]). "
            "Got %s" % (filename, str(data.shape))
        )
        raise ValueError(err_msg)
    if len(data.shape) == 2:
        N_ch = 1
    else:
        N_ch = data.shape[0]
    # Flipping axis
    data = data.flip([-2])
    for i in range(N_ch):
        if N_ch > 1:
            filename = filename.replace(".png", "_ch_" + str(i) + ".png")
        if N_ch > 1:
            plt.imsave(filename, data[i])
        else:
            plt.imsave(filename, data)


class TensorSaver(torch.nn.Module):
    """
    Save tensors on disk.

    Arguments
    ---------
    save_folder : str
        The folder where the tensors are stored.
    save_format : str, optional
        Default: "pkl"
        The format to use to save the tensor.
        See get_supported_formats() for an overview of
        the supported data formats.
    save_csv : bool, optional
        Default: False
        If True it saves the list of data written in a csv file.
    data_name : str, optional
        Default: "data"
        The name to give to saved data
    parallel_write : bool, optional
        Default: False
        If True it saves the data using parallel processes.
    transpose : bool, optional
        Default: False
        if True it transposes the data matrix
    decibel : bool, optional
        Default: False
        if True it saves the log of the data.

    Example:
    >>> tmpdir = getfixture('tmpdir')
    >>> save_signal = TensorSaver(save_folder=tmpdir, save_format='wav')
    >>> signal = 0.1 * torch.rand([1, 16000])
    >>> save_signal(signal, ['example_random'], torch.ones(1))
    """

    def __init__(
        self,
        save_folder,
        save_format="pkl",
        save_csv=False,
        data_name="data",
        sampling_rate=16000,
        parallel_write=False,
        transpose=False,
        decibel=False,
    ):
        super().__init__()

        self.save_folder = save_folder
        self.save_format = save_format
        self.save_csv = save_csv
        self.data_name = data_name
        self.sampling_rate = sampling_rate
        self.parallel_write = parallel_write
        self.transpose = transpose
        self.decibel = decibel

        # Definition of other variables
        self.supported_formats = self.get_supported_formats()

        # Creating the save folder if it does not exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Check specified format
        if self.save_format not in self.supported_formats:
            err_msg = (
                "the format %s specified in the config file is not "
                "supported. The current version supports %s"
                % (self.save_format, self.supported_formats.keys())
            )
            logger.error(err_msg)

        # Create the csv file (if specified)
        if self.save_csv:
            self.save_csv_path = os.path.join(self.save_folder, "csv.csv")
            open(self.save_csv_path, "w").close()
            self.first_line_csv = True

    def forward(self, data, data_id, data_len):
        """
        Arguments
        ---------
        data : torch.tensor
            batch of audio signals to save
        data_id : list
            list of ids in the batch
        data_len : torch.tensor
            length of each audio signal
        """
        # Convertion to log (if specified)
        if self.decibel:
            data = 10 * data.log10()

        # Writing data on disk (in parallel)
        self.write_batch(data, data_id, data_len)

    def write_batch(self, data, data_id, data_len):
        """
        Saves a batch of data.

        Arguments
        ---------
        data : torch.tensor
            batch of audio signals to save
        data_id : list
            list of ids in the batch
        data_len : torch.tensor
            relative length of each audio signal

        Example
        -------
        >>> save_folder = getfixture('tmpdir')
        >>> save_format = 'wav'
        >>> save_signal=TensorSaver(save_folder, save_format)
        >>> # random signal
        >>> signal=0.1*torch.rand([1,16000])
        >>> # saving
        >>> save_signal.write_batch(signal, ['example_random'], torch.ones(1))
        """

        # Write in parallel all the examples in the batch on disk:
        jobs = []

        # Move time dimension last
        data = data.transpose(1, -1)

        # Multiprocessing on gpu is something we have to fix
        data = data.cpu()

        if self.save_csv:
            csv_f = open(self.save_csv_path, "a")

            if self.first_line_csv:
                line = "ID, duration, %s, %s_format, %s_opts\n" % (
                    self.data_name,
                    self.data_name,
                    self.data_name,
                )
                self.first_line_csv = False

                csv_f.write(line)

        # Processing all the batches in data
        for j in range(data.shape[0]):

            # Selection up to the true data length (without padding)
            actual_size = int(torch.round(data_len[j] * data[j].shape[0]))
            data_save = data[j].narrow(0, 0, actual_size)

            # Transposing the data if needed
            if self.transpose:
                data_save = data_save.transpose(-1, -2)

            # Selection of the needed data writer
            writer = self.supported_formats[self.save_format]["writer"]

            # Output file
            data_file = os.path.join(
                self.save_folder, data_id[j] + "." + self.save_format
            )

            # Writing all the batches in parallel (if paralle_write=True)
            if self.parallel_write:
                p = mp.Process(
                    target=writer,
                    args=(data_save, data_file),
                    kwargs={"sampling_rate": self.sampling_rate},
                )
                p.start()
                jobs.append(p)
            else:
                # Writing data on disk with the selected writer
                writer(
                    data_save, data_file, sampling_rate=self.sampling_rate,
                )

            # Saving csv file
            if self.save_csv:
                line = "%s, %f, %s, %s, ,\n" % (
                    data_id[j],
                    actual_size,  # We are here saving the number of time steps
                    data_file,
                    self.save_format,
                )
                csv_f.write(line)

        # Waiting all jobs to finish
        if self.parallel_write:
            for j in jobs:
                j.join()

        # Closing the csv file
        if self.save_csv:
            csv_f.close()

    @staticmethod
    def get_supported_formats():
        """
        Lists the supported formats and their related writers

        Returns
        -------
        dict
            Maps from file name extensions to dicts which have the keys
            "writer" and "description"

        Example
        -------
        >>> save_folder = getfixture('tmpdir')
        >>> save_format = 'wav'
        >>> # class initialization
        >>> saver = TensorSaver(save_folder, save_format)
        >>> saver.get_supported_formats()['wav']
        {'writer': <function ...>, 'description': ...}
        """

        # Dictionary initialization
        supported_formats = {}

        # Adding sound file supported formats
        sf_formats = sf.available_formats()

        for wav_format in sf_formats.keys():
            wav_format = wav_format.lower()
            supported_formats[wav_format] = {}
            supported_formats[wav_format]["writer"] = write_wav_soundfile
            supported_formats[wav_format]["description"] = sf_formats[
                wav_format.upper()
            ]

        # Adding the other supported formats
        supported_formats["pkl"] = {}
        supported_formats["pkl"]["writer"] = save_pkl
        supported_formats["pkl"]["description"] = "Python binary format"

        supported_formats["txt"] = {}
        supported_formats["txt"]["writer"] = write_txt_file
        supported_formats["txt"]["description"] = "Plain text"

        supported_formats["png"] = {}
        supported_formats["png"]["writer"] = save_img
        supported_formats["png"]["description"] = "image in png format"

        supported_formats["stdout"] = {}
        supported_formats["stdout"]["writer"] = write_stdout
        supported_formats["stdout"]["description"] = "write on stdout"

        return supported_formats


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

    # Deals with the situation where two processes are trying
    # to access the same label dictionary by creating a lock
    count = 100
    while count > 0:
        if os.path.isfile(file + ".lock"):
            time.sleep(1)
            count -= 1
        else:
            break

    try:
        open(file + ".lock", "w").close()
        with open(file, "rb") as f:
            return pickle.load(f)
    finally:
        if os.path.isfile(file + ".lock"):
            os.remove(file + ".lock")


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
