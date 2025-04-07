"""
Data reading and writing.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Ju-Chieh Chou 2020
 * Samuele Cornell 2020
 * Abdel HEBA 2020
 * Gaëlle Laperrière 2021
 * Sahar Ghannay 2021
 * Sylvain de Langen 2022
 * Adel Moumen 2025
"""

import csv
import hashlib
import json
import os
import pickle
import re
import time
from io import BytesIO
from typing import Union

import numpy as np
import torch
import torchaudio

from speechbrain.utils.logger import get_logger
from speechbrain.utils.torch_audio_backend import (
    check_torchaudio_backend,
    validate_backend,
)

check_torchaudio_backend()
logger = get_logger(__name__)


def load_data_json(json_path, replacements={}):
    """Loads JSON and recursively formats string values.

    Arguments
    ---------
    json_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), e.g., {"data_folder": "/home/speechbrain/data"}.
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        JSON data with replacements applied.

    Example
    -------
    >>> json_spec = '''{
    ...   "ex1": {"files": ["{ROOT}/mic1/ex1.wav", "{ROOT}/mic2/ex1.wav"], "id": 1},
    ...   "ex2": {"files": [{"spk1": "{ROOT}/ex2.wav"}, {"spk2": "{ROOT}/ex2.wav"}], "id": 2}
    ... }
    ... '''
    >>> tmpfile = getfixture('tmpdir') / "test.json"
    >>> with open(tmpfile, "w", encoding="utf-8") as fo:
    ...     _ = fo.write(json_spec)
    >>> data = load_data_json(tmpfile, {"ROOT": "/home"})
    >>> data["ex1"]["files"][0]
    '/home/mic1/ex1.wav'
    >>> data["ex2"]["files"][1]["spk2"]
    '/home/ex2.wav'

    """
    with open(json_path, "r", encoding="utf-8") as f:
        out_json = json.load(f)
    _recursive_format(out_json, replacements)
    return out_json


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


def load_data_csv(csv_path, replacements={}):
    """Loads CSV and formats string values.

    Uses the SpeechBrain legacy CSV data format, where the CSV must have an
    'ID' field.
    If there is a field called duration, it is interpreted as a float.
    The rest of the fields are left as they are (legacy _format and _opts fields
    are not used to load the data in any special way).

    Bash-like string replacements with $to_replace are supported.

    Arguments
    ---------
    csv_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), e.g., {"data_folder": "/home/speechbrain/data"}
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        CSV data with replacements applied.

    Example
    -------
    >>> csv_spec = '''ID,duration,wav_path
    ... utt1,1.45,$data_folder/utt1.wav
    ... utt2,2.0,$data_folder/utt2.wav
    ... '''
    >>> tmpfile = getfixture("tmpdir") / "test.csv"
    >>> with open(tmpfile, "w", encoding="utf-8") as fo:
    ...     _ = fo.write(csv_spec)
    >>> data = load_data_csv(tmpfile, {"data_folder": "/home"})
    >>> data["utt1"]["wav_path"]
    '/home/utt1.wav'
    """

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        variable_finder = re.compile(r"\$([\w.]+)")
        for row in reader:
            # ID:
            try:
                data_id = row["ID"]
                del row["ID"]  # This is used as a key in result, instead.
            except KeyError:
                raise KeyError(
                    "CSV has to have an 'ID' field, with unique ids"
                    " for all data points"
                )
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            for key, value in row.items():
                try:
                    row[key] = variable_finder.sub(
                        lambda match: str(replacements[match[1]]), value
                    )
                except KeyError:
                    raise KeyError(
                        f"The item {value} requires replacements "
                        "which were not supplied."
                    )
            # Duration:
            if "duration" in row:
                row["duration"] = float(row["duration"])
            result[data_id] = row
    return result


def read_audio_info(
    path, backend=None
) -> "torchaudio.backend.common.AudioMetaData":
    """Retrieves audio metadata from a file path. Behaves identically to
    torchaudio.info, but attempts to fix metadata (such as frame count) that is
    otherwise broken with certain torchaudio version and codec combinations.

    Note that this may cause full file traversal in certain cases!

    Arguments
    ---------
    path : str
        Path to the audio file to examine.
    backend : str, optional
        Audio backend to use for loading the audio file. Must be one of
        'ffmpeg', 'sox', 'soundfile' or None. If None, uses torchaudio's default backend.

    Raises
    ------
    ValueError
        If the `backend` is not one of the allowed values.
        Must be one of [None, 'ffmpeg', 'sox', 'soundfile'].

    Returns
    -------
    torchaudio.backend.common.AudioMetaData
        Same value as returned by `torchaudio.info`, but may eventually have
        `num_frames` corrected if it otherwise would have been `== 0`.

    NOTE
    ----
    Some codecs, such as MP3, require full file traversal for accurate length
    information to be retrieved.
    In these cases, you may as well read the entire audio file to avoid doubling
    the processing time.
    """
    validate_backend(backend)

    _path_no_ext, path_ext = os.path.splitext(path)

    if path_ext == ".mp3":
        # Additionally, certain affected versions of torchaudio fail to
        # autodetect mp3.
        # HACK: here, we check for the file extension to force mp3 detection,
        # which prevents an error from occurring in torchaudio.
        info = torchaudio.info(path, format="mp3", backend=backend)
    else:
        info = torchaudio.info(path, backend=backend)

    # Certain file formats, such as MP3, do not provide a reliable way to
    # query file duration from metadata (when there is any).
    # For MP3, certain versions of torchaudio began returning num_frames == 0.
    #
    # https://github.com/speechbrain/speechbrain/issues/1925
    # https://github.com/pytorch/audio/issues/2524
    #
    # Accommodate for these cases here: if `num_frames == 0` then maybe something
    # has gone wrong.
    # If some file really had `num_frames == 0` then we are not doing harm
    # double-checking anyway. If I am wrong and you are reading this comment
    # because of it: sorry
    if info.num_frames == 0:
        channels_data, sample_rate = torchaudio.load(
            path, normalize=False, backend=backend
        )

        info.num_frames = channels_data.size(1)
        info.sample_rate = sample_rate  # because we might as well

    return info


def read_audio(waveforms_obj, backend=None):
    """General audio loading, based on a custom notation.

    Expected use case is in conjunction with Datasets
    specified by JSON.

    The parameter may just be a path to a file:
    `read_audio("/path/to/wav1.wav")`

    Alternatively, you can specify more options in a dict, e.g.:
    ```
    # load a file from sample 8000 through 15999
    read_audio({
        "file": "/path/to/wav2.wav",
        "start": 8000,
        "stop": 16000
    })
    ```

    Which codecs are supported depends on your torchaudio backend.
    Refer to `torchaudio.load` documentation for further details.

    Arguments
    ---------
    waveforms_obj : str, dict
        Path to audio or dict with the desired configuration.

        Keys for the dict variant:
        - `"file"` (str): Path to the audio file.
        - `"start"` (int, optional): The first sample to load.
        If unspecified, load from the very first frame.
        - `"stop"` (int, optional): The last sample to load (exclusive).
        If unspecified or equal to start, load from `start` to the end.
        Will not fail if `stop` is past the sample count of the file and will
        return less frames.
    backend : str, optional
        Audio backend to use for loading the audio file. Must be one of
        'ffmpeg', 'sox', 'soundfile' or None. If None, uses torchaudio's default backend.

    Returns
    -------
    torch.Tensor
        1-channel: audio tensor with shape: `(samples, )`.
        >=2-channels: audio tensor with shape: `(samples, channels)`.

    Raises
    ------
    ValueError
        If the `backend` is not one of the allowed values.
        Must be one of [None, 'ffmpeg', 'sox', 'soundfile'].

    Example
    -------
    >>> dummywav = torch.rand(16000)
    >>> import os
    >>> tmpfile = str(getfixture('tmpdir') / "wave.wav")
    >>> write_audio(tmpfile, dummywav, 16000)
    >>> asr_example = { "wav": tmpfile, "spk_id": "foo", "words": "foo bar"}
    >>> loaded = read_audio(asr_example["wav"])
    >>> loaded.allclose(dummywav.squeeze(0),atol=1e-4) # replace with eq with sox_io backend
    True
    """
    validate_backend(backend)

    # Case 1: Directly a file path (str) or file-like object or raw bytes.
    # If a file-like object, ensure the pointer is at the beginning.
    if hasattr(waveforms_obj, "seek"):
        waveforms_obj.seek(0)

    if isinstance(waveforms_obj, (str, BytesIO, bytes)):
        # If raw bytes, wrap them in a BytesIO.
        if isinstance(waveforms_obj, bytes):
            waveforms_obj = BytesIO(waveforms_obj)
            waveforms_obj.seek(0)
        audio, _ = torchaudio.load(waveforms_obj, backend=backend)
    # Case 2: A dict with more options. Only works with file paths.
    else:
        path = waveforms_obj["file"]
        start = waveforms_obj.get("start", 0)
        # To match past SB behavior, `start == stop` or omitted `stop` means to
        # load all frames from `start` to the file end.
        stop = waveforms_obj.get("stop", start)

        if start < 0:
            raise ValueError(
                f"Invalid sample range (start < 0): {start}..{stop}!"
            )

        if stop < start:
            # Could occur if the user tried one of two things:
            # - specify a negative value as an attempt to index from the end;
            # - specify -1 as an attempt to load up to the last sample.
            raise ValueError(
                f"Invalid sample range (stop < start): {start}..{stop}!\n"
                'Hint: Omit "stop" if you want to read to the end of file.'
            )

        # Requested to load until a specific frame?
        if start != stop:
            num_frames = stop - start
            audio, fs = torchaudio.load(
                path, num_frames=num_frames, frame_offset=start, backend=backend
            )
        else:
            # Load to the end.
            audio, fs = torchaudio.load(
                path, frame_offset=start, backend=backend
            )

    audio = audio.transpose(0, 1)
    return audio.squeeze(1)


def read_audio_multichannel(waveforms_obj, backend=None):
    """General audio loading, based on a custom notation.

    Expected use case is in conjunction with Datasets
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

    Or you can specify a single file more succinctly:
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
    ---------
    waveforms_obj : str, dict
        Audio reading annotation, see above for format.
    backend : str, optional
        Audio backend to use for loading the audio file. Must be one of
        'ffmpeg', 'sox', 'soundfile' or None. If None, uses torchaudio's default backend.

    Raises
    ------
    ValueError
        If the `backend` is not one of the allowed values.
        Must be one of [None, 'ffmpeg', 'sox', 'soundfile'].

    Returns
    -------
    torch.Tensor
        Audio tensor with shape: (samples, ).

    Example
    -------
    >>> dummywav = torch.rand(16000, 2)
    >>> import os
    >>> tmpfile = str(getfixture('tmpdir') / "wave.wav")
    >>> write_audio(tmpfile, dummywav, 16000)
    >>> asr_example = { "wav": tmpfile, "spk_id": "foo", "words": "foo bar"}
    >>> loaded = read_audio(asr_example["wav"])
    >>> loaded.allclose(dummywav.squeeze(0),atol=1e-4) # replace with eq with sox_io backend
    True
    """
    validate_backend(backend)

    # Case 1: Directly a file path (str) or file-like object or raw bytes.
    # If a file-like object, ensure the pointer is at the beginning.
    if hasattr(waveforms_obj, "seek"):
        waveforms_obj.seek(0)

    if isinstance(waveforms_obj, (str, BytesIO, bytes)):
        # If raw bytes, wrap them in a BytesIO.
        if isinstance(waveforms_obj, bytes):
            waveforms_obj = BytesIO(waveforms_obj)
            waveforms_obj.seek(0)
        audio, _ = torchaudio.load(waveforms_obj, backend=backend)
        return audio.transpose(0, 1)

    # Case 2: A dict with more options. Only works with file paths.
    files = waveforms_obj["files"]
    if not isinstance(files, list):
        files = [files]

    waveforms = []
    start = waveforms_obj.get("start", 0)
    # Default stop to start -> if not specified, num_frames becomes 0,
    # which is the torchaudio default
    stop = waveforms_obj.get("stop", start - 1)
    num_frames = stop - start
    for f in files:
        audio, fs = torchaudio.load(
            f, num_frames=num_frames, frame_offset=start, backend=backend
        )
        waveforms.append(audio)

    out = torch.cat(waveforms, 0)
    return out.transpose(0, 1)


def write_audio(filepath, audio, samplerate):
    """Write audio on disk. It is basically a wrapper to support saving
    audio signals in the speechbrain format (audio, channels).

    Arguments
    ---------
    filepath: path
        Path where to save the audio file.
    audio : torch.Tensor
        Audio file in the expected speechbrain format (signal, channels).
    samplerate: int
        Sample rate (e.g., 16000).


    Example
    -------
    >>> import os
    >>> tmpfile = str(getfixture('tmpdir') / "wave.wav")
    >>> dummywav = torch.rand(16000, 2)
    >>> write_audio(tmpfile, dummywav, 16000)
    >>> loaded = read_audio(tmpfile)
    >>> loaded.allclose(dummywav,atol=1e-4) # replace with eq with sox_io backend
    True
    """
    if len(audio.shape) == 2:
        audio = audio.transpose(0, 1)
    elif len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    torchaudio.save(filepath, audio, samplerate)


def load_pickle(pickle_path):
    """Utility function for loading .pkl pickle files.

    Arguments
    ---------
    pickle_path : str
        Path to pickle file.

    Returns
    -------
    out : object
        Python object loaded from pickle.
    """
    with open(pickle_path, "rb") as f:
        out = pickle.load(f)
    return out


def to_floatTensor(x: Union[list, tuple, np.ndarray]):
    """
    Arguments
    ---------
    x : (list, tuple, np.ndarray)
        Input data to be converted to torch float.

    Returns
    -------
    tensor : torch.Tensor
        Data now in torch.tensor float datatype.
    """
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    else:
        return torch.tensor(x, dtype=torch.float)


def to_doubleTensor(x: Union[list, tuple, np.ndarray]):
    """
    Arguments
    ---------
    x : (list, tuple, np.ndarray)
        Input data to be converted to torch double.

    Returns
    -------
    tensor : torch.Tensor
        Data now in torch.tensor double datatype.
    """
    if isinstance(x, torch.Tensor):
        return x.double()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).double()
    else:
        return torch.tensor(x, dtype=torch.double)


def to_longTensor(x: Union[list, tuple, np.ndarray]):
    """
    Arguments
    ---------
    x : (list, tuple, np.ndarray)
        Input data to be converted to torch long.

    Returns
    -------
    tensor : torch.Tensor
        Data now in torch.tensor long datatype.
    """
    if isinstance(x, torch.Tensor):
        return x.long()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).long()
    else:
        return torch.tensor(x, dtype=torch.long)


def convert_index_to_lab(batch, ind2lab):
    """Convert a batch of integer IDs to string labels.

    Arguments
    ---------
    batch : list
        List of lists, a batch of sequences.
    ind2lab : dict
        Mapping from integer IDs to labels.

    Returns
    -------
    list
        List of lists, same size as batch, with labels from ind2lab.

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
    """Converts SpeechBrain style relative length to the absolute duration.

    Operates on batch level.

    Arguments
    ---------
    batch : torch.Tensor
        Sequences to determine the duration for.
    relative_lens : torch.Tensor
        The relative length of each sequence in batch. The longest sequence in
        the batch needs to have relative length 1.0.
    rate : float
        The rate at which sequence elements occur in real-world time. Sample
        rate, if batch is raw wavs (recommended) or 1/frame_shift if batch is
        features. This has to have 1/s as the unit.

    Returns
    -------
    torch.Tensor
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
        SpeechBrain format, producing three fields: key, key_format, key_opts.
    defaults : dict
        Mapping from CSV key to corresponding default value.

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
        """Sets a default value for the given CSV field.

        Arguments
        ---------
        field : str
            A field in the CSV.
        value : str
            The default value.
        """
        if field not in self.fields:
            raise ValueError(f"{field} is not a field in this CSV!")
        self.defaults[field] = value

    def write(self, *args, **kwargs):
        """Writes one data line into the CSV.

        Arguments
        ---------
        *args : tuple
            Supply every field with a value in positional form OR.
        **kwargs : dict
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
        """Writes a batch of lines into the CSV.

        Here each argument should be a list with the same length.

        Arguments
        ---------
        *args : tuple
            Supply every field with a value in positional form OR.
        **kwargs : dict
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


def write_txt_file(data, filename, sampling_rate=None):
    """Write data in text format.

    Arguments
    ---------
    data : str, list, torch.Tensor, numpy.ndarray
        The data to write in the text file.
    filename : str
        Path to file where to write the data.
    sampling_rate : None
        Not used, just here for interface compatibility.

    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> signal=torch.tensor([1,2,3,4])
    >>> write_txt_file(signal, tmpdir / 'example.txt')
    """
    del sampling_rate  # Not used.
    # Check if the path of filename exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as fout:
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
    """Write data to standard output.

    Arguments
    ---------
    data : str, list, torch.Tensor, numpy.ndarray
        The data to write in the text file.
    filename : None
        Not used, just here for compatibility.
    sampling_rate : None
        Not used, just here for compatibility.

    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> signal = torch.tensor([[1,2,3,4]])
    >>> write_stdout(signal, tmpdir / 'example.txt')
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


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.

    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
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


def read_kaldi_lab(kaldi_ali, kaldi_lab_opts):
    """Read labels in kaldi format.

    Uses kaldi IO.

    Arguments
    ---------
    kaldi_ali : str
        Path to directory where kaldi alignments are stored.
    kaldi_lab_opts : str
        A string that contains the options for reading the kaldi alignments.

    Returns
    -------
    lab : dict
        A dictionary containing the labels.

    Note
    ----
    This depends on kaldi-io-for-python. Install it separately.
    See: https://github.com/vesis84/kaldi-io-for-python

    Example
    -------
    This example requires kaldi files.
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
            + "/final.mdl ark:- ark:-|"
        )
    }
    return lab


def get_md5(file):
    """Get the md5 checksum of an input file.

    Arguments
    ---------
    file : str
        Path to file for which compute the checksum.

    Returns
    -------
    md5
        Checksum for the given filepath.

    Example
    -------
    >>> get_md5('tests/samples/single-mic/example1.wav')
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
    """Saves the md5 of a list of input files as a pickled dict into a file.

    Arguments
    ---------
    files : list
        List of input files from which we will compute the md5.
    out_file : str
        The path where to store the output pkl file.

    Example
    -------
    >>> files = ['tests/samples/single-mic/example1.wav']
    >>> tmpdir = getfixture('tmpdir')
    >>> save_md5(files, tmpdir / "md5.pkl")
    """
    # Initialization of the dictionary
    md5_dict = {}
    # Computing md5 for all the files in the list
    for file in files:
        md5_dict[file] = get_md5(file)
    # Saving dictionary in pkl format
    save_pkl(md5_dict, out_file)


def save_pkl(obj, file):
    """Save an object in pkl format.

    Arguments
    ---------
    obj : object
        Object to save in pkl format
    file : str
        Path to the output file

    Example
    -------
    >>> tmpfile = getfixture('tmpdir') / "example.pkl"
    >>> save_pkl([1, 2, 3, 4, 5], tmpfile)
    >>> load_pkl(tmpfile)
    [1, 2, 3, 4, 5]
    """
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    """Loads a pkl file.

    For an example, see `save_pkl`.

    Arguments
    ---------
    file : str
        Path to the input pkl file.

    Returns
    -------
    The loaded object.
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
        open(file + ".lock", "w", encoding="utf-8").close()
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
        Containing the original labels. Must be of size: [batch_size, max_length].
    bos_index : int
        The index for <bos> token.

    Returns
    -------
    new_label : tensor
        The new label with <bos> at the beginning.

    Example
    -------
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
        Containing the original length of each label sequences. Must be 1D.
    eos_index : int
        The index for <eos> token.

    Returns
    -------
    new_label : tensor
        The new label with <eos> appended.

    Example
    -------
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
        Each item contains a list, and this list contains a character sequence.
    space : string
        The token represents space. Default: _

    Returns
    -------
    The list contains word sequences for each sentence.

    Example
    -------
    >>> sequences = [["a", "b", "_", "c", "_", "d", "e"], ["e", "f", "g", "_", "h", "i"]]
    >>> results = merge_char(sequences)
    >>> results
    [['ab', 'c', 'de'], ['efg', 'hi']]
    """
    results = []
    for seq in sequences:
        words = "".join(seq).split(space)
        results.append(words)
    return results


def merge_csvs(data_folder, csv_lst, merged_csv):
    """Merging several csv files into one file.

    Arguments
    ---------
    data_folder : string
        The folder to store csv files to be merged and after merging.
    csv_lst : list
        Filenames of csv file to be merged.
    merged_csv : string
        The filename to write the merged csv file.

    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> os.symlink(os.path.realpath("tests/samples/annotation/speech.csv"), tmpdir / "speech.csv")
    >>> merge_csvs(tmpdir,
    ... ["speech.csv", "speech.csv"],
    ... "test_csv_merge.csv")
    """
    write_path = os.path.join(data_folder, merged_csv)
    if os.path.isfile(write_path):
        logger.info("Skipping merging. Completed in previous run.")
    with open(
        os.path.join(data_folder, csv_lst[0]), newline="", encoding="utf-8"
    ) as f:
        header = f.readline()
    lines = []
    for csv_file in csv_lst:
        with open(
            os.path.join(data_folder, csv_file), newline="", encoding="utf-8"
        ) as f:
            for i, line in enumerate(f):
                if i == 0:
                    # Checking header
                    if line != header:
                        raise ValueError(
                            "Different header for " f"{csv_lst[0]} and {csv}."
                        )
                    continue
                lines.append(line)
    with open(write_path, "w", encoding="utf-8") as f:
        f.write(header)
        for line in lines:
            f.write(line)
    logger.info(f"{write_path} is created.")


def split_word(sequences, space="_"):
    """Split word sequences into character sequences.

    Arguments
    ---------
    sequences: list
        Each item contains a list, and this list contains a words sequence.
    space: string
        The token represents space. Default: _

    Returns
    -------
    The list contains word sequences for each sentence.

    Example
    -------
    >>> sequences = [['ab', 'c', 'de'], ['efg', 'hi']]
    >>> results = split_word(sequences)
    >>> results
    [['a', 'b', '_', 'c', '_', 'd', 'e'], ['e', 'f', 'g', '_', 'h', 'i']]
    """
    results = []
    for seq in sequences:
        chars = list(space.join(seq))
        results.append(chars)
    return results


def clean_padding_(tensor, length, len_dim=1, mask_value=0.0):
    """Sets the value of any padding on the specified tensor to mask_value.

    For instance, this can be used to zero out the outputs of an autoencoder
    during training past the specified length.

    This is an in-place operation

    Arguments
    ---------
    tensor: torch.Tensor
        a tensor of arbitrary dimension
    length: torch.Tensor
        a 1-D tensor of lengths
    len_dim: int
        the dimension representing the length
    mask_value: mixed
        the value to be assigned to padding positions

    Example
    -------
    >>> import torch
    >>> x = torch.arange(5).unsqueeze(0).repeat(3, 1)
    >>> x = x + torch.arange(3).unsqueeze(-1)
    >>> x
    tensor([[0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6]])
    >>> length = torch.tensor([0.4, 1.0, 0.6])
    >>> clean_padding_(x, length=length, mask_value=10.)
    >>> x
    tensor([[ 0,  1, 10, 10, 10],
            [ 1,  2,  3,  4,  5],
            [ 2,  3,  4, 10, 10]])
    >>> x = torch.arange(5)[None, :, None].repeat(3, 1, 2)
    >>> x = x + torch.arange(3)[:, None, None]
    >>> x = x * torch.arange(1, 3)[None, None, :]
    >>> x = x.transpose(1, 2)
    >>> x
    tensor([[[ 0,  1,  2,  3,  4],
             [ 0,  2,  4,  6,  8]],
    <BLANKLINE>
            [[ 1,  2,  3,  4,  5],
             [ 2,  4,  6,  8, 10]],
    <BLANKLINE>
            [[ 2,  3,  4,  5,  6],
             [ 4,  6,  8, 10, 12]]])
    >>> clean_padding_(x, length=length, mask_value=10., len_dim=2)
    >>> x
    tensor([[[ 0,  1, 10, 10, 10],
             [ 0,  2, 10, 10, 10]],
    <BLANKLINE>
            [[ 1,  2,  3,  4,  5],
             [ 2,  4,  6,  8, 10]],
    <BLANKLINE>
            [[ 2,  3,  4, 10, 10],
             [ 4,  6,  8, 10, 10]]])
    """
    max_len = tensor.size(len_dim)
    mask = length_to_mask(length * max_len, max_len).bool()
    mask_unsq = mask[(...,) + (None,) * (tensor.dim() - 2)]
    mask_t = mask_unsq.transpose(1, len_dim).expand_as(tensor)
    tensor[~mask_t] = mask_value


def clean_padding(tensor, length, len_dim=1, mask_value=0.0):
    """Sets the value of any padding on the specified tensor to mask_value.

    For instance, this can be used to zero out the outputs of an autoencoder
    during training past the specified length.

    This version of the operation does not modify the original tensor

    Arguments
    ---------
    tensor: torch.Tensor
        a tensor of arbitrary dimension
    length: torch.Tensor
        a 1-D tensor of lengths
    len_dim: int
        the dimension representing the length
    mask_value: mixed
        the value to be assigned to padding positions

    Returns
    -------
    result: torch.Tensor
        Tensor with updated padding.

    Example
    -------
    >>> import torch
    >>> x = torch.arange(5).unsqueeze(0).repeat(3, 1)
    >>> x = x + torch.arange(3).unsqueeze(-1)
    >>> x
    tensor([[0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6]])
    >>> length = torch.tensor([0.4, 1.0, 0.6])
    >>> x_p = clean_padding(x, length=length, mask_value=10.)
    >>> x_p
    tensor([[ 0,  1, 10, 10, 10],
            [ 1,  2,  3,  4,  5],
            [ 2,  3,  4, 10, 10]])
    >>> x = torch.arange(5)[None, :, None].repeat(3, 1, 2)
    >>> x = x + torch.arange(3)[:, None, None]
    >>> x = x * torch.arange(1, 3)[None, None, :]
    >>> x = x.transpose(1, 2)
    >>> x
    tensor([[[ 0,  1,  2,  3,  4],
             [ 0,  2,  4,  6,  8]],
    <BLANKLINE>
            [[ 1,  2,  3,  4,  5],
             [ 2,  4,  6,  8, 10]],
    <BLANKLINE>
            [[ 2,  3,  4,  5,  6],
             [ 4,  6,  8, 10, 12]]])
    >>> x_p = clean_padding(x, length=length, mask_value=10., len_dim=2)
    >>> x_p
    tensor([[[ 0,  1, 10, 10, 10],
             [ 0,  2, 10, 10, 10]],
    <BLANKLINE>
            [[ 1,  2,  3,  4,  5],
             [ 2,  4,  6,  8, 10]],
    <BLANKLINE>
            [[ 2,  3,  4, 10, 10],
             [ 4,  6,  8, 10, 10]]])
    """

    result = tensor.clone()
    clean_padding_(result, length, len_dim, mask_value)
    return result


def extract_concepts_values(sequences, keep_values, tag_in, tag_out, space):
    """keep the semantic concepts and values for evaluation.

    Arguments
    ---------
    sequences: list
        Each item contains a list, and this list contains a character sequence.
    keep_values: bool
        If True, keep the values. If not don't.
    tag_in: char
        Indicates the start of the concept.
    tag_out: char
        Indicates the end of the concept.
    space: string
        The token represents space. Default: _

    Returns
    -------
    The list contains concept and value sequences for each sentence.

    Example
    -------
    >>> sequences = [['<response>','_','n','o','_','>','_','<localisation-ville>','_','L','e','_','M','a','n','s','_','>'], ['<response>','_','s','i','_','>'],['v','a','_','b','e','n','e']]
    >>> results = extract_concepts_values(sequences, True, '<', '>', '_')
    >>> results
    [['<response> no', '<localisation-ville> Le Mans'], ['<response> si'], ['']]
    """
    results = []
    for sequence in sequences:
        # ['<response>_no_>_<localisation-ville>_Le_Mans_>']
        sequence = "".join(sequence)
        # ['<response>','no','>','<localisation-ville>','Le','Mans,'>']
        sequence = sequence.split(space)
        processed_sequence = []
        value = (
            []
        )  # If previous sequence value never used because never had a tag_out
        kept = ""  # If previous sequence kept never used because never had a tag_out
        concept_open = False
        for word in sequence:
            if re.match(tag_in, word):
                # If not close tag but new tag open
                if concept_open and keep_values:
                    if len(value) != 0:
                        kept += " " + " ".join(value)
                    concept_open = False
                    processed_sequence.append(kept)
                kept = word  # 1st loop: '<response>'
                value = []  # Concept's value
                concept_open = True  # Trying to catch the concept's value
                # If we want the CER
                if not keep_values:
                    processed_sequence.append(kept)  # Add the kept concept
            # If we have a tag_out, had a concept, and want the values for CVER
            elif re.match(tag_out, word) and concept_open and keep_values:
                # If we have a value
                if len(value) != 0:
                    kept += " " + " ".join(
                        value
                    )  # 1st loop: '<response>' + ' ' + 'no'
                concept_open = False  # Wait for a new tag_in to pursue
                processed_sequence.append(kept)  # Add the kept concept + value
            elif concept_open:
                value.append(word)  # 1st loop: 'no'
        # If not close tag but end sequence
        if concept_open and keep_values:
            if len(value) != 0:
                kept += " " + " ".join(value)
            concept_open = False
            processed_sequence.append(kept)
        if len(processed_sequence) == 0:
            processed_sequence.append("")
        results.append(processed_sequence)
    return results
