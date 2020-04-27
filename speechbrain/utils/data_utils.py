"""This library gathers utilities for data io operation.

Perhaps the most notable utility is the `load_extended_yaml` function
which implements the extended YAML syntax that SpeechBrain uses for
hyperparameters.

Authors: Mirco Ravanelli 2020, Peter Plantinga 2020, Aku Rouhe 2020
"""

import os
import re
import yaml
import torch
import ruamel.yaml
import collections.abc
from io import StringIO
from pydoc import locate


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


def recursive_update(d, u):
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
        else:
            d[k] = v


# NOTE: Empty dict as default parameter is fine here since overrides are never
# modified
def load_extended_yaml(yaml_stream, overrides={}):
    r'''This function implements the SpeechBrain extended YAML syntax

    The purpose for this syntax is a compact, structured hyperparameter and
    function definition. This function implements two extensions to the yaml
    syntax, references and and object instantiation.

    Reference substitution
    ----------------------
    Allows internal references to any scalar node in the file. Any
    node with tag `!ref` will have `<key>` references replaced with
    the referenced value, following reference chains.

        constants:
            output_folder: exp/asr
        alignment_saver: !asr.ali.hmm.save
            save_dir: !ref <constants.output_folder> # exp/asr

    Strings values are handled specially: references are substituted but
    the rest of the string is left in place, allowing filepaths to be
    easily extended:

        constants:
            output_folder: exp/asr
        alignment_saver: !asr.ali.hmm.save
            save_dir: !ref <constants.output_folder>/ali # exp/asr/ali

    Object instantiation
    --------------------
    If a '!'-prefixed tag is used, the node is interpreted as the
    parameters for instantiating the named class. In the previous example,
    the alignment_saver will be an instance of the `asr.ali.hmm.save` class,
    with `'exp/asr/ali'` passed to the `__init__()` method as a keyword
    argument. This is equivalent to:

        import asr.ali.hmm
        alignment_saver = asr.ali.hmm.save(save_dir='exp/asr/ali')

    Arguments
    ---------
    yaml_stream : stream
        A file-like object or string from which to read.
    overrides : mapping
        A set of overrides for the values read from the stream.
        As yaml implements a nested structure, so can the overrides.
        See `speechbrain.utils.data_utils.recursive_update`

    Returns
    -------
    A dictionary reflecting the structure of `yaml_stream`.

    Example
    -------
    >>> yaml_string = """
    ... constants:
    ...     a: 3
    ... thing: !collections.Counter
    ...     b: !ref <constants.a>
    ... """
    >>> load_extended_yaml(yaml_string)
    {'constants': {'a': 3}, 'thing': Counter({'b': 3})}
    '''
    yaml_stream = resolve_references(yaml_stream, overrides)
    yaml.SafeLoader.add_multi_constructor("!", object_constructor)
    return yaml.safe_load(yaml_stream)


def resolve_references(yaml_stream, overrides={}):
    r'''Resolves inter-document references, a component of extended YAML.

    Arguments
    ---------
    yaml_stream : stream
        A file-like object or string with the contents of a yaml file
        written with the extended YAML syntax.
    overrides : mapping
        A set of keys for which to change the value listed in the stream.

    Returns
    -------
    A text stream with all references and overrides resolved.

    Example
    -------
    >>> yaml_string = """
    ... constants:
    ...     a: 3
    ...     b: !ref <constants.a>
    ... """
    >>> overrides = {'constants': {'a': 4}}
    >>> resolve_references(yaml_string, overrides).getvalue()
    'constants:\n  a: 4\n  b: 4\n'
    '''
    # Load once to store references and apply overrides
    # using ruamel.yaml to preserve the tags
    ruamel_yaml = ruamel.yaml.YAML()
    preview = ruamel_yaml.load(yaml_stream)
    recursive_update(preview, overrides)
    _walk_tree_and_resolve(current_node=preview, tree=preview)

    # Dump back to string so we can load with bells and whistles
    yaml_stream = StringIO()
    ruamel_yaml.dump(preview, yaml_stream)
    yaml_stream.seek(0)

    return yaml_stream


def _walk_tree_and_resolve(current_node, tree):
    """A recursive function for resolving `!ref` tags.

    Arguments
    ---------
    current_node : node
        A node in the yaml tree loaded with ruamel.yaml.
    tree : node
        The base node in the yaml tree loaded with ruamel.yaml.

    Returns
    -------
    A yaml tree with all references resolved.
    """
    if (
        hasattr(current_node, "tag")
        and current_node.tag.value == "!PLACEHOLDER"
    ):
        MSG = f"Replace !PLACEHOLDER values in YAML."
        raise ValueError(MSG)
    elif hasattr(current_node, "tag") and current_node.tag.value == "!ref":
        current_node = recursive_resolve(current_node.value, [], tree)
    elif isinstance(current_node, list):
        for i, item in enumerate(current_node):
            current_node[i] = _walk_tree_and_resolve(item, tree)
    elif isinstance(current_node, dict):
        for k, v in current_node.items():
            current_node[k] = _walk_tree_and_resolve(v, tree)

    return current_node


def object_constructor(loader, callable_string, node):
    """A constructor method for a '!' tag with a class name.

    The class is instantiated, and the sub-tree is passed as arguments.

    Arguments
    ---------
    loader : yaml.loader
        The loader used to call this constructor (e.g. `yaml.SafeLoader`).
    callable_string : str
        The name of the callables (suffix after the '!' in this case).
    node : yaml.Node
        The sub-tree belonging to the tagged node.

    Returns
    -------
    The result of calling the callable.
    """

    # Parse arguments from the node
    if isinstance(node, yaml.MappingNode):
        kwargs = loader.construct_mapping(node, deep=True)
        return call(callable_string, kwargs=kwargs)
    elif isinstance(node, yaml.SequenceNode):
        args = loader.construct_sequence(node, deep=True)
        return call(callable_string, args=args)

    return call(callable_string)


def call(callable_string, args=[], kwargs={}):
    """Use pydoc.locate to create the callable, and then call it.

    Arguments
    ---------
    callable_string : str
        The fully-qualified name of a callable.
    args : list
        A list of parameters to pass to the callable.
    kwargs : dict
        A dict defining keyword parameters to pass to the callable.

    Example
    -------
    >>> kwargs = {'in_features': 100, 'out_features': 100}
    >>> model = call('torch.nn.Linear', kwargs=kwargs)
    >>> model.__class__.__name__
    'Linear'

    Raises
    ------
    ImportError: An invalid callable string was passed.
    TypeError: An invalid parameter was passed.
    """
    callable_ = locate(callable_string)
    if callable_ is None:
        raise ImportError("There is no such callable as %s" % callable_string)

    try:
        result = callable_(*args, **kwargs)
    except TypeError as e:
        err_msg = "Invalid argument to callable %s" % callable_string
        e.args = (err_msg, *e.args)
        raise

    return result


def deref(ref, preview):
    """Find the value referred to by a reference in dot-notation

    Arguments
    ---------
    ref : str
        The location of the requested value, e.g. 'constants.param'
    preview : dict
        The dictionary to use for finding values

    Returns
    -------
    The value in the preview dictionary referenced by `ref`.

    Example
    -------
    >>> deref('<constants.a.b>', {'constants': {'a': {'b': 'c'}}})
    'c'
    """

    # Follow references in dot notation
    for part in ref[1:-1].split("."):
        if part not in preview:
            raise ValueError('The reference "%s" is not valid' % ref)
        preview = preview[part]

    # For ruamel.yaml classes, the value is in the tag attribute
    try:
        preview = preview.value
    except AttributeError:
        pass

    return preview


def recursive_resolve(reference, reference_list, preview):
    """Resolve a reference to a value, following chained references

    Arguments
    ---------
    reference : str
        a string containing '<x.y>' in it where x.y refers
        to a scalar node in the file.
    reference_list : list
        list of prior references in the chain, in order
        to catch circular references.
    preview : dict
        the dictionary that stores all references and their values.

    Returns
    -------
    The dereferenced value, with possible string interpolation.

    Example
    -------
    >>> preview = {'a': 3, 'b': '<a>', 'c': '<b>/<b>'}
    >>> recursive_resolve('<c>', [], preview)
    '3/3'
    """
    # Non-greedy operator won't work here, because the fullmatch will
    # still match if the first and last things happen to be references
    reference_finder = re.compile(r"<[^>]*>")
    if len(reference_list) > 1 and reference in reference_list[1:]:
        raise ValueError("Circular reference detected: ", reference_list)

    # Base case, no '<ref>' present
    if not reference_finder.search(str(reference)):
        return reference

    # First check for a full match. These replacements preserve type
    if reference_finder.fullmatch(reference):
        value = deref(reference, preview)
        reference_list += [reference]
        return recursive_resolve(value, reference_list, preview)

    # Next, do replacements within the string (interpolation)
    matches = reference_finder.findall(reference)
    reference_list += [match[0] for match in matches]

    def replace_fn(x):
        return str(deref(x[0], preview))

    sub = reference_finder.sub(replace_fn, reference)
    return recursive_resolve(sub, reference_list, preview)


def compute_amplitude(waveforms, lengths):
    """Compute the average amplitude of a batch of waveforms.

    Arguments
    ---------
    waveform : tensor
        The waveforms used for computing amplitude.
    lengths : tensor
        The lengths of the waveforms excluding the padding
        added to put all waveforms in the same tensor.

    Returns
    -------
    The average amplitude of the waveforms.

    Example
    -------
    >>> import soundfile as sf
    >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
    >>> signal = torch.tensor(signal, dtype=torch.float32)
    >>> compute_amplitude(signal, len(signal))
    tensor([0.0125])
    """
    return (
        torch.sum(input=torch.abs(waveforms), dim=-1, keepdim=True,) / lengths
    )


def convolve1d(
    waveform,
    kernel,
    padding=0,
    pad_type="constant",
    stride=1,
    groups=1,
    use_fft=False,
    rotation_index=0,
):
    """Use torch.nn.functional to perform 1d padding and conv.

    Arguments
    ---------
    waveform : tensor
        The tensor to perform operations on.
    kernel : tensor
        The filter to apply during convolution
    padding : int or tuple
        The padding (pad_left, pad_right) to apply.
        If an integer is passed instead, this is passed
        to the conv1d function and pad_type is ignored.
    pad_type : str
        The type of padding to use. Passed directly to
        `torch.nn.functional.pad`, see PyTorch documentation
        for available options.
    stride : int
        The number of units to move each time convolution is applied.
        Passed to conv1d. Has no effect if `use_fft` is True.
    groups : int
        This option is passed to `conv1d` to split the input into groups for
        convolution. Input channels should be divisible by number of groups.
    use_fft : bool
        When `use_fft` is passed `True`, then compute the convolution in the
        spectral domain using complex multiply. This is more efficient on CPU
        when the size of the kernel is large (e.g. reverberation). WARNING:
        Without padding, circular convolution occurs. This makes little
        difference in the case of reverberation, but may make more difference
        with different kernels.
    rotation_index : int
        This option only applies if `use_fft` is true. If so, the kernel is
        rolled by this amount before convolution to shift the output location.

    Returns
    -------
    The convolved waveform.

    Example
    -------
    >>> import soundfile as sf
    >>> from speechbrain.data_io.data_io import save
    >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
    >>> signal = torch.tensor(signal[None, :, None])
    >>> filter = torch.rand(1, 10, 1, dtype=signal.dtype)
    >>> signal = convolve1d(signal, filter, padding=(9, 0))
    >>> save_signal = save(save_folder='exp/example', save_format='wav')
    >>> save_signal(signal, ['example_conv'], torch.ones(1))
    """
    if len(waveform.shape) != 3:
        raise ValueError("Convolve1D expects a 3-dimensional tensor")

    # Move time dimension last, which pad and fft and conv expect.
    waveform = waveform.transpose(2, 1)
    kernel = kernel.transpose(2, 1)

    # Padding can be a tuple (left_pad, right_pad) or an int
    if isinstance(padding, tuple):
        waveform = torch.nn.functional.pad(
            input=waveform, pad=padding, mode=pad_type,
        )

    # This approach uses FFT, which is more efficient if the kernel is large
    if use_fft:

        # Pad kernel to same length as signal, ensuring correct alignment
        zero_length = waveform.size(-1) - kernel.size(-1)

        # Handle case where signal is shorter
        if zero_length < 0:
            kernel = kernel[..., :zero_length]
            zero_length = 0

        # Perform rotation to ensure alignment
        zeros = torch.zeros(kernel.size(0), kernel.size(1), zero_length)
        after_index = kernel[..., rotation_index:]
        before_index = kernel[..., :rotation_index]
        kernel = torch.cat((after_index, zeros, before_index), dim=-1)

        # Compute FFT for both signals
        f_signal = torch.rfft(waveform, 1)
        f_kernel = torch.rfft(kernel, 1)

        # Complex multiply
        sig_real, sig_imag = f_signal.unbind(-1)
        ker_real, ker_imag = f_kernel.unbind(-1)
        f_result = torch.stack(
            [
                sig_real * ker_real - sig_imag * ker_imag,
                sig_real * ker_imag + sig_imag * ker_real,
            ],
            dim=-1,
        )

        # Inverse FFT
        convolved = torch.irfft(f_result, 1)

    # Use the implemenation given by torch, which should be efficient on GPU
    else:
        convolved = torch.nn.functional.conv1d(
            input=waveform,
            weight=kernel,
            stride=stride,
            groups=groups,
            padding=padding if not isinstance(padding, tuple) else 0,
        )

    # Return time dimension to the second dimension.
    return convolved.transpose(2, 1)


def dB_to_amplitude(SNR):
    """Returns the amplitude ratio, converted from decibels.

    Arguments
    ---------
    SNR : float
        The ratio in decibels to convert.

    Example
    -------
    >>> round(dB_to_amplitude(SNR=10), 3)
    3.162
    >>> dB_to_amplitude(SNR=0)
    1.0
    """
    return 10 ** (SNR / 20)


def notch_filter(notch_freq, filter_width=101, notch_width=0.05):
    """Returns a notch filter constructed from a high-pass and low-pass filter.

    (from https://tomroelandts.com/articles/
    how-to-create-simple-band-pass-and-band-reject-filters)

    Arguments
    ---------
    notch_freq : float
        frequency to put notch as a fraction of the
        sampling rate / 2. The range of possible inputs is 0 to 1.
    filter_width : int
        Filter width in samples. Longer filters have
        smaller transition bands, but are more inefficient.
    notch_width : float
        Width of the notch, as a fraction of the sampling_rate / 2.

    Example
    -------
    >>> import soundfile as sf
    >>> from speechbrain.data_io.data_io import save
    >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
    >>> signal = torch.tensor(signal, dtype=torch.float32)[None, :, None]
    >>> kernel = notch_filter(0.25)
    >>> notched_signal = convolve1d(signal, kernel)
    >>> save_signal = save(save_folder='exp/example', save_format='wav')
    >>> save_signal(notched_signal, ['freq_drop'], torch.ones(1))
    """

    # Check inputs
    assert 0 < notch_freq <= 1
    assert filter_width % 2 != 0
    pad = filter_width // 2
    inputs = torch.arange(filter_width) - pad

    # Avoid frequencies that are too low
    notch_freq += notch_width

    # Define sinc function, avoiding division by zero
    def sinc(x):
        def _sinc(x):
            return torch.sin(x) / x

        # The zero is at the middle index
        return torch.cat([_sinc(x[:pad]), torch.ones(1), _sinc(x[pad + 1 :])])

    # Compute a low-pass filter with cutoff frequency notch_freq.
    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    hlpf *= torch.blackman_window(filter_width)
    hlpf /= torch.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency notch_freq.
    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    hhpf *= torch.blackman_window(filter_width)
    hhpf /= -torch.sum(hhpf)
    hhpf[pad] += 1

    # Adding filters creates notch filter
    return (hlpf + hhpf).view(1, -1, 1)
