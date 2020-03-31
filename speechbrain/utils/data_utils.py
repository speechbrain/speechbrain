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
import torch
import inspect
import ruamel.yaml
import collections.abc
from io import StringIO
from pydoc import locate


def get_all_files(
    dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None
):
    """This function get a list of files within found within a folder.

    Different options can be used to restrict the search to some specific
    patterns.

    Args:
        dirName: the directory to search
        match_and: a list that contains patterns to match. The file is returned
            if it matches all the entries in match_and.
        match_or: a list that contains patterns to match. The file is returned
            if it matches one or more of the entries in match_or.
        exclude_and: a list that contains patterns to match. The file is
            returned if it matches none of the entries in exclude_and.
        exclude_or: a list that contains pattern to match. The
            file is returned if fails to match one of the entries in exclude_or

    Returns:
        a list of files.

    Example:
        >>> get_all_files('samples/rir_samples', match_and=['3.wav'])
        ['samples/rir_samples/rir3.wav']

    Author:
        Mirco Ravanelli 2020
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
    """Split the input list into N parts.

    Args:
        seq: the input list, to be split
        num: the number of chunks to produce

    Returns:
        a list containing all chunks created.

    Example:
        >>> split_list([1,2,3,4,5,6,7,8,9],4)
        [[1, 2], [3, 4], [5, 6], [7, 8, 9]]

    Author:
        Mirco Ravanelli 2020
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
    """Yield each (key, value) of a nested dictionary

    Args:
        dictionary: the dictionary (or dictionary of dictionaries) to list.

    Yields:
        key value tuples from the dictionary.

    Example:
        >>> rec_dict={'lev1': {'lev2': {'lev3': 'current_val'}}}
        >>> [item for item in recursive_items(rec_dict)]
        [('lev3', 'current_val')]

    Author:
        Mirco Ravanelli 2020
    """
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)


def recursive_update(d, u):
    """Similar function to dict.update, but for a nested dict

    If you have to a nested mapping structure, for example:
        {"a": 1, "b": {"c": 2}}
    Say you want to update the above structure with:
        {"b": {"d": 3}}
    This function will produce:
        {"a": 1, "b": {"c": 2, "d": 3}}
    Instead of:
        {"a": 1, "b": {"d": 3}}

    Args:
        d: mapping to be updated
        u: mapping to update with

    Example:
        >>> d = {'a': 1, 'b': {'c': 2}}
        >>> recursive_update(d, {'b': {'d': 3}})
        >>> d
        {'a': 1, 'b': {'c': 2, 'd': 3}}

    Author:
        Alex Martelli, with possibly other editors
        From: https://stackoverflow.com/a/3233356
    """
    # TODO: Consider cases where u has branch off k, but d does not.
    # e.g. d = {"a":1}, u = {"a": {"b": 2 }}
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            recursive_update(d.get(k, {}), v)
        else:
            d[k] = v


# NOTE: Empty dict as default parameter is fine here since overrides are never
# modified
def load_extended_yaml(
    yaml_string,
    overrides={},
):
    r"""This function implements the SpeechBrain extended YAML syntax

    The purpose for this syntax is a compact, structured hyperparameter and
    function definition. This function implements two extensions to the yaml
    syntax, references and and object instantiation.

    Reference substitution:
        Allows internal references to any other node in the file. Any
        node with tag '!$' will have $<key> references replaced with
        the referenced value, following reference chains.

            constants:
                output_folder: exp/asr
            alignment_saver: !asr.ali.hmm.save
                save_dir: !$ $constants.output_folder # exp/asr

        Strings values are handled specially: $-strings are substituted but
        the rest of the string is left in place, allowing filepaths to be
        easily extended:

            constants:
                output_folder: exp/asr
            alignment_saver: !asr.ali.hmm.save
                save_dir: !$ $constants.output_folder/ali # exp/asr/ali

    Object instantiation:
        If a '!'-prefixed tag is used, the node is interpreted as the
        parameters for instantiating the named class. In the previous example,
        the alignment_saver will be an instance of the asr.ali.hmm.save class,
        with 'exp/asr/ali' passed to the __init__() method as a keyword
        argument. This is equivalent to:

            import asr.ali.hmm
            alignment_saver = asr.ali.hmm.save(save_dir='exp/asr/ali')

    Args:
        yaml_string: A file-like object or string from which to read.
        overrides: mapping with which to override the values read the string.
            As yaml implements a nested structure, so can the overrides.
            See `speechbrain.utils.data_utils.recursive_update`

    Returns:
        A dictionary reflecting the structure of `yaml_string`.

    Example:
        >>> yaml_string = (""
        ... "constants:\n"
        ... "  a: 3\n"
        ... "thing: !collections.Counter\n"
        ... "  b: !$ <constants.a>")
        >>> load_extended_yaml(yaml_string)
        {'constants': {'a': 3}, 'thing': Counter({'b': 3})}

    Authors:
        Aku Rouhe and Peter Plantinga 2020
    """

    # Load once to store references and apply overrides
    # using ruamel.yaml to preserve the tags
    ruamel_yaml = ruamel.yaml.YAML()
    preview = ruamel_yaml.load(yaml_string)
    recursive_update(preview, overrides)

    # Dump back to string so we can load with bells and whistles
    yaml_string = StringIO()
    ruamel_yaml.dump(preview, yaml_string)
    yaml_string.seek(0)

    # NOTE: obj_and_ref_constructor needs to be defined in this scope to have
    # the correct version of preview
    def obj_and_ref_constructor(loader, tag_suffix, node):
        if tag_suffix == '$':
            nonlocal preview  # Not needed, but let's be explicit
            reference = loader.construct_scalar(node)
            return recursive_resolve(reference, [], preview)
        else:
            return object_constructor(loader, tag_suffix, node)

    # We also need a PyYAML Loader that is specific to this context
    # PyYAML syntax requires defining a new class to get a new loader
    class CustomLoader(yaml.SafeLoader):
        pass
    CustomLoader.add_multi_constructor('!', obj_and_ref_constructor)
    return yaml.load(yaml_string, Loader=CustomLoader)


def object_constructor(loader, class_name, node):
    """A constructor method for a '!' tag with a class name.

    The class is instantiated, and the sub-tree is passed as arguments.

    Args:
        loader: The loader used to call this constructor (e.g. CustomLoader)
        class_name: The name of the class (suffix after the '!' in this case)
        node: The sub-tree belonging to the tagged node

    Returns:
        The instantiated class

    Author:
        Peter Plantinga 2020
    """

    # Parse arguments from the node
    if isinstance(node, yaml.MappingNode):
        kwargs = loader.construct_mapping(node, deep=True)
        return instantiate(class_name, kwargs=kwargs)
    elif isinstance(node, yaml.SequenceNode):
        args = loader.construct_sequence(node, deep=True)
        return instantiate(class_name, args=args)

    return instantiate(class_name)


def instantiate(class_name, args=[], kwargs={}):
    """Use pydoc.locate to create an instance of the specified class+params.

    Args:
        class_name: The fully-qualified name of the class to instantiate.
        args: A list of parameters to pass the the class.
        kwargs: A dict defining keyword parameters to pass to the class.

    Example:
        >>> kwargs = {'in_features': 100, 'out_features': 100}
        >>> model = instantiate('torch.nn.Linear', kwargs=kwargs)
        >>> model.__class__.__name__
        'Linear'

    Author:
        Peter Plantinga 2020
    """
    class_ = locate(class_name)
    if class_ is None:
        raise ImportError('There is no such class as %s' % class_name)

    signature = inspect.signature(class_)

    try:
        signature.bind(*args, **kwargs)
    except TypeError as e:
        raise TypeError('Invalid argument to class %s' % class_name) from e

    return class_(*args, **kwargs)


def deref(ref, preview):
    """Find the value referred to by a reference in dot-notation

    Args:
        ref: The location of the requested value, e.g. 'constants.param'
        preview: The dictionary to use for finding values

    Returns:
        The value in the preview dictionary referenced by ref

    Example:
        >>> deref('<constants.a.b>', {'constants': {'a': {'b': 3}}})
        3

    Author:
        Peter Plantinga 2020
    """

    # Follow references in dot notation
    for part in ref[1:-1].split('.'):
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

    Args:
        reference: the current reference
        reference list: list of prior references in the chain, in order
            to catch circular references
        preview: the dictionary that stores all the values and references

    Returns:
        The dereferenced value, with possible string interpolation

    Example:
        >>> preview = {'a': 3, 'b': '<a>', 'c': '<b>/<b>'}
        >>> recursive_resolve('<c>', [], preview)
        '3/3'

    Author:
        Peter Plantinga 2020
    """
    # Non-greedy operator won't work here, because the fullmatch will
    # still match if the first and last things happen to be references
    reference_finder = re.compile(r'<[^>]*>')
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


def compute_amplitude(waveform, length):
    """Compute the average amplitude of a batch of waveforms.

    Args:
        waveform: The waveforms used for computing amplitude
        length: The length of the (un-padded) waveforms

    Returns:
        the average amplitude of the waveforms

    Example:
        >>> import torch
        >>> import soundfile as sf
        >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
        >>> signal = torch.tensor(signal, dtype=torch.float32)
        >>> compute_amplitude(signal, len(signal))
        tensor([0.0125])

    Author:
        Peter Plantinga 2020
    """
    return torch.sum(
        input=torch.abs(waveform),
        dim=-1,
        keepdim=True,
    ) / length


def convolve1d(
    waveform,
    kernel,
    padding=0,
    pad_type='constant',
    stride=1,
    groups=1,
    use_fft=False,
    rotation_index=0
):
    """Use torch.nn.functional to perform 1d padding and conv.

    Args:
        waveform: The tensor to perform operations on.
        kernel: The filter to apply during convolution
        padding: The padding (pad_left, pad_right) to apply.
            If an integer is passed instead, this is passed
            to the conv1d function and pad_type is ignored.
        pad_type: The type of padding to use. Passed directly to
            `torch.nn.functional.pad`, see PyTorch documentation
            for available options.
        stride: The number of units to move each time convolution
            is applied. Passed to conv1d. Has no effect if
            `use_fft` is True.
        groups: This option is passed to `conv1d` to split the input
            into groups for convolution. Input channels should
            be divisible by number of groups.
        use_fft: When `use_fft` is passed `True`, then compute the
            convolution in the spectral domain using complex
            multiply. This is more efficient on CPU when the
            size of the kernel is large (e.g. reverberation).
            WARNING: Without padding, circular convolution occurs.
            This makes little difference in the case of reverberation,
            but may make more difference with different kernels.
        rotation_index: This option only applies if `use_fft` is true. If so,
            the kernel is rolled by this amount before convolution
            to shift the output location.

    Returns:
        convolved waveform (type: torch.tensor)

    Example:
        >>> import torch
        >>> import soundfile as sf
        >>> from speechbrain.data_io.data_io import save
        >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
        >>> signal = torch.tensor(signal[None, None, :])
        >>> filter = torch.rand(1, 1, 10, dtype=signal.dtype)
        >>> signal = convolve1d(signal, filter, padding=(9, 0))
        >>> save_signal = save(save_folder='exp/example', save_format='wav')
        >>> save_signal(signal, ['example_conv'], torch.ones(1))

    Author:
        Peter Plantinga 2020
    """

    # Padding can be a tuple (left_pad, right_pad) or an int
    if isinstance(padding, tuple):
        waveform = torch.nn.functional.pad(
            input=waveform,
            pad=padding,
            mode=pad_type,
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
        f_result = torch.stack([
            sig_real*ker_real - sig_imag*ker_imag,
            sig_real*ker_imag + sig_imag*ker_real,
        ], dim=-1)

        # Inverse FFT
        return torch.irfft(f_result, 1)

    # Use the implemenation given by torch, which should be efficient on GPU
    return torch.nn.functional.conv1d(
        input=waveform,
        weight=kernel,
        stride=stride,
        groups=groups,
        padding=padding if not isinstance(padding, tuple) else 0,
    )


def dB_to_amplitude(SNR):
    """Convert decibels to amplitude

    Args:
        SNR: The ratio in decibels to convert

    Returns:
        ratio between average amplitudes

    Example:
        >>> round(dB_to_amplitude(SNR=10), 3)
        3.162

    Author:
        Peter Plantinga 2020
    """
    return 10 ** (SNR / 20)


def notch_filter(notch_freq, filter_width=101, notch_width=0.05):
    """Simple notch filter constructed from a high-pass and low-pass filter.

    Args:
        notch_freq: frequency to put notch as a fraction of the
            sampling rate / 2. The range of possible inputs is 0 to 1.
        filter_width: Filter width in samples. Longer filters have smaller
            transition bands, but are more inefficient
        notch_width: Width of the notch, as a fraction of the
            sampling_rate / 2.

    Returns:
        notch filter

    Example:
        >>> import torch
        >>> import soundfile as sf
        >>> from speechbrain.data_io.data_io import save
        >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
        >>> signal = torch.tensor(signal, dtype=torch.float32)[None, None, :]
        >>> kernel = notch_filter(0.25)
        >>> notched_signal = torch.nn.functional.conv1d(signal, kernel)
        >>> save_signal = save(save_folder='exp/example', save_format='wav')
        >>> save_signal(notched_signal, ['freq_drop'], torch.ones(1))

    Author:
        Tom Roelandts
        (from https://tomroelandts.com/articles/
        how-to-create-simple-band-pass-and-band-reject-filters)
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
        return torch.cat([_sinc(x[:pad]), torch.ones(1), _sinc(x[pad+1:])])

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
    return (hlpf + hhpf).view(1, 1, -1)
