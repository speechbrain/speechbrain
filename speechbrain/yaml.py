"""This library gathers utilities for extended yaml loading

Authors: Peter Plantinga 2020, Aku Rouhe 2020
"""

import re
import yaml
import copy
import pydoc
import functools
import ruamel.yaml
from io import StringIO
from types import SimpleNamespace
from speechbrain.utils.data_utils import recursive_update


# NOTE: Empty dict as default parameter is fine here since overrides are never
# modified
def load_extended_yaml(yaml_stream, overrides={}, overrides_must_match=True):
    r'''This function implements the SpeechBrain extended YAML syntax

    The purpose for this syntax is a compact, structured hyperparameter and
    function definition. This function implements a few extensions to the yaml
    syntax, listed below.

    References and copies
    ---------------------
    Allows internal references to any scalar node in the file. Any node with
    tag `!ref` will create an object reference to the yaml object at the
    `<key.subkey>` location within the yaml itself, following reference chains.

        output_folder: results/asr
        alignment_saver: !obj:asr.ali.hmm.save
            save_dir: !ref <output_folder>  # results/asr

    Strings values are handled specially: references are substituted but
    the rest of the string is left in place, allowing filepaths to be
    easily extended:

        output_folder: results/asr
        alignment_saver: !obj:asr.ali.hmm.save
            save_dir: !ref <output_folder>/ali # exp/asr/ali

    A more complex example for demonstration purposes:

        key1: {a: !obj:object {arg1: 1}}
        key2: !ref <key1.a>

    Here, "key2" will contain a reference to the "a" object, so changing
    a.arg1 will also change key2.arg1. If you need a
    deep copy of the object instead of a shallow reference, you
    can use a similar syntax with the tag `!copy`. For example:

        key1: {a: !obj:object {arg1: 1}}
        key2: !copy <key1.a>

    Object and function creation
    ----------------------------
    Part of our clean structured hyperparameter interface is being able to
    specify python objects and functions directly in the yaml. These tags
    include the full module path as part of the tag, so that the node can be
    used to pass the arguments to the function. A list is passed as
    positional arguments and a mapping is passed as keyword arguments.

    In the previous example,
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
    overrides_must_match : bool
        Whether an error will be thrown when an override does not match
        a corresponding key in the yaml_stream.

    Returns
    -------
    A dictionary reflecting the structure of `yaml_stream`.

    Example
    -------
    >>> yaml_string = """
    ... a: 3
    ... thing: !obj:collections.Counter
    ...     b: !ref <a>
    ... """
    >>> params = load_extended_yaml(yaml_string)
    >>> params.thing
    Counter({'b': 3})
    '''
    yaml_stream = resolve_references(
        yaml_stream, overrides, overrides_must_match
    )
    yaml.Loader.add_multi_constructor("!", object_constructor)
    return SimpleNamespace(**yaml.load(yaml_stream, Loader=yaml.Loader))


def resolve_references(yaml_stream, overrides={}, overrides_must_match=False):
    r'''Resolves inter-document references, a component of extended YAML.

    Arguments
    ---------
    yaml_stream : stream
        A file-like object or string with the contents of a yaml file
        written with the extended YAML syntax.
    overrides : mapping
        A set of keys for which to change the value listed in the stream.
    overrides_must_match : bool
        Whether an error will be thrown when an override does not match
        a corresponding key in the yaml_stream. This is the opposite
        default from `load_extended_yaml` because `resolve_references`
        doesn't need to be as strict by default.

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
    recursive_update(preview, overrides, must_match=overrides_must_match)
    _walk_tree_and_resolve(current_node=preview, tree=preview)

    # Dump back to string so we can load with bells and whistles
    yaml_stream = StringIO()
    ruamel_yaml.dump(preview, yaml_stream)
    yaml_stream.seek(0)

    return yaml_stream


def _walk_tree_and_resolve(current_node, tree):
    """A recursive function for resolving `!ref` and `!copy` tags.

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
        MSG = "Replace !PLACEHOLDER values in YAML."
        raise ValueError(MSG)
    elif hasattr(current_node, "tag") and current_node.tag.value in [
        "!ref",
        "!copy",
    ]:
        copy_mode = current_node.tag.value == "!copy"
        current_node = recursive_resolve(
            reference=current_node.value,
            reference_list=[],
            full_tree=tree,
            copy_mode=copy_mode,
        )
    elif isinstance(current_node, list):
        for i, item in enumerate(current_node):
            current_node[i] = _walk_tree_and_resolve(item, tree)
    elif isinstance(current_node, dict):
        for k, v in current_node.items():
            current_node[k] = _walk_tree_and_resolve(v, tree)

    return current_node


def object_constructor(loader, tag, node):
    """A constructor method for a '!obj:' or '!fn:' prefixed tag.

    The sub-tree is passed as arguments, and if '!obj' is the tag prefix,
    the callable is called before returning (to create the instance).

    Arguments
    ---------
    loader : yaml.loader
        The loader used to call this constructor (e.g. `yaml.SafeLoader`).
    tag : str
        The tag_prefix + callable name (e.g. '!obj:speechbrain.xxx').
    node : yaml.Node
        The sub-tree belonging to the tagged node.

    Returns
    -------
    The result of calling the callable.
    """

    try:
        tag_prefix, callable_string = tag.split(":")
    except ValueError:
        raise ValueError(
            "Expected tag with prefix `!obj:` or `!fn:` but got %s" % tag
        )

    make_the_call = tag_prefix == "obj"

    # Parse arguments from the node
    if isinstance(node, yaml.MappingNode):
        kwargs = loader.construct_mapping(node, deep=True)
        return construct(callable_string, kwargs=kwargs, call=make_the_call)
    elif isinstance(node, yaml.SequenceNode):
        args = loader.construct_sequence(node, deep=True)
        return construct(callable_string, args=args, call=make_the_call)

    return construct(callable_string, call=make_the_call)


def construct(callable_string, args=[], kwargs={}, call=True):
    """Use pydoc.locate to create the callable.

    Arguments
    ---------
    callable_string : str
        The fully-qualified name of a callable.
    args : list
        A list of parameters to pass to the callable.
    kwargs : dict
        A dict defining keyword parameters to pass to the callable.
    call : bool
        Whether to immediately call the callable (e.g. to construct an object)
        or to just return the callable (with arguments bound).

    Example
    -------
    >>> kwargs = {'in_features': 100, 'out_features': 100}
    >>> model = construct('torch.nn.Linear', kwargs=kwargs)
    >>> model.__class__.__name__
    'Linear'

    Raises
    ------
    ImportError: An invalid callable string was passed.
    TypeError: An invalid parameter was passed.
    """
    callable_ = pydoc.locate(callable_string)
    if callable_ is None:
        raise ImportError("There is no such callable as %s" % callable_string)

    if call:
        try:
            result = callable_(*args, **kwargs)
        except TypeError as e:
            err_msg = "Invalid argument to callable %s" % callable_string
            e.args = (err_msg, *e.args)
            raise

        return result

    # Bind the arguments to the callable
    else:
        callable_ = functools.partial(callable_, *args, **kwargs)
        return callable_


def deref(ref, full_tree, copy_mode=False):
    """Find the value referred to by a reference in dot-notation

    Arguments
    ---------
    ref : str
        The location of the requested value, e.g. 'constants.param'
    full_tree : dict
        The dictionary to use for finding values
    copy_mode : bool
        Whether to copy the node before dereferencing.

    Returns
    -------
    The value in the full_tree dictionary referenced by `ref`.

    Example
    -------
    >>> deref('<constants.a.b>', {'constants': {'a': {'b': 'c'}}})
    'c'
    """

    # Follow references in dot notation
    branch = full_tree
    for part in ref[1:-1].split("."):
        if part not in branch:
            raise ValueError('The reference "%s" is not valid' % ref)
        branch = branch[part]

    # For ruamel.yaml classes, the value is in the tag attribute
    try:
        branch = branch.value
    except AttributeError:
        pass

    if copy_mode:
        return copy.deepcopy(branch)
    else:
        return branch


def recursive_resolve(reference, reference_list, full_tree, copy_mode=False):
    """Resolve a reference to a value, following chained references

    Arguments
    ---------
    reference : str
        a string containing '<x.y>' in it where x.y refers
        to a scalar node in the file.
    reference_list : list
        list of prior references in the chain, in order
        to catch circular references.
    full_tree : dict
        the dictionary in which to find all references and their values.
    copy_mode : bool
        Whether to perform a deep copy of the referenced node, rather than
        a shallow reference to the same object.

    Returns
    -------
    The dereferenced value, with possible string interpolation.

    Example
    -------
    >>> tree = {'a': 3, 'b': '<a>', 'c': '<b>/<b>'}
    >>> recursive_resolve('<c>', [], tree)
    '3/3'
    """
    # Non-greedy operator won't work here, because the fullmatch will
    # still match if the first and last things happen to be references
    reference_finder = re.compile(r"<[^>]*>")

    # Base case, no <key> present
    if not reference_finder.search(str(reference)):
        return reference

    if len(reference_list) > 1 and reference in reference_list[1:]:
        raise ValueError("Circular reference detected: ", reference_list)

    # First check for a full match. These replacements preserve type
    if reference_finder.fullmatch(reference):
        value = deref(reference, full_tree, copy_mode)
        reference_list += [reference]
        return recursive_resolve(value, reference_list, full_tree, copy_mode)

    # Next, do replacements within the string (interpolation)
    matches = reference_finder.findall(reference)
    reference_list += [match[0] for match in matches]

    def replace_fn(x, tree=full_tree, copy_mode=copy_mode):
        return str(deref(x[0], full_tree=tree, copy_mode=copy_mode))

    sub = reference_finder.sub(replace_fn, reference)
    return recursive_resolve(sub, reference_list, full_tree, copy_mode)
