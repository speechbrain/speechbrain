"""This library gathers utilities for extended yaml loading

Authors: Peter Plantinga 2020, Aku Rouhe 2020
"""

import re
import ast
import yaml
import copy
import pydoc
import inspect
import functools
import ruamel.yaml
import operator as op
from io import StringIO
from types import SimpleNamespace
from speechbrain.utils.data_utils import recursive_update


# NOTE: Empty dict as default parameter is fine here since overrides are never
# modified
def load_extended_yaml(
    yaml_stream, overrides=None, overrides_must_match=True, return_dict=False
):
    r'''This function implements the SpeechBrain extended YAML syntax

    The purpose for this syntax is a compact, structured hyperparameter and
    function definition. This function implements a few extensions to the yaml
    syntax, listed below.

    Pyyaml complex tag shortcuts
    ----------------------------
    Part of our clean structured hyperparameter interface is being able to
    specify python objects easily and cleanly. This is possible with
    native YAML using the following syntax:

        alignment_saver: !!python/object/new:speechbrain.data_io.data_io.TensorSaver
            kwargs: {save_dir: results/asr/ali}

    However, due to the extensive use within speechbrain yaml files, we have
    added a shortcut for this that has the following syntax:

        alignment_saver: !new:speechbrain.data_io.data_io.TensorSaver
            save_dir: results/asr/ali

    In this example, the alignment_saver will be an instance of the
    `TensorSaver` class, with `'exp/asr/ali'` passed to the
    `__init__()` method as a keyword argument. This is equivalent to:

        import speechbrain.data_io.data_io
        alignment_saver = speechbrain.data_io.data_io.TensorSaver(
            save_dir='exp/asr/ali'
        )

    We have also implemented a few more shortcuts:

        !!python/name: => !name:
        !!python/module: => !module:

    References and copies
    ---------------------
    Allows internal references to any node in the file. Any node with
    tag `!ref` will create an object reference to the yaml object at the
    `<key.subkey>` location within the yaml itself, following reference chains.

        output_folder: results/asr
        alignment_saver: !new:speechbrain.data_io.data_io.TensorSaver
            save_dir: !ref <output_folder>

    Strings values are handled specially: references are substituted but
    the rest of the string is left in place, allowing filepaths to be
    easily extended:

        output_folder: results/asr
        alignment_saver: !new:speechbrain.data_io.data_io.TensorSaver
            save_dir: !ref <output_folder>/ali  # results/asr/ali

    A more complex example for demonstration purposes:

        key1: {a: !new:object {arg1: 1}}
        key2: !ref <key1.a>

    Here, "key2" will contain a reference to the "a" object, so changing
    a.arg1 will also change key2.arg1. If you need a
    deep copy of the object instead of a shallow reference, you
    can use a similar syntax with the tag `!copy`. For example:

        key1: {a: !new:object {arg1: 1}}
        key2: !copy <key1.a>

    These will also implement very basic arithmetic, so:

        key1: 1
        key2: !ref <key1> + 3  # this is 4

    Tuples
    ------
    One last minor enhancement is an implicit tuple resolver. Passing
    a string value of `(3, 4)` will be given a tag of `!tuple` which is
    then interpreted as a tuple.

    Arguments
    ---------
    yaml_stream : stream
        A file-like object or string from which to read.
    overrides : mapping or str
        A set of overrides for the values read from the stream.
        As yaml implements a nested structure, so can the overrides.
        See `speechbrain.utils.data_utils.recursive_update`
    overrides_must_match : bool
        Whether an error will be thrown when an override does not match
        a corresponding key in the yaml_stream.
    return_dict : bool
        Whether to return a dictionary rather than the default namespace.

    Returns
    -------
    A namespace reflecting the structure of `yaml_stream`. The namespace
    provides convenient "dot" access to all the first-level items in
    the yaml file.

    Example
    -------
    >>> yaml_string = """
    ... a: 3
    ... thing: !new:collections.Counter
    ...     b: !ref <a>
    ... """
    >>> params = load_extended_yaml(yaml_string)
    >>> params.thing
    Counter({'b': 3})
    '''
    yaml_stream = resolve_references(
        yaml_stream, overrides, overrides_must_match
    )

    # Parse flat tuples (no nesting of lists, dicts)
    yaml.Loader.add_constructor(tag="!tuple", constructor=_make_tuple)
    tuple_pattern = re.compile(r"^\(.*\)$")
    yaml.Loader.add_implicit_resolver("!tuple", tuple_pattern, first="(")

    # Parse shortcuts to `new`, `name`, and `module`
    yaml.Loader.add_multi_constructor("!new:", _construct_object)
    yaml.Loader.add_multi_constructor("!name:", _construct_name)
    yaml.Loader.add_multi_constructor("!module:", _construct_module)

    # If requested, return a dictionary as normal yaml (preserves order)
    if return_dict:
        return yaml.load(yaml_stream, Loader=yaml.Loader)

    # Return a namespace for clean dot-notation
    return SimpleNamespace(**yaml.load(yaml_stream, Loader=yaml.Loader))


def resolve_references(yaml_stream, overrides=None, overrides_must_match=False):
    r'''Resolves inter-document references, a component of extended YAML.

    Arguments
    ---------
    yaml_stream : stream
        A file-like object or string with the contents of a yaml file
        written with the extended YAML syntax.
    overrides : mapping or str
        Replacement values, either in a yaml-formatted string or a dict.
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

    if overrides is not None:
        if isinstance(overrides, str):
            overrides = ruamel_yaml.load(overrides)
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


def _make_tuple(loader, node):
    """Parse scalar node as a list, convert to tuple"""
    tuple_string = loader.construct_scalar(node)
    list_string = "[" + tuple_string[1:-1] + "]"
    parsed_list = yaml.load(list_string, Loader=yaml.Loader)
    return tuple(parsed_list)


def _load_node(loader, node):
    if isinstance(node, yaml.MappingNode):
        kwargs = loader.construct_mapping(node, deep=True)
        return [], kwargs
    elif isinstance(node, yaml.SequenceNode):
        args = loader.construct_sequence(node, deep=True)
        return args, {}
    return [], {}


def _construct_object(loader, callable_string, node):
    callable_ = pydoc.locate(callable_string)
    if callable_ is None:
        raise ImportError("There is no such class as %s" % callable_string)

    if not inspect.isclass(callable_):
        raise ValueError(
            f"!new:{callable_string} should be a class, but is {callable_}"
        )

    try:
        args, kwargs = _load_node(loader, node)
        return callable_(*args, **kwargs)
    except TypeError as e:
        err_msg = "Invalid argument to class %s" % callable_string
        e.args = (err_msg, *e.args)
        raise


def _construct_name(loader, callable_string, node):
    callable_ = pydoc.locate(callable_string)
    if callable_ is None:
        raise ImportError("There is no such callable as %s" % callable_string)

    if not (inspect.isclass(callable_) or inspect.isfunction(callable_)):
        raise ValueError(
            f"!name:{callable_string} should be class or function, "
            f"but is {callable_}"
        )

    try:
        args, kwargs = _load_node(loader, node)
        return functools.partial(callable_, *args, **kwargs)
    except TypeError as e:
        err_msg = "Invalid argument to callable %s" % callable_string
        e.args = (err_msg, *e.args)
        raise


def _construct_module(loader, module_name, node):
    module = pydoc.locate(module_name)
    if module is None:
        raise ImportError("There is no such module as %s" % module_name)

    args, kwargs = _load_node(loader, node)
    if args != [] or kwargs != {}:
        raise ValueError("Cannot pass args to module")
    if not inspect.ismodule(module):
        raise ValueError(
            f"!module:{module_name} should be module, but is {module}"
        )

    return module


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

    if copy_mode:
        return copy.deepcopy(branch)

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
    The dereferenced value, with possible string interpolation and
    arithmetic parsing.

    Example
    -------
    >>> tree = {'a': 3, 'b': 'x', 'c': '<a>', 'd': '<c>/<c>', 'e': '<b>/<b>'}
    >>> recursive_resolve('<d>', [], tree)
    1.0
    >>> recursive_resolve('<e>', [], tree)
    'x/x'
    """
    # Non-greedy operator won't work here, because the fullmatch will
    # still match if the first and last things happen to be references
    reference_finder = re.compile(r"<[^>]*>")

    # Base case, no <key> present
    if not isinstance(reference, str) or not reference_finder.search(reference):
        return reference

    if len(reference_list) > 1 and reference in reference_list[1:]:
        raise ValueError("Circular reference detected: ", reference_list)

    # First check for a full match. These replacements preserve type.
    if reference_finder.fullmatch(reference):
        value = deref(reference, full_tree, copy_mode)
        reference_list += [reference]
        return recursive_resolve(value, reference_list, full_tree, copy_mode)

    # Make sure reference list gets updated to prevent cycles
    matches = reference_finder.findall(reference)
    reference_list += [match[0] for match in matches]

    # Do replacements within the string (interpolation)
    def replace_fn(x, tree=full_tree, copy_mode=copy_mode):
        return str(deref(x[0], full_tree=tree, copy_mode=copy_mode))

    sub = reference_finder.sub(replace_fn, reference)
    reference = recursive_resolve(sub, reference_list, full_tree, copy_mode)

    # Finally check for arithmetic operations.
    return parse_arithmetic(reference)


def parse_arithmetic(reference_string):
    """Parses simple arithmetic operations in references

    Adapted from https://stackoverflow.com/a/9558001/1761970

    Arguments
    ---------
    reference_string : str
        A string with references and possible arithmetic operations.

    Returns
    -------
    result of parsing and applying the arithmetic

    Example
    -------
    >>> parse_arithmetic('2 * 6')
    12
    """
    try:
        return _ast_eval(ast.parse(reference_string, mode="eval").body)
    except (TypeError, SyntaxError, KeyError):
        return reference_string


def _ast_eval(node):
    ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Mod: op.mod,
    }
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return ops[type(node.op)](_ast_eval(node.left), _ast_eval(node.right))
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return ops[type(node.op)](_ast_eval(node.operand))
    else:
        raise TypeError(node)
