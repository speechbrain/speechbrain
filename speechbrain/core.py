import re
import os
import yaml
import inspect
import ruamel.yaml
from io import StringIO
from pydoc import locate
from types import SimpleNamespace
from speechbrain.utils.logger import setup_logger
from speechbrain.utils.data_utils import recursive_update


def load_extended_yaml(
    yaml_filename,
    overrides={},
    logger=None,
    start_experiment=False,
):
    """
    Description:
        This function implements the SpeechBrain extended YAML syntax
        by providing parser for it. The purpose for this syntax is a compact,
        structured hyperparameter and computation block definition. This
        function implements two extensions to the yaml syntax, $-reference
        and object instantiation.

        $-reference substitution:
            Allows internal references. These are restricted to refer to
            values in the variables on the top level of the variables section.
            A value of $<key> gets replaced by the value mapped to <key>
            in the variables section:

            ```
            constants:
                experiment_dir: exp/asr
            alignment_saver: !asr.ali.hmm.save
                save_dir: !$constants.experiment_dir # replaced with exp/asr
            ```

            Strings values are handled specially: $-strings are substituted but
            the rest of the string is left in place, allowing for example
            filepaths to be easily extended:

            ```
            constants:
                experiment_dir: exp/asr/
            alignment_saver: !asr.ali.hmm.save
                save_dir: !$constants.experiment_dir/ali # exp/asr/ali
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
        yaml_filename: a string representing a yaml filename, located
            in the same directory as the file calling this one.
        overrides: mapping with which to override the values read from f
            As yaml implements a nested structure, so can the overrides.
            See speechbrain.utils.data_utils.recursive_update
        logger: A logger object for recording errors

    Output:
        class_specs - the loaded class specifications
        variables - the loaded variables section

    Authors:
        Aku Rouhe and Peter Plantinga 2020
    """
    # Find path of the calling file, so we can load the yaml
    # file from the same directory
    calling_filename = inspect.getfile(inspect.currentframe().f_back)
    calling_dirname = os.path.dirname(os.path.abspath(calling_filename))
    yaml_filepath = os.path.join(calling_dirname, yaml_filename)

    # Load once to store references and apply overrides
    # using ruamel.yaml to preserve the tags
    ruamel_yaml = ruamel.yaml.YAML()
    preview = ruamel_yaml.load(open(yaml_filepath))
    preview = recursive_update(preview, overrides)

    # Dump back to string so we can load with bells and whistles
    yaml_string = StringIO()
    ruamel_yaml.dump(preview, yaml_string)
    if 'constants' in preview:
        const = preview['constants']
    else:
        raise ValueError("Required section: 'constants'")

    # TODO: Does converting to string really need to happen?
    # It may be inefficient. But then again, we may not care.
    yaml_string = yaml_string.getvalue()

    # Check for required variables in params file
    if start_experiment:
        for required in ['output_folder', 'verbosity']:
            if required not in const:
                raise ValueError('%s required in "constants" section of config'
                                 % required)

        # Set up output folder
        if not os.path.isdir(const['output_folder']):
            os.makedirs(const['output_folder'])

        # Setup logger
        if logger is None:
            log_file = const['output_folder'] + '/log.log'
            logger = setup_logger(
                "logger",
                log_file,
                verbosity_stdout=const['verbosity'],
            )

    # reference finder for string replacements
    reference_finder = re.compile(r'\$[\w.]+')

    # Now do the full parse, with dereferencing and object creation
    def deref(ref, constants):
        if isinstance(ref, re.Match):
            ref = ref[0]
        for part in ref[1:].split('.'):
            constants = constants[part]

        # For ruamel.yaml classes, the value is in the tag attribute
        try:
            constants = constants.tag.value[1:]
        except AttributeError:
            pass
        return constants

    def _recursive_resolve(reference, reference_list):
        if len(reference_list) > 1 and reference in reference_list[1:]:
            raise ValueError("Circular reference detected: ", reference_list)

        # Base case, no '$' present
        if '$' not in str(reference):
            return reference

        # First check for a full match. These replacements preserve type
        if reference_finder.fullmatch(reference):
            value = deref(reference, const)
            return _recursive_resolve(value, reference_list + [reference])

        # Next, do replacements within the string (interpolation)
        matches = reference_finder.findall(reference)
        reference_list += [match[0] for match in matches]
        sub = reference_finder.sub(lambda x: str(deref(x, const)), reference)
        return _recursive_resolve(sub, reference_list)

    def reference_constructor(loader, tag_suffix, node):

        # Check that this node is a scalar before resolving
        loader.construct_scalar(node)
        return _recursive_resolve(tag_suffix, [])

    def object_constructor(loader, tag_suffix, node):
        class_ = locate(tag_suffix)
        kwargs = {}
        if isinstance(node, yaml.MappingNode):
            kwargs = loader.construct_mapping(node, deep=True)
        elif isinstance(node, yaml.SequenceNode):
            args = loader.construct_sequence(node, deep=True)
            signature = inspect.signature(class_)
            kwargs = signature.bind(*args).arguments

        kwargs['global_config'] = const
        kwargs['logger'] = logger

        return class_(**kwargs)

    def obj_and_ref_constructor(loader, tag_suffix, node):
        if '$' in tag_suffix:
            return reference_constructor(loader, tag_suffix, node)
        else:
            return object_constructor(loader, tag_suffix, node)

    yaml.SafeLoader.add_multi_constructor('!', obj_and_ref_constructor)
    params = yaml.safe_load(yaml_string)
    constants = params['constants']
    del params['constants']

    return SimpleNamespace(**params), SimpleNamespace(**constants)
