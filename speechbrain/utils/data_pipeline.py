"""
A pipeline for data transformations.

Example
-------
>>> #We expect this to be used via extended YAML,
>>> # for a dataset:
>>> from speechbrain.yaml import load_extended_yaml
>>> yamlstring = '''
... pipeline: !apply:speechbrain.utils.data_pipeline.DataPipeline.from_configuration
...     dynamic_items:
...         foo:
...             func: !name:operator.add
...             argkeys: ["a", "b"]
...         bar:
...             func: !name:operator.sub
...             argkeys: ["foo", "b"]
...     output_keys: ["foo", "bar"]
... '''
>>> hparams = load_extended_yaml(yamlstring)
>>> hparams["pipeline"]({"a":1, "b":2})
{'foo': 3, 'bar': 1}

Author:
    * Aku Rouhe
"""

import collections
from speechbrain.utils.depgraph import DependencyGraph

DynamicItemConf = collections.namedtuple("DynamicItemConf", ["func", "argkeys"])


class DataPipeline:
    """
    Organises data transformations into a pipeline.

    Example
    -------
    >>> pipeline = DataPipeline.from_configuration(
    ...     dynamic_items={
    ...         "foo": {"func": lambda x: x.lower(), "argkeys": ["text"]},
    ...         "bar": {"func": lambda x: x[::-1], "argkeys": ["foo"]},
    ...     },
    ...     output_keys=["bar"],
    ... )
    >>> pipeline({"text": "Test"})
    {'bar': 'tset'}

    """

    def __init__(self, output_keys=None):
        self.output_mapping = {}
        self.dg = DependencyGraph()
        self._exec_order = None
        self._dynamic_item_keys = []
        self.set_output_keys(output_keys)

    @classmethod
    def from_configuration(cls, dynamic_items=None, output_keys=None):
        """
        Arguments
        ---------
        dynamic_items : dict, optional
            Nested dict with the format (in YAML notation):
            <key>:
                func: <callable> # To be called
                argkeys: <list> # keys of args, either other dynamic_items or in data
            <key2>: ...
        output_keys : dict, list, optional
            List of keys (either dynamic_items or entries in data)
            to add in the final output.

            If a dict is given; it is used to map internal keys to output keys.
            From the output_keys dict key:value pairs the key appears outside,
            and value is the internal key.
        """
        pipeline = cls()
        if dynamic_items is None:
            dynamic_items = {}
        if output_keys is None:
            output_keys = []
        for key, conf in dynamic_items.items():
            if isinstance(conf, list):
                pipeline.add_dynamic_item(key, *conf)
            else:
                pipeline.add_dynamic_item(key, **conf)
        pipeline.set_output_keys(output_keys)
        return pipeline

    def add_dynamic_item(self, key, func, argkeys):
        """
        Arguments
        ---------
        key : str
            Unique key
        func : callable
            To be called
        argkeys : list, str
            List of keys. When func is called, each key is resolved to
            either an entry in the data or the output of another dynamic_item.
            The func is then called with these as positional arguments,
            in the same order as specified here.
            A single arg can be given directly.
        """
        if key in self._dynamic_item_keys:
            raise ValueError(f"Duplicate function key {key}")
        else:
            self._dynamic_item_keys.append(key)
        if not isinstance(argkeys, list):
            argkeys = [argkeys]
        conf = DynamicItemConf(func, argkeys)
        self.dg.add_node(key, data=conf)
        for depended in argkeys:
            self.dg.add_edge(key, depended)
        self._exec_order = None

    def set_output_keys(self, keys):
        """Use this to change the output keys

        Also re-evaluates execution order.
        So if you request different outputs, some parts of the
        data pipeline may be skipped.

        Arguments
        ---------
        keys : dict, list, None
            List of of keys (str) to produce in output.

            If a dict is given; it is used to map internal keys to output keys.
            From the output_keys dict key:value pairs the key appears outside,
            and value is the internal key.
        """
        self.output_mapping = self._output_keys_to_mapping(keys)
        self._exec_order = None

    @staticmethod
    def _output_keys_to_mapping(keys):
        # Ensure a mapping (accept a list for convenience, too)
        if keys is None:
            output_mapping = {}
        elif isinstance(keys, dict):
            output_mapping = keys
        else:
            output_mapping = {key: key for key in keys}
        return output_mapping

    def compute_outputs(self, data):
        """
        Arguments
        ---------
        data : dict
            Dictionary with data entries by key.

        Returns
        -------
        dict
            With the keys that were set.
        """
        if self._exec_order is None:
            self._prepare_run(data)
        intermediate = {}
        for compute_key, edges, conf in self._exec_order:
            if compute_key in data:
                continue
            # It is a dynamic_item, so conf is a DynamicItemConf, which we can unpack:
            try:
                func, argkeys = conf
            except TypeError:
                raise RuntimeError(
                    f"Could not find {compute_key} in data or dynamic items"
                )
            args = [
                data[argkey] if argkey in data else intermediate[argkey]
                for argkey in argkeys
            ]
            intermediate[compute_key] = func(*args)
        return {
            outkey: data[inkey] if inkey in data else intermediate[inkey]
            for outkey, inkey in self.output_mapping.items()
        }

    def compute_specific(self, keys, data):
        """Compute output of specific item, without changing output_keys"""
        # If a key in data is requested as an output key, it might not exist
        # in the dependency graph yet. It's safe to add it here implicitly,
        # since we know that the key is found in data.
        output_mapping = self._output_keys_to_mapping(keys)
        for output_key in output_mapping.values():
            if output_key in data and output_key not in self.dg:
                self.dg.add_node(output_key)
        intermediate = {}
        for compute_key, edges, conf in self.dg.get_evaluation_order(
            selected_keys=output_mapping.values()
        ):
            if compute_key in data:
                continue
            # It is a dynamic_item, so conf is a DynamicItemConf, which we can unpack:
            func, argkeys = conf
            args = [
                data[argkey] if argkey in data else intermediate[argkey]
                for argkey in argkeys
            ]
            intermediate[compute_key] = func(*args)
        return {
            outkey: data[inkey] if inkey in data else intermediate[inkey]
            for outkey, inkey in output_mapping.items()
        }

    def __call__(self, data):
        return self.compute_outputs(data)

    def _prepare_run(self, data):
        for key in self._dynamic_item_keys:
            if key in data:
                raise ValueError(f"Dynamic item key {key} appears in data")
        # If a key in data is requested as an output key, it might not exist
        # in the dependency graph yet. It's safe to add it here implicitly,
        # since we know that the key is found in data.
        for output_key in self.output_mapping.values():
            if output_key in data and output_key not in self.dg:
                self.dg.add_node(output_key)
        self._exec_order = list(
            self.dg.get_evaluation_order(
                selected_keys=self.output_mapping.values()
            )
        )
