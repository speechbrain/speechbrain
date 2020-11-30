"""
A pipeline for data transformations.

Example
-------
>>> #We expect this to be used via extended YAML,
>>> # for a dataset:
>>> from speechbrain.yaml import load_extended_yaml
>>> yamlstring = '''
... pipeline: !apply:speechbrain.utils.data_pipeline.DataPipeline.from_configuration
...     funcs:
...         foo:
...             func: !name:operator.add
...             argnames: ["a", "b"]
...         bar:
...             func: !name:operator.sub
...             argnames: ["foo", "b"]
...     final_names: ["foo", "bar"]
... '''
>>> hparams = load_extended_yaml(yamlstring)
>>> hparams["pipeline"]({"a":1, "b":2})
{'foo': 3, 'bar': 1}

Author:
    * Aku Rouhe
"""

import collections
from speechbrain.utils.depgraph import DependencyGraph

FuncConf = collections.namedtuple("FuncConf", ["func", "argnames"])


class DataPipeline:
    """
    Organises data transformations into a pipeline.

    Example
    -------
    >>> pipeline = DataPipeline.from_configuration(
    ...     funcs={
    ...         "foo": {"func": lambda x: x.lower(), "argnames": ["text"]},
    ...         "bar": {"func": lambda x: x[::-1], "argnames": ["foo"]},
    ...     },
    ...     final_names=["bar"],
    ... )
    >>> pipeline({"text": "Test"})
    {'bar': 'tset'}

    """

    def __init__(self):
        self.final_names = []  # Add names here to produce at output
        self.dg = DependencyGraph()
        self._exec_order = None
        self._func_names = []

    @classmethod
    def from_configuration(cls, funcs, final_names):
        """
        Arguments
        ---------
        funcs : dict
            Nested dict with the format (in YAML notation):
            <name>:
                func: <callable> # To be called
                argnames: <list> # Names of args, either other funcs or in data
            <name2>: ...
        final_names : list
            List of names (either funcs or entries in data)
            to add in the final output.
        """
        pipeline = cls()
        for name, conf in funcs.items():
            if isinstance(conf, list):
                pipeline.add_func(name, *conf)
            else:
                pipeline.add_func(name, **conf)
        pipeline.final_names = final_names
        return pipeline

    def add_func(self, name, func, argnames):
        """
        Arguments
        ---------
        name : str
            Unique name
        func : callable
            To be called
        argnames : list
            List of names. When func is called, each name is resolved to
            either an entry in the data or the output of another func.
            The func is then called with these as positional arguments,
            in the same order as specified here.
        """
        if name in self._func_names:
            raise ValueError(f"Duplicate function name {name}")
        else:
            self._func_names.append(name)
        conf = FuncConf(func, argnames)
        self.dg.add_node(name, data=conf)
        for depended in argnames:
            self.dg.add_edge(name, depended)
        self._exec_order = None

    def compute_outputs(self, data):
        """
        Arguments
        ---------
        data : dict
            Dictionary with named data entries.

        Returns
        -------
        dict
            With keys as in self.final_names
        """
        if self._exec_order is None:
            self._prepare_run(data)
        intermediate = {}
        for name, edges, conf in self._exec_order:
            if name in data:
                continue
            # It is a func, so conf is a FuncConf, which we can unpack:
            func, argnames = conf
            args = [
                data[name] if name in data else intermediate[name]
                for name in argnames
            ]
            intermediate[name] = func(*args)
        return {
            name: data[name] if name in data else intermediate[name]
            for name in self.final_names
        }

    def __call__(self, data):
        return self.compute_outputs(data)

    def _prepare_run(self, data):
        for name in self._func_names:
            if name in data:
                raise ValueError(f"Function name {name} appears in data")
        self._exec_order = list(
            self.dg.get_evaluation_order(selected_keys=self.final_names)
        )
