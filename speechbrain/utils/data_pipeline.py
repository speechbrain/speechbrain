"""
A pipeline for data transformations.

Author:
    * Aku Rouhe
"""

import collections
from speechbrain.utils.depgraph import DependencyGraph

FuncConf = collections.namedtuple("FuncConf", ["func", "argnames"])


class DataPipeline:
    """

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

        final_names : list
            List of names to add in the final output.
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
