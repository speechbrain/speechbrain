"""Chaining together callables, if some require relative lengths"""

import inspect


def lengths_arg_exists(func):
    """Check if func takes ``lengths`` keyword argument.

    Arguments
    ---------
    func : callable
        The function, method, or other callable to search for the lengths arg.

    Returns
    -------
    True if func takes ``lengths`` keyword argument.
    """
    spec = inspect.getfullargspec(func)
    return "lengths" in spec.args + spec.kwonlyargs


LENGTHS_ARG_NAMES = ("lengths", "wav_lens")


def lengths_arg_name(func):
    """Return the name of the relative-lengths argument of ``func``, if any.

    Both ``lengths`` (the general SpeechBrain convention) and ``wav_lens``
    (the name used by e.g. the HuggingFace integration lobes) are recognized.

    Arguments
    ---------
    func : callable
        The function, method, or other callable to search for a lengths arg.

    Returns
    -------
    str or None
        The name of the lengths argument, or None if ``func`` does not take
        one.
    """
    spec = inspect.getfullargspec(func)
    candidates = spec.args + spec.kwonlyargs
    for name in LENGTHS_ARG_NAMES:
        if name in candidates:
            return name
    return None


class LengthsCapableChain:
    """Chain together callables. Can handle relative lengths.

    This is a more light-weight version of
    speechbrain.nnet.containers.LengthsCapableSequential

    Arguments
    ---------
    *funcs : list, optional
        Any number of functions or other callables, given in order of
        execution.
    """

    def __init__(self, *funcs):
        self.funcs = []
        self.takes_lengths = []
        self.lengths_arg_names = []
        for func in funcs:
            self.append(func)

    def __call__(self, x, lengths=None):
        """Run the chain of callables on the given input

        Arguments
        ---------
        x : Any
            The main input
        lengths : Any
            The lengths argument which will be conditionally passed to
            any functions in the chain that take a 'lengths' (or
            'wav_lens') argument. In SpeechBrain the convention is to
            use relative lengths.

        Returns
        -------
        The input as processed by each function. If no functions were given,
        simply returns the input.

        Note
        ----
        By convention, if a callable in the chain returns multiple outputs
        (returns a tuple), only the first output is passed to the next
        callable in the chain.
        """
        if not self.funcs:
            return x
        for func, lengths_arg in zip(self.funcs, self.lengths_arg_names):
            if lengths_arg is not None:
                x = func(x, **{lengths_arg: lengths})
            else:
                x = func(x)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def append(self, func):
        """Add a function to the chain"""
        self.funcs.append(func)
        lengths_arg = lengths_arg_name(func)
        self.lengths_arg_names.append(lengths_arg)
        self.takes_lengths.append(lengths_arg is not None)

    def __str__(self):
        clsname = self.__class__.__name__
        if self.funcs:
            return f"{clsname}:\n" + "\n".join(str(f) for f in self.funcs)
        else:
            return f"Empty {clsname}"
