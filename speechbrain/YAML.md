Introduction
============

A crucial element of systems for processing speech, as with any data-analysis
systems, is laying out all the hyperparameters of that system so they can be
easily examined and modified. For our hyperparameter specification we adopt
YAML (YAML Ain't Markup Language), a popular human-readable data-serialization
language. We also add a few useful extensions to the YAML language in order to
support a rather expansive idea of what constitutes a hyperparameter.

This document will first go over a few YAML basics, then our extensions.

YAML basics
===========

YAML is a data-serialization language, similar to JSON, and it supports
three basic types of nodes: scalar, sequential, and mapping. PyYAML naturally
converts sequential nodes to python lists and mapping nodes to python dicts.

Scalar nodes can take one of the following forms:

```yaml
string: abcd  # No quotes needed
integer: 1
float: 1.3
bool: True
none: null
```

Note that we've used a simple mapping to demonstrate the scalar nodes. Mappings
can also be specified in a similar manner as JSON:

```yaml
foo: {bar: baz}
```

Sequences can also be specified in two ways:

```yaml
- foo
- bar
- [foo, bar, baz]
```

Note that when not using the inline version, YAML uses whitespace to denote
nested items:

```yaml
foo:
    bar: 1
    baz: 2
```

YAML has a few more advanced features (such as
[aliases](https://pyyaml.org/wiki/PyYAMLDocumentation#aliases) and
[merge keys](https://yaml.org/type/merge.html)) that you may want to explore,
on your own. We will briefly discuss one here since it is relevant later:
[YAML tags](https://pyyaml.org/wiki/PyYAMLDocumentation#tags).

Tags are added with a `!` prefix, and they specify the type of the node. This
allows types beyond the simple types listed above to be used. PyYAML supports a
few additional types, such as:

```yaml
!!set
!!timestamp
!!python/tuple
!!python/complex
!!python/name:module.name       # A class or function
!!python/module:package.module  # A module
!!python/object/new:module.cls  # An instance of a class
```

These can all be quite useful, however we found that this system was a bit
cumbersome, especially with the frequency with which we were using them. So
we decided to implement some shortcuts for these features, which we are
internally calling "extended YAML".

Extended YAML
=============

Our first extension is to simplify the structure for specifying an instance,
module, class, or function.

```yaml
model: !new:speechbrain.nnet.RNN.RNN
    layers: 5
    neurons_per_layer: 512
```

This tag, prefixed with `!new:`, constructs an instance of the specified class.
If the node is a mapping node, all the items are passed as keyword arguments
to the class when the instance is created. A list can similarly be used to
pass positional arguments.

We also simplify the interface for specifying a function or class:

```yaml
model: !name:speechbrain.nnet.RNN.RNN
    layers: 5
    neurons_per_layer: 512
```

Here, instead of passing the arguments to construct the object, the arguments
are bound and the result is a sort of factory that can be used to construct
objects.

Another main extension is a nicer alias interface, that supports things like
string interpolation. To do this, we've added a tag written `!ref` that
looks for things in angle brackets inside the yaml itself. As an example:

```yaml
seed: 1234
output_folder: !ref results/blstm/<seed>
```

This allows us to override the `seed` and have the output folder automatically
change its location accordingly. This tag also supports basic arithmetic:

```yaml
block_index: 1
cnn: !new:speechbrain.nnet.CNN.Conv
    out_channels: !ref <block_index> * 64
    kernel_size: (3, 3)
```

You can also refer to other references, and to sub-nodes using dot-notation.

```yaml
block_index: 1
cnn1: !new:speechbrain.nnet.CNN.Conv
    out_channels: !ref <block_index> * 64
    kernel_size: (3, 3)
cnn2: !new:speechbrain.nnet.CNN.Conv
    out_channels: !ref <cnn1.out_channels>
    kernel_size: (3, 3)
```

Finally, you can make references to objects, not just scalars.

```yaml
block_index: 1
cnn1: !new:speechbrain.nnet.CNN.Conv
    out_channels: !ref <block_index> * 64
    kernel_size: (3, 3)
cnn2: !ref <cnn1>
```

However, note that this makes only a shallow copy, so that here `cnn2`
refers to the same layer as `cnn1`. If you want a deep copy, use a tag
that we've developed to work very similarly to `!ref`, written `!copy`:

```yaml
block_index: 1
cnn1: !new:speechbrain.nnet.CNN.Conv
    out_channels: !ref <block_index> * 64
    kernel_size: (3, 3)
cnn2: !copy <cnn1>
```

One last minor extension to the yaml syntax we've made is to implicitly
resolve any string starting with `(` and ending with `)` to a tuple.
This makes the use of YAML more intuitive for Python users.


How to use Extended YAML
========================

First, we'd like to note that all of the following extensions are available
by loading yaml using the `speechbrain.yaml.load_extended_yaml` function.
This function returns a namespace object, so that the top-level items
are conveniently available using dot-notation. Using the yaml example
above:

```python
>>> from speechbrain.yaml import load_extended_yaml
>>> with open("hyperparameters.yaml") as f:
...     hyperparameters = load_extended_yaml(f)
>>> hyperparameters.block_index
1
>>> hyperparameters.cnn1
Conv()
>>> hyperparameters.cnn1.out_channels
64
```

Also, `load_extended_yaml` takes an optional argument, `overrides`
which allows changes to any of the parameters listed in the YAML.
The following example demonstrates changing the `out_channels`
of the CNN layer:

```python
>>> overrides = {"block_index": 2}
>>> with open("hyperparameters.yaml") as f:
...    hyperparameters = load_extended_yaml(f, overrides)
>>> hyperparameters.block_index
2
>>> hyperparameters.cnn1.out_channels
128
```

Conclusion
==========

We've defined a number of extensions to the YAML syntax, designed to
make it easier to use for hyperparameter specification. Feedback is welcome!
