Introduction
------------

A crucial element of systems for processing speech, as with any data-analysis
systems, is laying out all the hyperparameters of that system so they can be
easily examined and modified. For our hyperparameter specification we adopt
YAML (YAML Ain't Markup Language), a popular human-readable data-serialization
language. We also add a few useful extensions to the YAML language in order to
support a rather expansive idea of what constitutes a hyperparameter.

### Table of Contents
* [YAML basics](#yaml-basics)
* [Extended YAML](#extended-yaml)
    * [Objects](#objects)
    * [Aliases](#aliases)
    * [Tuples](#tuples)
* [How to use Extended YAML](#how-to-use-extended-yaml)
* [Conclusion](#conclusion)

YAML basics
-----------

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

Note that we've used a simple mapping to demonstrate the scalar nodes. A mapping
is a set of `key: value` pairs, defined so that the key can be used to easily
retrieve the corresponding value. In addition to the format above, mappings
can also be specified in a similar manner to JSON:

```yaml
{foo: 1, bar: 2.5, baz: "abc"}
```

Sequences, or lists of items, can also be specified in two ways:

```yaml
- foo
- bar
- baz
```

or

```yaml
[foo, bar, baz]
```

Note that when not using the inline version, YAML uses whitespace to denote
nested items:

```yaml
foo:
    a: 1
    b: 2
bar:
    - c
    - d
```

YAML has a few more advanced features (such as
[aliases](https://pyyaml.org/wiki/PyYAMLDocumentation#aliases) and
[merge keys](https://yaml.org/type/merge.html)) that you may want to explore
on your own. We will briefly discuss one here since it is relevant for our
extensions: [YAML tags](https://pyyaml.org/wiki/PyYAMLDocumentation#tags).

Tags are added with a `!` prefix, and they specify the type of the node. This
allows types beyond the simple types listed above to be used. PyYAML supports a
few additional types, such as:

```yaml
!!set                           # set
!!timestamp                     # datetime.datetime
!!python/tuple                  # tuple
!!python/complex                # complex
!!python/name:module.name       # A class or function
!!python/module:package.module  # A module
!!python/object/new:module.cls  # An instance of a class
```

These can all be quite useful, however we found that this system was a bit
cumbersome, especially with the frequency with which we were using them. So
we decided to implement some shortcuts for these features, which we are
calling "extended YAML".

Extended YAML
-------------

We make several extensions to yaml including easier object creation, nicer
aliases, and tuples.

### Objects

Our first extension is to simplify the structure for specifying an instance,
module, class, or function. As an example:

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
train_logger: !new:speechbrain.utils.train_logger.FileTrainLogger
    summary_fns:
        loss: !name:speechbrain.utils.train_logger.summarize_average
```

This yaml passes the `summarize_average()` function as a parameter to the
`train_logger`, which it uses to summarize an epoch's list of losses.


### Aliases

Another extension is a nicer alias system that supports things like
string interpolation. We've added a tag written `!ref` that
takes keys in angle brackets, and searches for them inside the yaml
file itself. As an example:

```yaml
seed: 1234
output_folder: !ref results/blstm/<seed>
```

This allows us to change the value of `seed` and automatically change the
output folder location accordingly. This tag also supports basic arithmetic:

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

Finally, you can make references to nodes that are objects, not just scalars.

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

### Tuples

One last minor extension to the yaml syntax we've made is to implicitly
resolve any string starting with `(` and ending with `)` to a tuple.
This makes the use of YAML more intuitive for Python users.


How to use Extended YAML
------------------------

All of the listed extensions are available by loading yaml using the
[speechbrain.yaml.load_extended_yaml](https://github.com/speechbrain/speechbrain/blob/a800131b3de3915a83393d3aead5670a08907b8d/speechbrain/yaml.py#L22)
function. This function returns a namespace object, so that the top-level items
are conveniently available using dot-notation. Assuming the last yaml
example above is stored in "hyperparameters.yaml", it can be loaded with:

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
----------

We've defined a number of extensions to the YAML syntax, designed to
make it easier to use for hyperparameter specification. Feedback is welcome!
