# PRs—testing through the kaleidoscope.

Sometimes, there are so many lenses, one gets lost in vision.
<br/>_(but the spectacle is fun)_

---

Some tools can be used by PR contributors and reviewers alike:
```
tests/.run-linters.sh
pytest tests/consistency
tests/.run-doctests.sh
tests/.run-unittests.sh
pytest tests/integration
```

## Business as usual
```
BEFORE
------
           (python)                              (yaml)

def func_sig(x, arg0, arg1=None):    |    my_var: !new:func_sig
    # just to demonstrate changes    |        arg0: 6.28 # tau
    if arg1 is None:                 |
        return x + arg0              |    my_other: !new:func_sig
    else:                            |        arg0: !ref <my_var>
        return x + arg1              |        arg1: 1/137 # fine structure constant


AFTER - A. Changes to function body &/or interface parameterization via YAML
-----
           (python)                              (yaml)

def func_sig(x, arg0,                |    my_arg: !new:func_sig
                arg1=None,):         |        arg0: 6.28
    if arg1 is None:                 |
        return x / arg0              |    my_other: !new:func_sig
    else:                            |        arg0: !ref <my_arg>
        return x - arg1              |        arg1: 0.0073
```

Summary of changes in V.A with comments and tasks (for reviewers):
1. new break in `func_sig` declaration (in the signature of the function)
   > __Reviewer: is every (tops every other) code line commented?__
2. Comment dropped (docstring changed)
   > __Reviewer: are outer calls documented with `Arguments`; `Returns` & `Example`?__
3. `x + arg0` to `x / arg0 ` & `x + arg1` to `x - arg1`
   > __Reviewer: is the expected behaviour of these lines covered through pytest checks (does it work as should)?__
4. Yaml: `my_var` to `my_arg`
   > __Reviewer: is the logic consistent between `train.py` & `hparams.yaml` (likewise for custom pretrained model interfaces: between `custom_model.py` & `hyperparameters.yaml`)?__
   > <br/>Note: conversely, if a hparams is required in script.py, simply run the script (either it fails/not).
5. Yaml comments dropped
   <br/> _(no reviewer task)_
6. Yaml: `1/137` to `0.0073`
   > __Reviewer: is the tutorial/recipe/script/snippet (still) functional after this change?__

By using the above tools, the reviewing task narrows down to what needs to be done, essentially.

## Legacy-preserving
```
BEFORE
------
           (python)                              (yaml)

def func_sig(x, arg0, arg1=None):    |    my_var: !new:func_sig
    # just to demonstrate changes    |        arg0: 6.28 # tau
    if arg1 is None:                 |
        return x + arg0              |    my_other: !new:func_sig
    else:                            |        arg0: !ref <my_var>
        return x + arg1              |        arg1: 1/137 # fine structure constant


AFTER - B. Changes to function signature (interface), legacy-preserving
-----
           (python)                              (yaml)

def func_sig(x, arg0, arg1=None,     |    my_arg: !new:func_sig
                arg2=true,):         |        arg0: 6.28
    return next_gen(x, arg0=arg0,    |
                       arg1=arg1,    |    my_other: !new:func_sig
                       arg2=arg2,)   |        arg0: !ref <my_arg>
                                     |        arg1: 0.0073
# the new interface being introduced |
def next_gen(x, arg0=6.28,           |    my_arg_same: !new:next_gen
                arg1=1/137,          |        arg1: None
                arg2=true,):         |
    if !arg2:                        |    my_other_same: !new:next_gen
        return x                     |
    if arg1 is None:                 |    my_new_feature: !new:next_gen
        return x / arg0              |        arg0: 2.718 # e
    else:                            |        arg1: 1.618 # what could it be...
        return x - arg1              |        arg2: false # ;-)
```
Summary:
1. added argument `arg2=true`
   > __Reviewer: do all interfaces still work & are all scripts still functional?__
2. new function `next_gen`
   > __Reviewer: are minimally required tests provided for docs, units & integrations?__
3. YAML: new variables `my_arg_same`; `my_other_same` & `my_new_feature`
   > __Reviewer: are all YAML files functional w/ old/new interface?__

Such PRs can be merged on the develop branch; the main reason to not break legacy immediately might be to make a new feature preliminary available as an EasterEgg – BUT – the intention is to break legacy with a subsequent PR.

## Legacy-breaking
```
BEFORE
------
           (python)                              (yaml)

def func_sig(x, arg0, arg1=None):    |    my_var: !new:func_sig
    # just to demonstrate changes    |        arg0: 6.28 # tau
    if arg1 is None:                 |
        return x + arg0              |    my_other: !new:func_sig
    else:                            |        arg0: !ref <my_var>
        return x + arg1              |        arg1: 1/137 # fine structure constant


AFTER - C. Changes to function signature (interface), legacy-breaking
-----
           (python)                              (yaml)

def next_gen(x, arg0=6.28,           |    my_arg: !new:next_gen
                arg1=1/137,          |        arg1: None
                arg2=true,):         |
    if !arg2:                        |    my_other: !new:next_gen
        return x                     |
    if arg1 is None:                 |    my_new_feature: !new:next_gen
        return x / arg0              |        arg0: 2.718
    else:                            |        arg1: 1.618
        return x - arg1              |        arg2: false

```
Summary:
1. Interface substitution, throughout
   > __Reviewer: all working under expected performance criteria (w/o recomputing all pretrained models)?__

Testing-wise, this on the same level as the legacy-breaking (needs the same tools; ideally provided by the contributor if non-exiting).
