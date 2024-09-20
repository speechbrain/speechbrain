# Project Structure & Ecosystem

"SpeechBrain" refers to both the software and recipes here on GitHub, and to a wider ecosystem spanning various platforms (PyPI, readthedocs, HuggingFace, DropBox).

This document hopes to untangle the general structure of the project and its ecosystem, for contributors and regular users.

## Directory Structure

This is not quite a complete list, but it gives a broad outline.

| Directory | Contents |
|-|-|
| **Core** [(API doc)](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.html) | |
| **[`speechbrain/`](speechbrain/)** | Source code for the core |
| **[`speechbrain/inference/`](speechbrain/inference/)** | Easy-to-use level inference code with HuggingFace integration |
| **[`speechbrain/utils/`](speechbrain/utils/)** | Miscellaneous utilities that don't really fit elsewhere |
| **Documentation** | |
| **[`docs/`](docs/)** | Documentation pages and configuration |
| **[`docs/tutorials/`](docs/tutorials/)** | Jupyter Notebook tutorials | 
| **Recipes** | | 
| **[`recipes/`](recipes/)** | Ready-to-use recipes under the form `dataset/task/model/` |
| **[`templates/`](templates/)** | Reference implementation for tasks to (optionally) use for new recipes |
| **Testing/linting/meta** | |
| **[`.github/`](.github/)** | GitHub issue/PR templates and Actions workflows for testing |
| **[`tests/`](tests/)** | Automated tests, some run under CI, some manually |
| **[`tools/`](tools/)** | One-off complete scripts and tools for specific tasks |
| **[`.pre-commit-config.yaml`](`.pre-commit-config.yaml`)** | Linter configuration (style check, formatting) |

## External Platforms

| URL | Contents |
|-|-|
|**<https://github.com/speechbrain/speechbrain>**| Official SpeechBrain repository |
|<https://speechbrain.github.io/>| Landing page (deployed from [here](https://github.com/speechbrain/speechbrain.github.io>)) |
|<https://github.com/speechbrain/benchmarks>| Standardized benchmarks based on SpeechBrain |
|<https://github.com/speechbrain/HyperPyYAML>| Official HyperPyYAML repository |
|<https://speechbrain.readthedocs.io>| Documentation and tutorials (deployed from [`docs/`](docs/)) |
|<https://huggingface.co/speechbrain>| Pre-trained models ready for inference |
| DropBox links in repository | Data, training logs and checkpoints |

## Testing Infrastructure

| Scope | Description |
|-|-|
| **CI-automated** | Tests that are verified continuously through Actions |
| Linting | Enforcing good practice, formatting, etc., see [`.pre-commit-config.yaml`](`.pre-commit-config.yaml`) |
| Consistency | Enforcing rules on YAMLs, presence of tests, among others |
| Doctests | Testing simple usecases at class/function level, and providing examples |
| Unit tests | Tests for specific components. Deeper testing than doctests |
| Integration tests | Testing for regressions at a larger scale (e.g. mini-recipes) |
| **Semi-manual** | Tests that are manually run by you or the Core Team at a varying frequency |
| [URL checks](tests/.run-url-checks.sh) | Checking for dead links in documentation, code and tutorials |
| [Recipe tests](tests/recipes/) | Test model training for all recipe `.csv` on sample data |
| [HuggingFace checks](tests/.run-HF-checks.sh) | Check if known models on HF seem to execute fine  |

<!--
## Contributing code to SpeechBrain

This assumes you have prior knowledge of Git and GitHub.

TODO MOVE THIS IS LITERALLY WHAT THE MAIN CONTRIBUTING DOCUMENT IS ABOUT OMG

### My code/branch is ready, now what?

1. Run linting checks by using `pre-commit run -a` (some fixes will be done automatically, some may require your intervention). Remember to commit any fixes.
2. Do you think your PR is ready to be reviewed by the team?
    - **Yes:** Create your PR normally
    - **Work in progress/I would like feedback before I continue:** Create a **draft** pull request using the dropdown menu on GitHub while creating a PR.
3. If you know your PR is relevant to the scope of work of a specific member, feel free to mention them.
4. Your PR should get picked up at some point by a reviewer. You may be mentioned later if they found changes to your code to be necessary. Sometimes, contributors will fix code themselves.

## The location of a change foreshadows its integrative complexity.

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
How would you approach testing each of them?
<br/>Such changes happen not only once, but on a regular basis, throughout all core modules.

Changes can be internal to a function &/or alter the function signature:
* function-internal changes are not of concern to other function (so long they do what they should),
* function signature changes impact the overall—the multi-platform ecosystem.

Legacy-breaking changes will impact the outline of all recipes:
<br/>how will all work after a change—, and after the next major refactoring (after that first one)?
-->