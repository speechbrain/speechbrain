# Project Structure & Ecosystem

"SpeechBrain" refers to both the software and recipes here on GitHub, and to a wider ecosystem spanning various platforms (PyPI, readthedocs, HuggingFace, DropBox).

This document hopes to untangle the general structure of the project and its ecosystem, for contributors and regular users.

## Directory Structure

This is not quite a complete list, but it gives a broad outline.

| Directory | Contents |
|-|-|
| **Core** [(API doc)](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.html) | |
| **[`speechbrain/`](https://github.com/speechbrain/speechbrain/tree/develop/speechbrain/)** | Source code for the core |
| **[`speechbrain/inference/`](https://github.com/speechbrain/speechbrain/tree/develop/speechbrain/inference/)** | Easy-to-use inference code with HuggingFace integration |
| **[`speechbrain/utils/`](https://github.com/speechbrain/speechbrain/tree/develop/speechbrain/utils/)** | Miscellaneous utilities that don't really fit elsewhere |
| **Documentation** | |
| **[`docs/`](https://github.com/speechbrain/speechbrain/tree/develop/docs/)** | Documentation pages and configuration |
| **[`docs/tutorials/`](https://github.com/speechbrain/speechbrain/tree/develop/docs/tutorials/)** | Jupyter Notebook tutorials |
| **Recipes** | |
| **[`recipes/`](https://github.com/speechbrain/speechbrain/tree/develop/recipes/)** | Ready-to-use recipes under the form `dataset/task/model/` |
| **[`templates/`](https://github.com/speechbrain/speechbrain/tree/develop/templates/)** | Reference implementation for tasks to (optionally) use for new recipes |
| **Testing/linting/meta** | |
| **[`.github/`](https://github.com/speechbrain/speechbrain/tree/develop/.github/)** | GitHub issue/PR templates and Actions workflows for testing |
| **[`tests/`](https://github.com/speechbrain/speechbrain/tree/develop/tests/)** | Automated tests, some run under CI, some manually |
| **[`tools/`](https://github.com/speechbrain/speechbrain/tree/develop/tools/)** | One-off complete scripts and tools for specific tasks |
| **[`.pre-commit-config.yaml`](`https://github.com/speechbrain/speechbrain/tree/develop/.pre-commit-config.yaml`)** | Linter configuration (style check, formatting) |

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
| Linting | Enforcing good practice, formatting, etc., see [`.pre-commit-config.yaml`](`https://github.com/speechbrain/speechbrain/tree/develop/.pre-commit-config.yaml`) |
| Consistency | Enforcing rules on YAMLs, presence of tests, among others |
| Doctests | Testing simple usecases at class/function level, and providing examples |
| Unit tests | Tests for specific components. Deeper testing than doctests |
| Integration tests | Testing for regressions at a larger scale (e.g. mini-recipes) |
| **Semi-manual** | Tests that are manually run by you or the Core Team at a varying frequency |
| [URL checks](https://github.com/speechbrain/speechbrain/tree/develop/tests/.run-url-checks.sh) | Checking for dead links in documentation, code and tutorials |
| [Recipe tests](https://github.com/speechbrain/speechbrain/tree/develop/tests/recipes/) | Test model training for all recipe `.csv` on sample data |
| [HuggingFace checks](https://github.com/speechbrain/speechbrain/tree/develop/tests/.run-HF-checks.sh) | Check if known models on HF seem to execute fine  |
