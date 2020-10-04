# The SpeechBrain Toolkit

[![](https://speechbrain.github.io/assets/logo_noname_rounded_small.png)](https://speechbrain.github.io/)

SpeechBrain is an **open-source** and **all-in-one** speech toolkit based on PyTorch.

The goal is to create a **single**, **flexible**, and **user-friendly** toolkit that can be used to easily develop **state-of-the-art speech technologies**, including systems for **speech recognition**, **speaker recognition**, **speech enhancement**, **multi-microphone signal processing** and many others.

*SpeechBrain is currently under development*.

# Table of Contents
- [Basics](#basics)
  * [License](#license)
  * [Development requirements](#development-requirements)
  * [Test installation](#test-installation)
  * [Folder Structure](#folder-structure)
  * [How to run an experiment](#how-to-run-an-experiment)
  * [Tensor format](#tensor-format)
- [Developer Guidelines](#developer-guidelines)
  * [GitHub](#github)
  * [Python](#python)
  * [Documentation](#documentation)
  * [Development tools](#development-tools)
  * [Continuous integration](#continuous-integration)
  * [Pull Request review guide](#pull-request-review-guide)

# Basics
In the following sections, the basic functionalities of SpeechBrain are described.

## License
SpeechBrain is licensed under the [Apache License v2.0](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)) (i.e., the same as the popular Kaldi toolkit).

## Development requirements
```
pip install -r requirements.txt
pip install --editable .
```
## Test Installation
Please, run the following script to make sure your installation is working:
```
pytest tests
pytest --doctest-modules speechbrain
```

## Folder Structure
The current version of Speechbrain has the following folder/file organization:
- **speechbrain**: The core library
- **recipes**: Experiment scripts and configurations
- **samples**: Some toy data for debugging and testing
- **tools**: Additional, runnable utility script
- **tests**: Unittests and integration tests

## How to run an experiment
In SpeechBrain experiments can be run from anywhere, but the experimental `results/`
directory will be created relative to the directory you are in. The most common
pattern for running experiments is as follows:

```
> cd recipes/<dataset>/<task>/
> python experiment.py params.yaml
```

At the top of the `experiment.py` file, the function
`sb.core.create_experiment_directory()` is called to create an output directory
(by default: `<cwd>/results/`). Both detailed logs and experiment output are saved
there. Furthermore, less detailed logs are output to stdout. The experiment script
and configuration (including possible command-line overrides) are also copied to
the output directory.

Also have a look at the YAML files in recipe directories. The YAML files
specify the hyperparameters of the recipes. The syntax is explained in
`speechbrain.utils.data_utils` in the docstring of `load_extended_yaml`.

A quick look at the extended YAML features, using an example:
```
seed: !PLACEHOLDER
output_dir: !ref results/vgg_blstm/<seed>
save_dir: !ref <output_dir>/save
data_folder: !PLACEHOLDER # e.g. /path/to/TIMIT

model: !new:speechbrain.lobes.models.CRDNN.CRDNN
    output_size: 40 # 39 phonemes + 1 blank symbol
    cnn_blocks: 2
    dnn_blocks: 2
```
- `!new:speechbrain.lobes.models.CRDNN.CRDNN` creates a `CRDNN` instance
  from the module `speechbrain.lobes.models.CRDNN`
- The indented keywords (`output_size` etc.) after it are passed as keyword
  arguments.
- `!ref <output_dir>/save` evaluates the part in angle brackets,
  referencing the YAML itself.
- `!PLACEHOLDER` simply errors out when loaded; it should be replaced by
  every user either by editing the yaml, or with an override (passed to
  `load_extended_yaml`).

For more details on yaml and our extensions, see [speechbrain/YAML.md](speechbrain/YAML.md)

# Tensor format
All the tensors within SpeechBrain are formatted using the following convention:
```
tensor=(batch, time_steps, channels[optional])
```
**The batch is always the first element, and time_steps is always the second one.
The rest of the dimensions are as many channels as you need**.

*Why do we need all tensors to have the same format?*
It is crucial to have a shared format for all the classes that process data and all the processing functions must be designed considering it. In SpeechBrain we might have pipelines of modules and if each module was based on different tensor formats, exchanging data between processing units would have been painful. Many formats are possible. For SpeechBrain we selected this one because
it is commonly used with recurrent layers, which are common in speech applications.

The format is very **flexible** and allows users to read different types of data. As we have seen, for **single-channel** raw waveform signals, the tensor will be ```tensor=(batch, time_steps)```, while for **multi-channel** raw waveform it will be ```tensor=(batch, time_steps, n_channel)```. Beyond waveforms, this format is used for any tensor in the computation pipeline. For instance,  fbank features that are formatted in this way:
```
(batch, time_step, n_filters)
```
The Short-Time Fourier Transform (STFT) tensor, instead, will be:
```
(batch, time_step, n_fft, 2)
```
where the "2" is because STFT is based on complex numbers with a real and imaginary part.
We can also read multi-channel SFT data, that will be formatted in this way:
```
(batch, time_step, n_fft, 2, n_audio_channels)
```


# Developer Guidelines
The goal is to write a set of libraries that process audio and speech in several different ways. The goal is to build a set of homogeneous libraries that are all compliant with the guidelines described in the following sub-sections.

## GitHub

Our development strategy is as follows:

1. Clone the main speechbrain repository (no fork necessary). SSH example:
    `git clone git@github.com:speechbrain/speechbrain`
2. Create a branch for specific feature you are developing.
    `git checkout -b your-branch-name`
3. Make + commit changes. Do not commit to `master`.
4. Push branch to github.
    `git push --set-upstream origin your-branch-name`
5. Navigate to github, and create a pull request from your branch to master.
6. A reviewer will be assigned to your PR to (hopefully quickly) review.
7. When reviewer is satisfied that the code improves repository quality, they can merge.
8. Reviewer should delete the source branch in the origin. You can do this in your
local copy too, after first making sure master is up-to-date (so git doesn't complain
that your branch changes aren't in master):
    `git checkout master`
    `git pull`
    `git branch -d your-branch-name`

Note that CI tests will be run when you create a PR. If you want to be sure that your
code will not fail these tests, we have set up pre-commit hooks that you can install:

```
> pip install pre-commit
> pre-commit install
> pre-commit install --hook-type pre-push --config .pre-push-config.yaml
```

These will automatically check the code when you commit and when you push.

## Python
### Version
SpeechBrain targets Python >= 3.7.

### Formatting
To settle code formatting, SpeechBrain adopts the [black](https://black.readthedocs.io/en/stable/) code formatter. Before submitting pull requests, please run the black formatter on your code.

In addition, we use [flake8](https://flake8.pycqa.org/en/latest/) to test code
style. Black as a tool does not enforce everything that flake8 tests.

You can run the formatter with: `black <file-or-directory>`. Similarly the
flake8 tests can be run with `flake8 <file-or-directory>`.

### Adding dependencies
In general, we strive to have as few dependencies as possible. However, we will
debate dependencies on a case-by-case basis. We value easy installability via
pip.

In case the dependency is only needed for a specific recipe or specific niche
module, we suggest the extra tools pattern: don't add the dependency to general
requirements, but check for installation and instruct to if the dependant code is run.

### Testing
We are adopting unit tests using
[pytest](https://docs.pytest.org/en/latest/contents.html).
Run unit tests with `pytest tests`

Additionally we have runnable doctests, though primarily these serve as
examples of the documented code. Run doctests with
`pytest --doctest-modules <file-or-directory>`

## Documentation
In SpeechBrain, we plan to provide documentation at different levels:

-  **Docstrings**: For each class/function in the repository, there should a header that properly describes its functionality, inputs, and outputs. It is also crucial to provide an example that shows how it can be used as a stand-alone function. We use [Numpy-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) docstrings. Consistent docstring style enables automatic API documentation. Also note the automatic doctests (see [here](#testing).

-  **Comments**: We encourage developers to write self-documenting code, and use
comments only where the implementation is surprising (to a Python-literate audience)
and where the implemented algorithm needs clarification.

In addition we have plans for:

-  **Website documentation**. In the SpeechBrain website, we will put detailed documentation where we put both the written tutorials and descriptions of all the functionalities of the toolkit.

-  **The SpeechBrain book**: Similarly to HTK (an old HMM-based speech toolkit developed by Cambridge) we plan to have a book that summarized the functionalities of speechbrain. The book will be mainly based on the website documentation, but also summarizing everything in a book, make it simpler to cite us.

-  **Video tutorial**: For each important topic (e.g, speech recognition, speaker recognition, speech enhancement) we plan to have some video tutorials.

## Development tools

### flake8
- A bit like pycodestyle: make sure the codestyle is according to guidelines.
- Compatible with black, in fact current flake8 config directly taken from black
- Code compliance can be tested simply with: `flake8 <file-or-directory>`
- You can bypass flake8 for a line with `# noqa: <QA-CODE> E.G. # noqa: E731 to allow lambda assignment`

### pre-commit
- Python tool which takes a configuration file (.pre-commit-config.yaml) and installs the git commit hooks specified in it.
- Git commit hooks are local so all who want to use them need to install them separately. This is done by: `pre-commit install`
- The tool can also install pre-push hooks. This is done separately with: `pre-commit install --hook-type pre-push --config .pre-push-config.yaml`

### the git pre-commit hooks
- Automatically run black
- Automatically fix trailing whitespace, end of file, sort requirements.txt
- Check that no large (>512kb) files are added by accident
- Automatically run flake8
- NOTE: If the hooks fix something (e.g. trailing whitespace or reformat with black), these changes are not automatically added and committed. You’ll have to add the fixed files again, and run the commit again. I guess this is a safeguard: don’t blindly accept changes from git hooks.
- NOTE2: The hooks are only run on the files you git added to the commit. This is in contrast to the CI pipeline, which always tests everything.

### the git pre-push hooks
- Black and flake8 as checks on the whole repo
- Unit-tests and doctests run on the whole repo
- These hooks can only be run in the full environment, so if you install these, you’ll need to e.g. activate virtualenv before pushing.

### pytest doctests
- This is not an additional dependency, but just that doctests are now run with pytest. Use: `pytest --doctest-modules <file-or-directory>`
- Thus you may use some pytest features in docstring examples. Most notably IMO: `tmpdir = getfixture('tmpdir')` which makes a temp dir and gives you a path to it, without needing a `with tempfile.TemporaryDirectory() as tmpdir:`

## Continuous integration

### What is CI
- loose term for a tight merge schedule
- typically assisted by automated testing and code review tools + practices

### CI / CD Pipelines
- GitHub Actions (and also available as third-party solution) feature, which automatically runs basically anything in reaction to git events.
- The CI pipeline is triggered by pull requests.
- Runs in a Ubuntu environment provided by GitHub
- GitHub offers a limited amount of CI pipeline minutes for free.
- CD would stand for continuous deployment, though we’re not doing that yet

### Our test suite
- Code linters are run. This means black and flake8. These are run on everything in speechbrain (the library directory), everything in recipes and everything in tests.
- Note that black will only error out if it would change a file here, but won’t reformat anything at this stage. You’ll have to run black on your code and push a new commit. The black commit hook helps avoid these errors.
- All unit-tests and doctests are run. You can check that these pass by running them yourself before pushing, with `pytest tests`  and `pytest --doctest-modules speechbrain`
- Integration tests (minimal examples). The minimal examples serve both to
  illustrate basic tasks and experiment running, but also as integration tests
  for the toolkit. For this purpose, any file which is prefixed with
  `example_` gets collected by pytest, and we add a short `test_` function at
  the end of the minimal examples.
- Currently, these are not run: docstring format tests (this should be added once the docstring conversion is done).
- If all tests pass, the whole pipeline takes a couple of minutes.

## Pull Request review guide

This is not a comprehensive code review guide, but some rough guidelines to unify the
general review practices across this project.

Firstly, let the review take some time. Try to read every line that was added,
if possible. Read the surrounding context of the code if needed to understand
the changes introduced. Possibly ask for clarifications if you don't understand.
If the pull request changes are hard to understand, maybe that that's a sign that
the code is not clear enough yet. However, don't nitpick every detail.

Secondly, focus on the major things first, and only then move on to smaller,
things. Level of importance:
- Immediate deal breakers (code does wrong thing, or feature shouldn't be added etc.)
- Things to fix before merging (Add more documentation, reduce complexity, etc.)
- More subjective things which could be changed if the author also agrees with you.

Thirdly, approve the pull request only once you believe the changes "improve overall
code health" as attested to [here](https://google.github.io/eng-practices/review/reviewer/standard.html).
However, this also means the pull request does not have to be perfect. Some features are
best implemented incrementally over many pull requests, and you should be more concerned
with making sure that the changes introduced lend themselves to painless further improvements.

Fourthly, use the tools that GitHub has: comment on specific code lines, suggest edits,
and once everyone involved has agreed that the PR is ready to merge, merge the
request and delete the feature branch.

Fifthly, though code review is a place for professional constructive criticism,
a nice strategy to show (and validate) that you understand what the PR is really
doing, is to provide some affirmative comments on its strengths.


# Zen of Speechbrain
SpeechBrain could be used for *research*, *academic*, *commercial*, *non-commercial* purposes. Ideally, the code should have the following features:
- **Simple:**  the code must be easy to understand even by students or by users that are not professional programmers or speech researchers. Try to design your code such that it can be easily read. Given alternatives with the same level of performance, code the simplest one. (the most explicit and straightforward manner is preferred)

- **Readable:** SpeechBrain mostly adopts the code style conventions in PEP8. The code written by the users must be compliant with that. We test codestyle with `flake8`

- **Efficient**: The code should be as efficient as possible. When possible, users should maximize the use of pytorch native operations.  Remember that in generally very convenient to process in parallel multiple signals rather than processing them one by one (e.g try to use *batch_size > 1* when possible). Test the code carefully with your favorite profiler (e.g, torch.utils.bottleneck https://pytorch.org/docs/stable/bottleneck.html ) to make sure there are no bottlenecks if your code.  Since we are not working in *c++* directly, performance can be an issue. Despite that, our goal is to make SpeechBrain as fast as possible.

- **modular:** Write your code such that is is very modular and fits well with the other functionalities of the toolkit. The idea is to develop a bunch of models that can be naturally interconnected with each other to implement complex modules.

- **well documented:**  Given the goals of SpeechBrain, writing a rich a good documentation is a crucial step. Many existing toolkits are not well documented, and we have to succeed in that to make the difference.
This aspect will be better described in the following sub-section.
