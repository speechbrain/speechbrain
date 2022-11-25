# Test coverage of SpeechBrain

This readme serves as overview and outlines proof concept for our testing coverage.
Given this is an after-the-fact reporting, future validity is discerned.
<br/>_(I.e.: This file is not generated from sources.)_

<br/> I. Functionality provided on multiple platforms in the SpeechBrain ecosystem.
<br/> II. How is "functionality" provided?
<br/> III. How is the SpeechBrain community improving quality?
<br/> IV. GitHub workflows, the SpeechBrain set-up
<br/> V. User tools for PR drafting = reviewer tools before merging
<br/> VI. Developer tools for refactoring
<br/> VII. Maintainer checks for releases
<br/> VIII. Future testing


## I. Functionality provided on multiple platforms, in the SpeechBrain ecosystem.

```
                  (documentation)           (tutorials)
                  .—————————————.            .———————.
                  | readthedocs |       ‚––> | Colab |
                  \—————————————/      ∕     \———————/
                         ^       ‚––––‘          |
    (release)            |      ∕                v
    .——————.       .———————————. (landing) .———————————.
    | PyPI | –––>  | github.io |  (page)   | templates |   (reference)
    \——————/       \———————————/       ‚–> \———————————/ (implementation)
        |                |        ‚–––‘          |
        v                v       ∕               v
.———————————–—.   .———————————–—.           .—————————.           .~~~~~~~~~~~~~.
| HyperPyYAML |~~~| speechbrain | ––––––––> | recipes | ––––––––> | HuggingFace |
\————————————–/   \————————————–/           \—————————/     ∕     \~~~~~~~~~~~~~/
  (usability)     (source/modules)          (use cases)    ∕    (pretrained models)
                                                          ∕
                        |                        |       ∕               |
                        v                        v      ∕                v
                  .~~~~~~~~~~~~~.            .~~~~~~~~.            .———————————.
                  |   PyTorch   | ––––––––-> | GDrive |            | Inference |
                  \~~~~~~~~~~~~~/            \~~~~~~~~/            \———————————/
                   (checkpoints)             (results)            (code snippets)
```

<br/>What is where?

1. https://speechbrain.github.io/
  <br/> a. via: https://github.com/speechbrain/speechbrain.github.io
  <br/> b. pointing to several tutorials on Google Colab
  <br/> `python & yaml`
2. https://github.com/speechbrain/speechbrain
  <br/> a. [docs](https://github.com/speechbrain/speechbrain/tree/develop/docs) for https://speechbrain.readthedocs.io/
  <br/> b. [recipes](https://github.com/speechbrain/speechbrain/tree/develop/recipes) 
  <br/>`python & yaml & README`
  <br/> c. [speechbrain](https://github.com/speechbrain/speechbrain/tree/develop/speechbrain), heavily tied with [HyperPyYAML](https://github.com/speechbrain/HyperPyYAML); released on [PyPI](https://pypi.org/project/speechbrain/ 
  <br/>`python & yaml`
  <br/> d. [templates](https://github.com/speechbrain/speechbrain/tree/develop/templates) 
  <br/>`python & yaml & README`
  <br/> e. [tools](https://github.com/speechbrain/speechbrain/tree/develop/tools) for non-core functionality 
  <br/>`perl; python & yaml`
3. https://huggingface.co/speechbrain/
  <br/> hosting several model cards (pretrained models with code snippets)
  <br/> `python & yaml`
  <br/> [option to host datasets]
4. Gdrive (and alike)
  <br/> hosting training results; checkpoints; ...

These points need testing coverage (demonstrated below).

linters: yaml; python

## II. How is functionality provided?

```
   (imported)        (used in)    (as units in) (integrated by)   (to code)
.——————————————.    .—————————.    .—————————.    .—————————.    .—————————.  |  code & yaml 
| dependencies | => | helpers | => | classes | => | modules | => | scripts |  | style checks 
\——————————————/    \—————————/    \—————————/    \—————————/    \—————————/  |   (linters)
        |                |              |              |              |
        v                v              v              v              v
 version updates     docstring      unittests     integration      tutorials,
 may change their   examples as      assert       tests ensure     templates,
    interface;     modular tests    expected     working vanilla    recipes &
 latest versions                    behaviour     experiments       snippets
 are controlled by                                               need advanced
requirements configs                                           testing strategies

    [irregular]     [ --- github push workflow actions --- ]  [ hybrid periodicity]
```

<br/> What coverage approaches are needed?

1. Dependencies: version control (check commit ID dates)
  <br/> see: [requirements.txt](https://github.com/speechbrain/speechbrain/blob/develop/requirements.txt)
  <br/> run: `find *txt . | grep extra`
2. Docstring tests: commented function signatures <br/>_(of functions intended for outer calls)_
3. [Unittests](https://github.com/speechbrain/speechbrain/tree/develop/tests/unittests) per function-critical code block
4. [Integration tests](https://github.com/speechbrain/speechbrain/tree/develop/tests/integration) for vanilla experiments to cover use-cases on a generic task basis
5. Advanced testing: standing interfaces & their refactoring
6. Linters for automated style checks & corrections of python & yaml code

_Also below: How to know all of II. covers all of I. ?_


## III. How is the SpeechBrain community improving quality, continuously?

All works: great !

If not... then, let's fix it :)

1. Raise your questions & engage in [Discussions](https://github.com/speechbrain/speechbrain/discussions)
2. Report a bug or request a feature, open [Issues](https://github.com/speechbrain/speechbrain/issues/new/choose)
3. Contribute [Pull requests](https://github.com/speechbrain/speechbrain/pulls)
4. Release pretrained models through SpeechBrain
   <br/> e.g. registering linking HuggingFace account to SpeechBrain for hosting your model card

(1.) and (2.) are w/o direct impact on the code base.

For (3.) and (4.), let's consider a full contribution cycle.
```
            .———————————.
            | Closed PR |  (but not merged)
         ‚->\———————————/<-˛
        ∕                   \                      (made it! :)
.——————————.              .—————————.              .———————————.
| Draft PR | –––––––––––> | Open PR | –––––––––––> | Merged PR |
\——————————/              \—————————/              \———————————/
 * create initial          * ensure all             * pre-release
   branch to improve         workflow                 checks (later)
 * state todo list           tests pass             * contribution
   and fulfill it          * collaborate              log entry
 * inquire feedback          on change              * part of next
   early on                  requests                 release tag

     |                          |                     (more below)
     v                          v

To push formatted code:    Review of:
git add ...                 A. changes to core modules [I.2.c]
pre-commit                  B. enhanced testing/documentation [I.2.a & II.2-II.4]
git status                  C. contributed tutorial [I.1.b]
git add ...                 D. new/edited template/tool [I.2.d & I.2.e]
git commit -m ...           E. added/modified recipe [1.2.b]
git push                    F. uploaded pretrained model [I.3]

Missed out on one?
pre-commit run --all-files  G. Well-formatted py & yaml files []
git status
git add ...
```

How to know test coverage changes of Open PRs to be merged?
<br/>_(snippet for cpu-only)_
```
# Example: install more dependencies to avoid ignoring modules
sudo apt-get install -y libsndfile1
pip install ctc_segmentation

# install coverage
pip install pytest-cov

# run the test (w/ duration reporting)
pytest --durations=0 --cov=speechbrain --cov-context=test --doctest-modules speechbrain tests --ignore=speechbrain/nnet/loss/transducer_loss.py
```
Example: _After collecting 506 testing items, 4687/17279 statements are reported "missing" (73% coverage)._

YET—python code of the core modules is not all to be covered; thus far, only, consistency is ensured for III.A (through III.B).

For III.D to III.F, more consistency checks are demanded:
* templates & recipes rely on YAML files, loaded thourgh [HyperPyYAML](https://github.com/speechbrain/HyperPyYAML)
* [recipe] READMEs contain URLs to HuggingFace & GDrive locations
* recipe-depending HuggingFace model cards contain interface code snippets

Below, we need to further clarify on consistency checks for:
I.1.a;
I.2.a;
II.1; II.5, and
III.C to III.G.
Through III., checks for I.2.c, the speechbrain core modules, are standing as of II.2 to II.4 (doctests; unittests & integration tests).
That II.2 to II.4 are carried out will be demonstrated below, also.

## IV. GitHub workflows, the SpeechBrain set-up

API configurations are located at [.github/workflows](https://github.com/speechbrain/speechbrain/tree/develop/.github/workflows)
<br/>_(all creating a one-time ubuntu-latest environment)_
1. [pre-commit.yml](https://github.com/speechbrain/speechbrain/blob/develop/.github/workflows/pre-commit.yml) [III.G]
   > SpeechBrain pre-commit / pre-commit (pull_request)
   * python-version: '3.8'
   * run pre-commit action, configured in [.pre-commit-config.yaml](https://github.com/speechbrain/speechbrain/blob/develop/.pre-commit-config.yaml)
     * hook: https://github.com/pre-commit/pre-commit-hooks
       <br/> trailing-whitespace
       <br/> end-of-file-fixer
       <br/> requirements-txt-fixer
       <br/> mixed-line-ending
       <br/> check-added-large-files
     * hook: https://github.com/psf/black
       <br/> black
       <br/> click
     * hook: https://gitlab.com/pycqa/flake8.git
       <br/> flake8; see: [.flake8](https://github.com/speechbrain/speechbrain/blob/develop/.flake8)
     * hook: https://github.com/adrienverge/yamllint
       <br/> yamllint; see: [.yamllint.yaml](https://github.com/speechbrain/speechbrain/blob/develop/.yamllint.yaml)
2. [pythonapp.yml](https://github.com/speechbrain/speechbrain/blob/develop/.github/workflows/pythonapp.yml)
   > SpeechBrain toolkit CI / Tests (3.7) (pull_request)<br/>
   > SpeechBrain toolkit CI / Tests (3.8) (pull_request)<br/>
   > SpeechBrain toolkit CI / Tests (3.9) (pull_request)
   * python-version: [3.7, 3.8, 3.9]
   * create fresh environment
     ```shell
     sudo apt-get install -y libsndfile1
     pip install -r requirements.txt
     pip install --editable .
     pip install ctc-segmentation
     ```
   * run PyTest checks
     <br/> see: [pytest.ini](https://github.com/speechbrain/speechbrain/blob/develop/pytest.ini) - files: `test_*.py`; `check_*.py`; `example_*.py` & norecursedirs
     <br/> see: [conftest.py](https://github.com/speechbrain/speechbrain/blob/develop/conftest.py) - prepare test item collection & direct discovery
     ```
     # excerpts
     parser.addoption("--device", action="store", default="cpu")
     ...
     try:
       import numba  # noqa: F401
     except ModuleNotFoundError:
       collect_ignore.append("speechbrain/nnet/loss/transducer_loss.py")
     ...
     ```
     * a. hook: Consistency tests with pytest [part of II.5]
       <br/> `pytest tests/consistency`
     * b. hook: Unittests with pytest [II.3]
       <br/> `pytest tests/unittests`
     * c. hook: Doctests with pytest [II.2]
       <br/> `pytest --doctest-modules speechbrain`
     * d. hook: Integration tests with pytest [II.4]
       <br/> `pytest tests/integration`
3. [verify-docs-gen.yml](https://github.com/speechbrain/speechbrain/blob/develop/.github/workflows/verify-docs-gen.yml) [I.2.a]
   > Verify docs generation / docs (pull_request)
   * python-version: '3.8'
   * create fresh environment
     ```shell
     pip install -r requirements.txt
     pip install --editable .
     pip install -r docs/docs-requirements.txt
     ```
   * generates docs
     ```shell
     cd docs
     make html
     ```
   * compare: [.readthedocs.yaml](https://github.com/speechbrain/speechbrain/blob/develop/.readthedocs.yaml) - python version: 3.8
4. [newtag.yml](https://github.com/speechbrain/speechbrain/blob/develop/.github/workflows/newtag.yml)
   > Draft release when pushing new tag
   * tagging of `develop` branch commit ID
   * before
     * follow through [tests/PRE-RELEASE-TESTS.md](https://github.com/speechbrain/speechbrain/blob/develop/tests/PRE-RELEASE-TESTS.md)
       * set-up fresh environment [re/assert II.1]
       * run `pytest` [re/assert II.3; II.4; part of II.5]
       * a. hook: [tests/.run-load-yaml-tests.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-load-yaml-tests.sh) [another part of II.5, parts of III.D & III.E]
       * b. hook: [tests/.run-recipe-tests.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-recipe-tests.sh) [another part of II.5, parts of III.D to III.F]
       * c. hook: [tests/.run-HF-checks.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-HF-checks.sh) [another part of II.5, parts of III.D to III.F]
       * d. hook: [ests/.run-url-checks.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-url-checks.sh) [another part of II.5, parts of III.D to III.F]
     * update of [speechbrain/version.txt](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/version.txt) to the next
   * action: draft push to `main` branch
     <br/> implies pre-push hook, see: [.pre-push-config.yaml](https://github.com/speechbrain/speechbrain/blob/develop/.pre-push-config.yaml) with hooks to:
     * e. [tests/.run-linters.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-linters.sh) [re/assert III.G]
     * f. [tests/.run-unittests.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-unittests.sh) [re/assert II.3]
     * g. [tests/.run-doctests.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-doctests.sh) [re/assert II.2]
5. [release.yml](https://github.com/speechbrain/speechbrain/blob/develop/.github/workflows/release.yml) [I.2.c]
   > Publish to PyPI
   * python-version: 3.8
   * action: checkout to `main` branch
   * creates: `pypa/build` for binary wheel and source tarball
   * action: Publish to PyPI via `pypa/gh-action-pypi-publish@master`
     <br/> implies use of
     * [LICENSE](https://github.com/speechbrain/speechbrain/blob/develop/LICENSE)
     * [README.md](https://github.com/speechbrain/speechbrain/blob/develop/README.md)
     * [pyproject.toml](https://github.com/speechbrain/speechbrain/blob/develop/pyproject.toml) - target-version = ['py38']
     * [setup.py](https://github.com/speechbrain/speechbrain/blob/develop/setup.py)
       * python_requires=">=3.7",
       * uses: [speechbrain/version.txt](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/version.txt)
       * requires:
         ```
          "hyperpyyaml",
          "joblib",
          "numpy",
          "packaging",
          "scipy",
          "sentencepiece",
          "torch>=1.9",
          "torchaudio",
          "tqdm",
          "huggingface_hub",
         ```
       * points to https://speechbrain.github.io/ [part of I.1.a]

The versions of tools used/hooked in these checks are controlled via [lint-requirements.txt](https://github.com/speechbrain/speechbrain/blob/develop/lint-requirements.txt), a nested dependency in [requirements.txt](https://github.com/speechbrain/speechbrain/blob/develop/requirements.txt).
With major version releases of SpeechBrain, the versions of each hook should be updated—alongside requirement consistency in source, testing & builds incl. running spell-checking. [II.1]

_Note: [PyTorch statement](https://pytorch.org/get-started/locally/) on Python versions (as of 2022-11-09)_
> _It is recommended that you use Python 3.6, 3.7 or 3.8_

---

Info: although our PyTorch requirements are
```
torch>=1.9.0
torchaudio>=0.9.0
```
our tests cover one PyTorch version only, _the latest_.

---

Summary of how II covers I in terms of test invocation.
```

Concerning III.A-G, w/o III.C:
IV.4                       -> II.1
IV.2.c                     -> II.2
IV.2.b                     -> II.3
IV.2.d                     -> II.4
IV.4.a to IV.4.d & IV.2.a  -> II.5
IV.1                       -> II.6

I.1.a                       <= IV.5
I.1.b python & yaml         N/A (TODO); relates to III.C
I.2.a                       <= IV.3
I.2.b python                <= IV.2.a & IV.4.b
I.2.b yaml                  <= IV.2.a & IV.4.a
I.2.b README                <= IV.4.d
I.2.c python                <= IV.2.{b,c,d} & IV.5 | core modules
I.2.c yaml                  <= IV.2.{b,c,d} & IV.5 | core modules
I.2.d python                <= IV.2.a & IV.4.b
I.2.d yaml                  <= IV.2.a & IV.4.a
I.2.d README                <= IV.4.d
I.2.e perl                  (indirect via IV.2.{b,c})
I.2.e python                <= IV.2.b
I.2.e yaml                  (indirect via IV.2.{b,c}) 
I.3 python snippets         <= IV.4.c
I.3 python pretrained model (indirect via IV.4.c)
I.3 yaml pretrained model   (indirect via IV.4.c)
I.4                         (indirect via IV.4.b)
```

This schema holds so long as interfaces remain unchanged.
Hence, below, we need to address refactorings, their natures, and how we ensure continuity of code & validity of pre-trained models through testing.

We are moving testing from the sphere of pytest to the multi-platform SpeechBrain ecosystem. 
Therefore, testing of recipes and of pre-trained models is explained in more detail.

## V. User tools for PR drafting = reviewer tools before merging

Yet another lens for the testing kaleidoscope :D

_Food for thought: The location of a change foreshadows its integrative complexity._
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

Summary of changes in V.A with comments and tasks (for reviewers):
1. new break in `func_sig` declartion (in the signature of the function);
   <br/>yet—this is formatting only, which is checked by pre-commit/linters [IV.1].
   > __Reviewer: is every (tops every other) code line commented?__
2. Comment dropped (docstring changed); [IV.2.c]
   > __Reviewer: are outer calls documented with `Arguments`; `Returns` & `Example`?__
3. `x + arg0` to `x / arg0 ` & `x + arg1` to `x - arg1`; [IV.2.a to IV.2.d]
   > __Reviewer: is the expected behaviour of these lines covered through pytest checks (does it work as should)?__
4. Yaml: `my_var` to `my_arg`; [IV.2.a & IV.4.a] (consistency checks; also with use in Python scripts, nothing should remain unused & all hparams used are declared)
   > __Reviewer: is the logic consistent between `train.py` & `hparams.yaml` (likewise for custom pretrained model interfaces: between `custom_model.py` & `hyperparameters.yaml`)?__
   > <br/>Note: conversely, if a hparams is required in script.py, simply run the script (either it fails/not).
5. Yaml comments dropped; [IV.1] checks formatting issues (e.g. trailing whitespaces)
   <br/> _(no reviewer task)_
6. Yaml: `1/137` to `0.0073`; [IV.4.b & IV.4.c] (recipe & pretrained model checks: is data processed to the end & for some, are certain performance test criteria fulfilled)
   > __Reviewer: is the tutorial/recipe/script/snippet (still) functional after this change?__

These are the conventional types of changes to templates, recipes, and pretrained models. 
V.A.6 hints at pytest limitations (the recipe folder is not part of doc; unit/integration tests).
This demands to understand better the composure of templates & recipes.

_Note: Changes as shown in V.B & V.C are discussed in the next section._

---

Templates & recipes comprise:
* `recipes/DATASET/prepare_data.py` – a Data prep file
  > __Reviewer: with OpenData, does the recipe work with `--debug`? If no data is available, skip this preparation and use `tests/samples` data to check if the recipe breaks/not.__
  <br/><hr/>
  > **User:** provide required _test_debug_flags_ for the reviewing task.
* `recipes/DATASET/extra_requirements.txt` – additional dependencies
* `recipes/DATASET/TASK/METHOD/extra_requirements.txt` – particular, additional dependencies
  > _Note: this can lead to conflicting recipes / which need to point to different e.g. HF hub caches to not conflict one another._
* `recipes/DATASET/TASK/METHOD/train.py` – a _Script_file_
* `recipes/DATASET/TASK/METHOD/hparams/hparam.yaml` – a _Hparam_file_
* `recipes/DATASET/TASK/METHOD/README.md` – a _Readme_file_, which points to
  * some GDrive url – a _Result_url_ [optional]
  * some HuggingFace url – a _HF_repo_ [optional], which has
    * pretrained model – `hyperparameters.yaml` to be loaded either by [a pretrained interface](https://github.com/speechbrain/speechbrain/tree/develop/speechbrain/pretrained) or a custom interface
    * code snippets, for demonstration
  * additional references, incl. further URLs
  > _Note: [IV.4.d] checks that all URLs referenced (in .py, .md & .txt files) are valid._
* `tests/recipes/DATASET.csv` – a summary of testing parameters for templates & recipes, including derived pretrained models
  <br/>_(as hinted above; example: [tests/recipes/LibriSpeech.csv](https://github.com/speechbrain/speechbrain/tree/develop/tests/recipes/LibriSpeech.csv):2)_
  * Task
    >_ASR_
  * Dataset
    > _LibriSpeech_
  * Script_file
    > _recipes/LibriSpeech/ASR/CTC/train_with_wav2vec.py_
  * Hparam_file
    > _recipes/LibriSpeech/ASR/CTC/hparams/train_hf_wav2vec.yaml_
  * Data_prep_file
    > _recipes/LibriSpeech/ASR/CTC/librispeech_prepare.py_
  * Readme_file
    > _recipes/LibriSpeech/ASR/CTC/README.md_
  * Result_url (mandatory/optional?)
    > _https://drive.google.com/drive/folders/1pg0QzW-LqAISG8Viw_lUTGjXwOqh7gkl?usp=sharing_
  * HF_repo (optional)
    > _https://huggingface.co/speechbrain/asr-wav2vec2-librispeech_
  * test_debug_flags
    > _--data_folder=tests/samples/ASR/ --train_csv=tests/samples/annotation/ASR_train.csv --valid_csv=tests/samples/annotation/ASR_train.csv --test_csv=[tests/samples/annotation/ASR_train.csv] --number_of_epochs=10 --skip_prep=True --wav2vec2_folder=tests/tmp/wav2vec2_checkpoint_
  * test_debug_checks (optional)
    > _"file_exists=[env.log,hyperparams.yaml,log.txt,train_log.txt,train_with_wav2vec.py,wer_ASR_train.txt,save/label_encoder.txt] performance_check=[train_log.txt, train loss, <3.5, epoch: 10]"_

These testing parameters are used by [IV.4.a to IV.4.d] (checks before releases) and by [IV.2.a] (checks after each `git push`).

---

Yet: _How to know all pretrained model (e.g. on HuggingFace) are enlisted?_

---

From the above hooks, some tools can be used by PR contributors and reviewers alike:
```
tests/.run-linters.sh
pytest tests/consistency
tests/.run-doctests.sh
tests/.run-unittests.sh
pytest tests/integration
```

---

Yet: _How to avail the following tools to github workflow checks concerning what was changed only, and what is impacted from that change?_
```
tests/.run-load-yaml-tests.sh
tests/.run-recipe-tests.sh
tests/.run-HF-checks.sh
tests/.run-url-checks.sh
```

---

## VI. Developer tools for refactoring

Changes in V.B:
1. added argument `arg2=true`; [IV.2.a to IV.2.d & IV.4.a to IV.4.c] & more below
   > __Reviewer: do all interfaces still work & are all scripts still functional?__
2. new function `next_gen`; [IV.2.a to IV.2.d & IV.4.a to IV.4.c] & more below
   > __Reviewer: are minimally required tests provided for docs, units & integrations?__
3. YAML: new variables `my_arg_same`; `my_other_same` & `my_new_feature`; [IV.2.a to IV.2.d & IV.4.a to IV.4.c] & more below
   > __Reviewer: are all YAML files functional w/ old/new interface?__

Such PRs can be merged on the develop branch; the main reason to not break legacy immediately might be to make a new feature preliminary available as an EasterEgg – BUT – the intention is to break legacy with a subsequent PR.

Changes in V.C:
1. Interface substitution, throughout; [IV.2.a to IV.2.d & IV.4.a to IV.4.c] & more below
   > __Reviewer: all working under expected performance criteria (w/o recomputing all pretrained models)?__

Testing-wise, this on the same level as V.A—but following through this needs the testing tools provided for V.B.

Info: There is no one tool, to check these changes adequately (and report on coverage), for the SpeechBrain ecosystem.
Revisit section I., and identify gaps (compare "Future testing" section).

---

Branch topology: release <- CI/CD <- ecosystem-spanning refactorings. 
```
    release | main             | business 
      CI/CD |   \--- develop   | as usual
  ecosystem |         \   \<~> testing-refactoring   |  the tricky
refactoring |          \--- unstable <~>/            | bits & pieces
```

The `testing-refactoring` branch contains this folder:<br/>
https://github.com/speechbrain/speechbrain/tree/testing-refactoring/updates_pretrained_models

which contains folders identical to the model names uploaded to HuggingFace:<br/>
https://huggingface.co/speechbrain

e.g. [testing-refactoring/updates_pretrained_models/asr-wav2vec2-librispeech](https://github.com/speechbrain/speechbrain/tree/testing-refactoring/updates_pretrained_models/asr-wav2vec2-librispeech) outlines testing of the pretrained model for [speechbrain/asr-wav2vec2-librispeech](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech) and the folders can contain:
* `test.yaml` - test definition w/ integrated code [**mandatory**]
* `hyperparams.yaml` - the standing (or updated) specification [**mandatory**] 
* `custom_interface.py` - the standing (or updated) custom interface [optional]

_Note: changing parameters mean either a model revision &/or a new model._

While `hyperparams.yaml` & `custom_interface.py` shall be updated through PRs complementary to conventional PRs, `test.yaml` is to be defined once only (and fixed when needed).
Such a complementary PR is for example: 
https://github.com/speechbrain/speechbrain/pull/1623

Depending on the testing need, `test.yaml` grows - some examples
1. [ssl-wav2vec2-base-librispeech/test.yaml](https://github.com/speechbrain/speechbrain/blob/testing-refactoring/updates_pretrained_models/ssl-wav2vec2-base-librispeech/test.yaml) - the play between test sample, interface class, and batch function is handled via HF testing in `tests/utils`
   ```yaml
   sample: example.wav # test audio provided via HF repo
   cls: WaveformEncoder # existing speechbrain.pretrained.interfraces class
   fnx: encode_batch # it's batch-wise function after audio loading
   ```
2. [asr-wav2vec2-librispeech/test.yaml](https://github.com/speechbrain/speechbrain/blob/testing-refactoring/updates_pretrained_models/asr-wav2vec2-librispeech/test.yaml) - testing single example & against a dataset test partition
   ```yaml
   sample: example.wav # as above
   cls: EncoderASR # as above
   fnx: transcribe_batch # as above
   dataset: LibriSpeech # which dataset to use -> will create a tests/tmp/LibriSpeech folder
   recipe_yaml: recipes/LibriSpeech/ASR/CTC/hparams/train_hf_wav2vec.yaml # the training recipe for dataloader etc
   overrides: # what of the recipe_yaml needs to be overriden
     output_folder: !ref tests/tmp/<dataset> # the output folder is at the tmp dataset (data prep & eval tasks only)
   dataio: from recipes.LibriSpeech.ASR.CTC.train_with_wav2vec import dataio_prepare # which dataio_prepare to import 
   test_datasets: dataio_prepare(recipe_hparams)[2] # where to get the test dataset from that prep pipeline (w/ input args)
   test_loader: test_dataloader_opts # dataloader name as in recipe_yaml
   performance: # which metric classes are used in the training recipe
     CER: # name for testing
       handler: cer_computer # name as in recipe_yaml
       field: error_rate # field/function as used in train script
     WER: # another one
       handler: error_rate_computer # another one
       field: error_rate # another one
   predicted: predictions[0] # what of the forward to use to compute metrics
   ```
3. [emotion-recognition-wav2vec2-IEMOCAP/test.yaml](https://github.com/speechbrain/speechbrain/blob/testing-refactoring/updates_pretrained_models/emotion-recognition-wav2vec2-IEMOCAP/test.yaml) - custom interfaces
   ```yaml
   sample: anger.wav # as above
   cls: CustomEncoderWav2vec2Classifier # => name of custom class provided through custom interface
   fnx: classify_batch # as above
   foreign: custom_interface.py # name of custom interface availed through HF repo
   dataset: IEMOCAP # as above
   recipe_yaml: recipes/IEMOCAP/emotion_recognition/hparams/train_with_wav2vec2.yaml # as above
   overrides: # as above
     output_folder: !ref tests/tmp/<dataset> # as above
   dataio: from recipes.IEMOCAP.emotion_recognition.train_with_wav2vec2 import dataio_prep # as above
   test_datasets: dataio_prepare(recipe_hparams)["test"] # as above
   test_loader: dataloader_options # as above
   performance: # as above
     ClassError: # as above
       handler: error_stats # as above
       field: average # as above
   predicted: predictions[0] # as above
   ```

When testing the HF snippets, use the functions `gather_expected_results()` and `gather_refactoring_results()`.
They will create another yaml in which the gather before/after refactoring test results.
While standing interfaces are drawn from HF repos, their updated/refactored counterparts need to be specified to clone the PR git+branch into, e.g., `tests/tmp/hf_interfaces`. See the default values:
```python
def gather_refactoring_results(
    new_interfaces_git="https://github.com/speechbrain/speechbrain",  # change to yours
    new_interfaces_branch="testing-refactoring",  # maybe you have another branch
    new_interfaces_local_dir="tests/tmp/hf_interfaces",  # you can leave this, or put it elsewhere
    yaml_path="tests/tmp/refactoring_results.yaml",  # same here, change only if necessary
):
    ...
```

When testing against a dataset's test partition, this function is used: `test_performance()`.
It will be handled through the main function of `tests/utils/refactoring_checks.py`, which expects its own config e.g.:
`tests/utils/overrides.yaml`.

Example:
```yaml
LibriSpeech_data: !PLACEHOLDER
CommonVoice_EN_data: !PLACEHOLDER
CommonVoice_FR_data: !PLACEHOLDER
IEMOCAP_data: !PLACEHOLDER

new_interfaces_git: https://github.com/speechbrain/speechbrain
new_interfaces_branch: testing-refactoring
new_interfaces_local_dir: tests/tmp/hf_interfaces

# Filter HF repos (will be used in a local glob dir crawling)
# glob_filter: "*wav2vec2*"
# glob_filter: "*libri*"
glob_filter: "*"

# put False to test 'before' only, e.g. via override
after: True

LibriSpeech:
  data_folder: !ref <LibriSpeech_data>
  skip_prep: True

CommonVoice_EN:
  data_folder: !ref <CommonVoice_EN_data>

CommonVoice_FR:
  data_folder: !ref <CommonVoice_FR_data>

IEMOCAP:
  data_folder: !ref <IEMOCAP_data>
```

Example call:
```
python tests/integration/HuggingFace_transformers/refactoring_checks.py tests/integration/HuggingFace_transformers/overrides.yaml --LibriSpeech_data="" --CommonVoice_EN_data="" --CommonVoice_FR_data="" --IEMOCAP_data=""
--glob_filter="*commonvoice*"
```

The use case for this construction is a legacy-preserving refactoring, providing an alternative interface. [V.B]

---

The `unstable` branch serves to collect a series of legacy-breaking PRs before making a major release through develop. [V.C] 

_Note: ofc, the just introduced testing-refactoring strategy is applicable here, also. Especially, as it relaxes testing demands._ 


## VII. Maintainer checks for releases

Up until here, all the above madness should have settled.
Commit logs outline what happened; features are summarized.

_Note: a good point to check https://speechbrain.github.io/ is up-to-date._

The task at hand is:
* change the version number;
* compile a changelog, and
* release the latest version.

[IV.4 & IV.5] outline the process for a new release on PyPI; another CI/CD lifecycle begins.

## VIII. Future testing

how to know all GDrive & HF repos are referenced in tests/recipes?

tutorial tests [I.1.b => III.C => n/a]

pre-trained interfaces & data caching

targeted testing tools for where changes happened

suggestion tools around hyperpyyaml

once in a while run all recipes to some target (community retrains)

multiple/particular pytorch versions

further automate reviewer tasks (=> community self-service)

check readthedocs for consistent docmuentation style

speeding up recipe tests through hparam override runs into this issue for some recipes: Hyperparameters of nested/late-imported yamls cannot be changed through the override mechanism (unless provided for); see teacher/student in TIMIT

coverage tables for readme: python x pytorch versions

---

// summary (by I.x.* coverage) & appendix?

---

Futher reading:
<br/> https://breadcrumbscollector.tech/how-to-use-code-coverage-in-python-with-pytest/ (pointer by @Adel-Moumen)

---

```
---------- coverage: platform linux, python 3.9.12-final-0 -----------
Name                                                      Stmts   Miss  Cover
-----------------------------------------------------------------------------
speechbrain/alignment/aligner.py                            380     61    84%
speechbrain/alignment/ctc_segmentation.py                   189     10    95%
speechbrain/core.py                                         424    155    63% <== < 80%
speechbrain/dataio/batch.py                                  99      8    92%
speechbrain/dataio/dataio.py                                279     50    82%
speechbrain/dataio/dataloader.py                            140     25    82%
speechbrain/dataio/dataset.py                               100      8    92%
speechbrain/dataio/encoder.py                               328     46    86%
speechbrain/dataio/iterators.py                              80     62    22% <== < 80%
speechbrain/dataio/legacy.py                                121     41    66% <== < 80%
speechbrain/dataio/preprocess.py                             22      4    82%
speechbrain/dataio/sampler.py                               224     61    73% <== < 80%
speechbrain/dataio/wer.py                                    63     54    14% <== < 80%
speechbrain/decoders/ctc.py                                 111     89    20% <== < 80%
speechbrain/decoders/seq2seq.py                             370     46    88%
speechbrain/decoders/transducer.py                          133     64    52% <== < 80%
speechbrain/lm/arpa.py                                       77      3    96%
speechbrain/lm/counting.py                                   37      4    89%
speechbrain/lm/ngram.py                                      36      1    97%
speechbrain/lobes/augment.py                                154     55    64% <== < 80%
speechbrain/lobes/beamform_multimic.py                       20     14    30% <== < 80%
speechbrain/lobes/features.py                                96      9    91%
speechbrain/lobes/models/CRDNN.py                            52     12    77% <== < 80%
speechbrain/lobes/models/ContextNet.py                       83      3    96%
speechbrain/lobes/models/ECAPA_TDNN.py                      157      7    96%
speechbrain/lobes/models/HifiGAN.py                         321    146    55% <== < 80%
speechbrain/lobes/models/MetricGAN.py                        74     29    61% <== < 80%
speechbrain/lobes/models/Tacotron2.py                       364     66    82%
speechbrain/lobes/models/conv_tasnet.py                     121      6    95%
speechbrain/lobes/models/dual_path.py                       357     55    85%
speechbrain/lobes/models/fairseq_wav2vec.py                  93     93     0% <== < 80%
speechbrain/lobes/models/g2p/dataio.py                      136    107    21% <== < 80%
speechbrain/lobes/models/g2p/homograph.py                   118     20    83%
speechbrain/lobes/models/g2p/model.py                       132    109    17% <== < 80%
speechbrain/lobes/models/huggingface_wav2vec.py             145     47    68% <== < 80%
speechbrain/lobes/models/resepformer.py                     180     21    88%
speechbrain/lobes/models/segan_model.py                     102     88    14% <== < 80%
speechbrain/lobes/models/transformer/Conformer.py           111      7    94%
speechbrain/lobes/models/transformer/Transformer.py         180     22    88%
speechbrain/lobes/models/transformer/TransformerASR.py       92     28    70% <== < 80%
speechbrain/lobes/models/transformer/TransformerLM.py        47      5    89%
speechbrain/lobes/models/transformer/TransformerSE.py        20      2    90%
speechbrain/lobes/models/transformer/TransformerST.py        81     60    26% <== < 80%
speechbrain/lobes/models/wav2vec.py                         123     55    55% <== < 80%
speechbrain/nnet/CNN.py                                     417     56    87%
speechbrain/nnet/RNN.py                                     471     51    89%
speechbrain/nnet/activations.py                              39      1    97%
speechbrain/nnet/attention.py                               234     44    81%
speechbrain/nnet/complex_networks/c_CNN.py                  130     23    82%
speechbrain/nnet/complex_networks/c_RNN.py                  374     67    82%
speechbrain/nnet/complex_networks/c_normalization.py        277     68    75% <== < 80%
speechbrain/nnet/complex_networks/c_ops.py                  108     40    63% <== < 80%
speechbrain/nnet/containers.py                              139     14    90%
speechbrain/nnet/linear.py                                   27      1    96%
speechbrain/nnet/loss/si_snr_loss.py                         20     16    20% <== < 80%
speechbrain/nnet/loss/stoi_loss.py                           81      1    99%
speechbrain/nnet/loss/transducer_loss.py                    136    136     0% <== < 80%
speechbrain/nnet/losses.py                                  323    112    65% <== < 80%
speechbrain/nnet/normalization.py                           142      6    96%
speechbrain/nnet/pooling.py                                 156     31    80%
speechbrain/nnet/quantisers.py                               47      2    96%
speechbrain/nnet/quaternion_networks/q_CNN.py               150     25    83%
speechbrain/nnet/quaternion_networks/q_RNN.py               370     59    84%
speechbrain/nnet/quaternion_networks/q_linear.py             50     11    78% <== < 80%
speechbrain/nnet/quaternion_networks/q_normalization.py      44      4    91%
speechbrain/nnet/quaternion_networks/q_ops.py               229    122    47% <== < 80%
speechbrain/nnet/schedulers.py                              363    103    72% <== < 80%
speechbrain/nnet/transducer/transducer_joint.py              33      5    85%
speechbrain/pretrained/fetching.py                           48      6    88%
speechbrain/pretrained/interfaces.py                        786    338    57% <== < 80%
speechbrain/pretrained/training.py                           33     28    15% <== < 80%
speechbrain/processing/PLDA_LDA.py                          345     96    72% <== < 80%
speechbrain/processing/decomposition.py                     102      8    92%
speechbrain/processing/diarization.py                       319    157    51% <== < 80%
speechbrain/processing/features.py                          359     75    79% <== < 80%
speechbrain/processing/multi_mic.py                         345      2    99%
speechbrain/processing/signal_processing.py                 166     39    77% <== < 80%
speechbrain/processing/speech_augmentation.py               386     34    91%
speechbrain/tokenizers/SentencePiece.py                     181     74    59% <== < 80%
speechbrain/utils/Accuracy.py                                24     17    29% <== < 80%
speechbrain/utils/DER.py                                     44     33    25% <== < 80%
speechbrain/utils/bleu.py                                    50     43    14% <== < 80%
speechbrain/utils/callchains.py                              28      5    82%
speechbrain/utils/checkpoints.py                            294     52    82%
speechbrain/utils/data_pipeline.py                          181     15    92%
speechbrain/utils/data_utils.py                             197     77    61% <== < 80%
speechbrain/utils/depgraph.py                                82      1    99%
speechbrain/utils/distributed.py                             61     37    39% <== < 80%
speechbrain/utils/edit_distance.py                          180     50    72% <== < 80%
speechbrain/utils/epoch_loop.py                              55     22    60% <== < 80%
speechbrain/utils/hparams.py                                  2      1    50% <== < 80%
speechbrain/utils/hpopt.py                                  134     41    69% <== < 80%
speechbrain/utils/logger.py                                  73     45    38% <== < 80%
speechbrain/utils/metric_stats.py                           285     48    83%
speechbrain/utils/parameter_transfer.py                      87     17    80%
speechbrain/utils/profiling.py                              191     54    72% <== < 80%
speechbrain/utils/superpowers.py                             20      6    70% <== < 80%
speechbrain/utils/text_to_sequence.py                        77     22    71% <== < 80%
speechbrain/utils/torch_audio_backend.py                      9      2    78% <== < 80%
speechbrain/utils/train_logger.py                           150    113    25% <== < 80%
speechbrain/wordemb/transformer.py                           90     67    26% <== < 80%
-----------------------------------------------------------------------------
TOTAL                                                     16782   4481    73%
```