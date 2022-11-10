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
2. https://github.com/speechbrain/speechbrain
  <br/> a. [docs](https://github.com/speechbrain/speechbrain/tree/develop/docs) for https://speechbrain.readthedocs.io/
  <br/> b. [recipes](https://github.com/speechbrain/speechbrain/tree/develop/recipes)
  <br/> c. [speechbrain](https://github.com/speechbrain/speechbrain/tree/develop/speechbrain), heavily tied with [HyperPyYAML](https://github.com/speechbrain/HyperPyYAML); released on [PyPI](https://pypi.org/project/speechbrain/
  <br/> d. [templates](https://github.com/speechbrain/speechbrain/tree/develop/templates)
  <br/> e. [tools](https://github.com/speechbrain/speechbrain/tree/develop/tools) for non-core functionality
3. https://huggingface.co/speechbrain/
  <br/> hosting several model cards (pretrained models with code snippets)
  <br/> [option to host datasets]
4. Gdrive (and alike)
  <br/> hosting training results; checkpoints; ...

These points need testing coverage (demonstrated below).



## II. How is functionality provided?

```
   (imported)        (used in)    (as units in) (integrated by)   (to code)
.——————————————.    .—————————.    .—————————.    .—————————.    .—————————.
| dependencies | => | helpers | => | classes | => | modules | => | scripts |
\——————————————/    \—————————/    \—————————/    \—————————/    \—————————/
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
5. Advanced testing (see below)

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
pre-commit run --all-files  G. Well-formatted py & yaml files
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

This section demonstrated how II. covers all of I. in terms of test invocation.
Below, we clarify on the parts of II.5, our particular/advanced testing strategy, which encompasses III.C to III.F and is here, in parts, referred to IV.4.

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


AFTER - A. Changes to function body &/or interface invocation via YAML
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
        return x / arg0              |        arg0: 2.718 # e
    else:                            |        arg1: 1.618 # what could it be...
        return x - arg1              |        arg2: false # ;-)

```

Changes in V.A:
1. new break in `func_sig` declartion (in the signature of the function);
   <br/>yet—this is formatting only, which is checked by pre-commit/linters [IV.1].
   > __Reviewer: is every (tops every other) code line commented?__
2. Comment dropped (docstring changed); [IV.2.c]
   > __Reviewer: are outer calls documented with `Arguments`; `Returns` & `Example`?__
3. `x + arg0` to `x / arg0 ` & `x + arg1` to `x - arg1`; [IV.2.a to IV.2.d]
   > __Reviewer: is the expected behaviour of these lines covered through pytest checks (does it work as should)?__
4. Yaml: `my_var` to `my_arg`; [IV.2.a & IV.4.a] (consistency checks; also with use in Python scripts, nothing should remain unused & all hparams used are declared)
   > __Reviewer: is the logic consistent between `train.py` & `hparams.yaml` (likewise for custom pretrained model interfaces: between `custom_model.py` & `hyperparameters.yaml`)?__
5. Yaml comments dropped; [IV.1] checks formatting issues (e.g. trailing whitespaces)
   <br/> _(no reviewer task)_
6. Yaml: `1/137` to `0.0073`; [IV.4.b & IV.4.c] (recipe & pretrained model checks: is data processed to the end & for some, are certain performance test criteria fulfilled)
   > __Reviewer: is the tutorial/recipe/script/snippet (still) functional after this change?__

These are the conventional types of changes to templates, recipes, and pretrained models. [II.5 regarding III.D to III.F]

_Note: Changes as shown in V.B & V.C are discussed in the next section._

V.A.6 hints at pytest limitations (the recipe folder is not part of doc; unit/integration tests).
Which demands to understand better the composure of templates & recipes.

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

Yet: _How to know all pretrained model (e.g. on HuggingFace) are enlisted?_

> Note: support tools that both PR contributors and reviewers use offline would be a great addition to the github workflow checks.


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

### TODO: the `unstable` branch

### TODO: the `testing-refactoring` branch

---

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

---

// summary (by I.x.* coverage) & appendix?

---

Futher reading:
<br/> https://breadcrumbscollector.tech/how-to-use-code-coverage-in-python-with-pytest/ (pointer by @Adel-Moumen)

---

PyTest coverage, 2022-11-08 _(cpu-only)_
```
---------- coverage: platform linux, python 3.9.12-final-0 -----------
Name                                                      Stmts   Miss  Cover
-----------------------------------------------------------------------------
speechbrain/__init__.py                                      15      0   100%
speechbrain/alignment/__init__.py                             0      0   100%
speechbrain/alignment/aligner.py                            380     61    84%
speechbrain/alignment/ctc_segmentation.py                   189     10    95%
speechbrain/core.py                                         424    155    63%
speechbrain/dataio/__init__.py                                7      0   100%
speechbrain/dataio/batch.py                                  99      8    92%
speechbrain/dataio/dataio.py                                279     50    82%
speechbrain/dataio/dataloader.py                            140     25    82%
speechbrain/dataio/dataset.py                               100      8    92%
speechbrain/dataio/encoder.py                               328     46    86%
speechbrain/dataio/iterators.py                              80     62    22%
speechbrain/dataio/legacy.py                                121     41    66%
speechbrain/dataio/preprocess.py                             22      4    82%
speechbrain/dataio/sampler.py                               224     61    73%
speechbrain/dataio/wer.py                                    63     54    14%
speechbrain/decoders/__init__.py                              2      0   100%
speechbrain/decoders/ctc.py                                 111     89    20%
speechbrain/decoders/seq2seq.py                             370     46    88%
speechbrain/decoders/transducer.py                          133     64    52%
speechbrain/lm/__init__.py                                    0      0   100%
speechbrain/lm/arpa.py                                       77      3    96%
speechbrain/lm/counting.py                                   37      4    89%
speechbrain/lm/ngram.py                                      36      1    97%
speechbrain/lobes/__init__.py                                 1      0   100%
speechbrain/lobes/augment.py                                154     55    64%
speechbrain/lobes/beamform_multimic.py                       20     14    30%
speechbrain/lobes/features.py                                96      9    91%
speechbrain/lobes/models/CRDNN.py                            52     12    77%
speechbrain/lobes/models/ContextNet.py                       83      3    96%
speechbrain/lobes/models/ECAPA_TDNN.py                      157      7    96%
speechbrain/lobes/models/ESPnetVGG.py                        20      0   100%
speechbrain/lobes/models/EnhanceResnet.py                    64      0   100%
speechbrain/lobes/models/HifiGAN.py                         321    146    55%
speechbrain/lobes/models/MetricGAN.py                        74     29    61%
speechbrain/lobes/models/MetricGAN_U.py                      63      0   100%
speechbrain/lobes/models/RNNLM.py                            32      0   100%
speechbrain/lobes/models/Tacotron2.py                       364     66    82%
speechbrain/lobes/models/VanillaNN.py                         8      0   100%
speechbrain/lobes/models/Xvector.py                          51      0   100%
speechbrain/lobes/models/__init__.py                          0      0   100%
speechbrain/lobes/models/conv_tasnet.py                     121      6    95%
speechbrain/lobes/models/convolution.py                      32      0   100%
speechbrain/lobes/models/dual_path.py                       357     55    85%
speechbrain/lobes/models/fairseq_wav2vec.py                  93     93     0%
speechbrain/lobes/models/g2p/__init__.py                      4      0   100%
speechbrain/lobes/models/g2p/dataio.py                      136    107    21%
speechbrain/lobes/models/g2p/homograph.py                   118     20    83%
speechbrain/lobes/models/g2p/model.py                       132    109    17%
speechbrain/lobes/models/huggingface_wav2vec.py             145     47    68%
speechbrain/lobes/models/resepformer.py                     180     21    88%
speechbrain/lobes/models/segan_model.py                     102     88    14%
speechbrain/lobes/models/transformer/Conformer.py           111      7    94%
speechbrain/lobes/models/transformer/Transformer.py         180     22    88%
speechbrain/lobes/models/transformer/TransformerASR.py       92     28    70%
speechbrain/lobes/models/transformer/TransformerLM.py        47      5    89%
speechbrain/lobes/models/transformer/TransformerSE.py        20      2    90%
speechbrain/lobes/models/transformer/TransformerST.py        81     60    26%
speechbrain/lobes/models/transformer/__init__.py              0      0   100%
speechbrain/lobes/models/wav2vec.py                         123     55    55%
speechbrain/nnet/CNN.py                                     417     56    87%
speechbrain/nnet/RNN.py                                     471     51    89%
speechbrain/nnet/__init__.py                                  8      0   100%
speechbrain/nnet/activations.py                              39      1    97%
speechbrain/nnet/attention.py                               234     44    81%
speechbrain/nnet/complex_networks/__init__.py                 0      0   100%
speechbrain/nnet/complex_networks/c_CNN.py                  130     23    82%
speechbrain/nnet/complex_networks/c_RNN.py                  374     67    82%
speechbrain/nnet/complex_networks/c_linear.py                26      0   100%
speechbrain/nnet/complex_networks/c_normalization.py        277     68    75%
speechbrain/nnet/complex_networks/c_ops.py                  108     40    63%
speechbrain/nnet/containers.py                              139     14    90%
speechbrain/nnet/dropout.py                                  15      0   100%
speechbrain/nnet/embedding.py                                24      0   100%
speechbrain/nnet/linear.py                                   27      1    96%
speechbrain/nnet/loss/__init__.py                             0      0   100%
speechbrain/nnet/loss/guidedattn_loss.py                     25      0   100%
speechbrain/nnet/loss/si_snr_loss.py                         20     16    20%
speechbrain/nnet/loss/stoi_loss.py                           81      1    99%
speechbrain/nnet/loss/transducer_loss.py                    136    136     0%
speechbrain/nnet/losses.py                                  323    112    65%
speechbrain/nnet/normalization.py                           142      6    96%
speechbrain/nnet/pooling.py                                 156     31    80%
speechbrain/nnet/quantisers.py                               47      2    96%
speechbrain/nnet/quaternion_networks/__init__.py              0      0   100%
speechbrain/nnet/quaternion_networks/q_CNN.py               150     25    83%
speechbrain/nnet/quaternion_networks/q_RNN.py               370     59    84%
speechbrain/nnet/quaternion_networks/q_linear.py             50     11    78%
speechbrain/nnet/quaternion_networks/q_normalization.py      44      4    91%
speechbrain/nnet/quaternion_networks/q_ops.py               229    122    47%
speechbrain/nnet/schedulers.py                              363    103    72%
speechbrain/nnet/transducer/__init__.py                       0      0   100%
speechbrain/nnet/transducer/transducer_joint.py              33      5    85%
speechbrain/pretrained/__init__.py                            1      0   100%
speechbrain/pretrained/fetching.py                           48      6    88%
speechbrain/pretrained/interfaces.py                        786    338    57%
speechbrain/pretrained/training.py                           33     28    15%
speechbrain/processing/NMF.py                                50      0   100%
speechbrain/processing/PLDA_LDA.py                          345     96    72%
speechbrain/processing/__init__.py                            0      0   100%
speechbrain/processing/decomposition.py                     102      8    92%
speechbrain/processing/diarization.py                       319    157    51%
speechbrain/processing/features.py                          359     75    79%
speechbrain/processing/multi_mic.py                         345      2    99%
speechbrain/processing/signal_processing.py                 166     39    77%
speechbrain/processing/speech_augmentation.py               386     34    91%
speechbrain/tokenizers/SentencePiece.py                     181     74    59%
speechbrain/tokenizers/__init__.py                            0      0   100%
speechbrain/utils/Accuracy.py                                24      3    88%
speechbrain/utils/DER.py                                     44     33    25%
speechbrain/utils/__init__.py                                 7      0   100%
speechbrain/utils/bleu.py                                    50      9    82%
speechbrain/utils/callchains.py                              28      5    82%
speechbrain/utils/check_HF_repo.py                           58     52    10%
speechbrain/utils/check_docstrings.py                        62      9    85%
speechbrain/utils/check_url.py                               50     40    20%
speechbrain/utils/check_yaml.py                             116     50    57%
speechbrain/utils/checkpoints.py                            294     37    87%
speechbrain/utils/data_pipeline.py                          181     13    93%
speechbrain/utils/data_utils.py                             197     61    69%
speechbrain/utils/depgraph.py                                82      1    99%
speechbrain/utils/distributed.py                             61     37    39%
speechbrain/utils/edit_distance.py                          180     50    72%
speechbrain/utils/epoch_loop.py                              55      5    91%
speechbrain/utils/hparams.py                                  2      1    50%
speechbrain/utils/hpopt.py                                  134     21    84%
speechbrain/utils/logger.py                                  73     24    67%
speechbrain/utils/metric_stats.py                           285     46    84%
speechbrain/utils/parameter_transfer.py                      87     17    80%
speechbrain/utils/profiling.py                              191     54    72%
speechbrain/utils/recipe_tests.py                           211    196     7%
speechbrain/utils/superpowers.py                             20      6    70%
speechbrain/utils/text_to_sequence.py                        77     22    71%
speechbrain/utils/torch_audio_backend.py                      9      2    78%
speechbrain/utils/train_logger.py                           150    113    25%
speechbrain/wordemb/__init__.py                               0      0   100%
speechbrain/wordemb/transformer.py                           90     67    26%
speechbrain/wordemb/util.py                                  11      0   100%
-----------------------------------------------------------------------------
TOTAL                                                     17279   4687    73%
```
