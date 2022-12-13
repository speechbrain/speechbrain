# What coverage approaches are needed?

1. Dependencies: version control (check commit ID dates)
  <br/> see: [requirements.txt](https://github.com/speechbrain/speechbrain/blob/develop/requirements.txt)
  <br/> run: `find *txt . | grep extra`
2. Docstring tests: commented function signatures <br/>_(of functions intended for outer calls)_
3. [Unittests](https://github.com/speechbrain/speechbrain/tree/develop/tests/unittests) per function-critical code block
4. [Integration tests](https://github.com/speechbrain/speechbrain/tree/develop/tests/integration) for vanilla experiments to cover use-cases on a generic task basis
5. Advanced testing: standing interfaces & their refactoring
6. Linters for automated style checks & corrections of python & yaml code

## Where to get things done?

1. Raise your questions & engage in [Discussions](https://github.com/speechbrain/speechbrain/discussions)
2. Report a bug or request a feature, open [Issues](https://github.com/speechbrain/speechbrain/issues/new/choose)
3. Contribute [Pull requests](https://github.com/speechbrain/speechbrain/pulls)
4. Release pretrained models through SpeechBrain
   <br/> e.g. registering linking HuggingFace account to SpeechBrain for hosting your model card

## github workflow: strategy by configuration

API configurations are located at [.github/workflows](https://github.com/speechbrain/speechbrain/tree/develop/.github/workflows)
<br/>_(all creating a one-time ubuntu-latest environment)_

---

Info: although our PyTorch requirements are
```
torch>=1.9.0
torchaudio>=0.9.0
```
our tests cover one PyTorch version only, _the latest_.


### [pre-commit.yml](https://github.com/speechbrain/speechbrain/blob/develop/.github/workflows/pre-commit.yml)
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

### [pythonapp.yml](https://github.com/speechbrain/speechbrain/blob/develop/.github/workflows/pythonapp.yml)
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
  * a. hook: Consistency tests with pytest
    <br/> `pytest tests/consistency`
  * b. hook: Unittests with pytest
    <br/> `pytest tests/unittests`
  * c. hook: Doctests with pytest
    <br/> `pytest --doctest-modules speechbrain`
  * d. hook: Integration tests with pytest
    <br/> `pytest tests/integration`

### [verify-docs-gen.yml](https://github.com/speechbrain/speechbrain/blob/develop/.github/workflows/verify-docs-gen.yml) [I.2.a]
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

### [newtag.yml](https://github.com/speechbrain/speechbrain/blob/develop/.github/workflows/newtag.yml)
   > Draft release when pushing new tag
* tagging of `develop` branch commit ID
* before
  * follow through [tests/PRE-RELEASE-TESTS.md](https://github.com/speechbrain/speechbrain/blob/develop/tests/PRE-RELEASE-TESTS.md)
    * set-up fresh environment
    * run `pytest`
    * a. hook: [tests/.run-load-yaml-tests.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-load-yaml-tests.sh)
    * b. hook: [tests/.run-recipe-tests.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-recipe-tests.sh)
    * c. hook: [tests/.run-HF-checks.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-HF-checks.sh)
    * d. hook: [ests/.run-url-checks.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-url-checks.sh)
  * update of [speechbrain/version.txt](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/version.txt) to the next
* action: draft push to `main` branch
  <br/> implies pre-push hook, see: [.pre-push-config.yaml](https://github.com/speechbrain/speechbrain/blob/develop/.pre-push-config.yaml) with hooks to:
  * e. [tests/.run-linters.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-linters.sh)
  * f. [tests/.run-unittests.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-unittests.sh)
  * g. [tests/.run-doctests.sh](https://github.com/speechbrain/speechbrain/blob/develop/tests/.run-doctests.sh)

### [release.yml](https://github.com/speechbrain/speechbrain/blob/develop/.github/workflows/release.yml)
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
    * points to https://speechbrain.github.io/

The versions of tools used/hooked in these checks are controlled via [lint-requirements.txt](https://github.com/speechbrain/speechbrain/blob/develop/lint-requirements.txt), a nested dependency in [requirements.txt](https://github.com/speechbrain/speechbrain/blob/develop/requirements.txt).
With major version releases of SpeechBrain, the versions of each hook should be updated—alongside requirement consistency in source, testing & builds incl. running spell-checking.

_Note: [PyTorch statement](https://pytorch.org/get-started/locally/) on Python versions (as of 2022-11-09)_
> _It is recommended that you use Python 3.6, 3.7 or 3.8_
