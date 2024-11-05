# What testing coverage approaches are needed?

1. Dependencies: version control (check commit ID dates)
  <br/> see: [requirements.txt](https://github.com/speechbrain/speechbrain/blob/develop/requirements.txt)
  <br/> run: `find *txt . | grep extra`
2. Docstring tests: commented function signatures <br/>_(of functions intended for outer calls)_
3. [Unittests](https://github.com/speechbrain/speechbrain/tree/develop/tests/unittests) per function-critical code block
4. [Integration tests](https://github.com/speechbrain/speechbrain/tree/develop/tests/integration) for vanilla experiments to cover use-cases on a generic task basis
5. Regression testing: standing interfaces & their refactoring
6. Linters for automated style checks & corrections of python & yaml code

## Where to get things done?

1. Raise your questions & engage in [Discussions](https://github.com/speechbrain/speechbrain/discussions)
2. Report a bug or request a feature, open [Issues](https://github.com/speechbrain/speechbrain/issues/new/choose)
3. Contribute [Pull requests](https://github.com/speechbrain/speechbrain/pulls)
4. Release pretrained models through SpeechBrain
   <br/> e.g. registering linking HuggingFace account to SpeechBrain for hosting your model card

## GitHub workflow: strategy by configuration

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

## PyTest for reporting code coverage rates

How to know test coverage changes of Open PRs to be merged?
<br/>_(snippet for cpu-only)_
```
# Example: install more dependencies to avoid ignoring modules
sudo apt install -y libsndfile1
pip install ctc_segmentation

# install coverage
pip install pytest-cov

# run the test (w/ duration reporting)
pytest --durations=0 --cov=speechbrain --cov-context=test --doctest-modules speechbrain tests --ignore=speechbrain/nnet/loss/transducer_loss.py
```
Example: _After collecting 459 testing items, 4481/16782 statements are reported "missing" (73% coverage)._

YET—python code of the core modules is not all to be covered; thus far, only, consistency is ensured..

---

Further reading:
<br/> pytest & coverage - https://breadcrumbscollector.tech/how-to-use-code-coverage-in-python-with-pytest/ (pointer by @Adel-Moumen)

---

```
pytest --durations=0 --cov=speechbrain --cov-context=test --doctest-modules speechbrain tests --ignore=speechbrain/nnet/loss/transducer_loss.py

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