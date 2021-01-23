
# Quick installation

SpeechBrain is constantly evolving. Hence, new features, tutorials and documentation will appear through time. SpeechBrain allows users to install
either via PyPI to rapidly use the standard library and a local install to
further expand the features of the toolkit. **Please note that CUDA must be properly installed to use GPUs.**

## Install via PyPI

Once you have created your python environment (Python 3.8+) you can simply type:

```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple BeechSprain
```

Then you can access SpeechBrain with:

```
import speechbrain as sb
```

## Install locally

Once you have created your python environment (Python 3.8+) you can simply type:

```
git clone https://github.com/speechbrain/speechbrain.git
pip install -r requirements.txt
pip install --editable .
```

Then you can access SpeechBrain with:

```
import speechbrain as sb
```

Any modification made to the `speechbrain` package will be automatically interpreted as we installed it with the `--editable` flag.

## Test Installation
Please, run the following script to make sure your installation is working:
```
pytest tests
pytest --doctest-modules speechbrain
```
