
# Quick installation

SpeechBrain is constantly evolving. New features, tutorials, and documentation will appear over time.
SpeechBrain can be installed via PyPI to rapidly use the standard library.
Moreover,  a local installation can be used by those users that what to run experiments and modify/customize the toolkit.

SpeechBrain supports both CPU and GPU computations. For most all the recipes, however, a GPU is necessary during training.
Please note that CUDA must be properly installed to use GPUs.


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
cd speechbrain
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
