Audio Tokenizers
----------------

This folder contains code for creating and using discrete audio tokens. The files:

* `kmeans.py` - code for clustering continuous representations into discrete, an example
recipe can be found at `/recipes/LibriSpeech/quantization/train.py`, depends on `sklearn`.
* `speechtokenizer_interface.py` - code for generating discrete tokens using
[SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer), depends on `speechtokenizer` and `beartype`.
* `wavtokenizer_interface.py` - code for generating discrete tokens using
[WavTokenizer](https://github.com/Tomiinek/WavTokenizer), depends on `wavtokenizer`.
* `discrete_ssl.py` - code for extracting discrete audio tokens using pretrained SSL models (e.g. WavLM),
depends on `transformers`.

Here is a record of test setup and relevant results:

```bash
$ pip install scikit-learn==1.5.1 speechtokenizer==1.0.1 beartype==0.19.0 transformers==4.51.3 git+https://github.com/Tomiinek/WavTokenizer
$ pytest --cov=speechbrain/integrations/discrete/ --cov-context=test --doctest-modules speechbrain/integrations/audio_tokenizers/

=================== test session starts =======================
platform linux -- Python 3.11.11, pytest-7.4.0, pluggy-1.5.0
rootdir: /home/competerscience/Documents/Repositories/speechbrain
configfile: pytest.ini
plugins: anyio-4.8.0, hydra-core-1.3.2, cov-6.1.1, typeguard-4.4.1
collected 4 items

audio_tokenizers/discrete_ssl.py .
audio_tokenizers/kmeans.py .
audio_tokenizers/speechtok.py .
audio_tokenizers/wavtok.py .

===================== tests coverage =========================
_____ coverage: platform linux, python 3.11.11-final-0 _______

Name                                               Stmts   Miss  Cover
----------------------------------------------------------------------
audio_tokenizers/discrete_ssl.py                     100     12    88%
audio_tokenizers/kmeans.py                            51     10    80%
audio_tokenizers/speechtokenizer_interface.py         28      3    89%
audio_tokenizers/wavtokenizer_interface.py            33      5    85%
----------------------------------------------------------------------
TOTAL                                                212     30    86%

```
