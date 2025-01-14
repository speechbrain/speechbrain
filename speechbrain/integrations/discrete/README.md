Discrete
--------

This folder contains code for creating and using discrete audio tokens. The three files:

* `kmeans.py` - code for clustering continuous representations into discrete, an example
recipe can be found at `/recipes/LibriSpeech/quantization/train.py`, depends on `sklearn`.
* `speechtok.py` - code for generating discrete tokens using
[SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer), depends on `speechtokenizer` and `beartype`.
* `wavtok.py` - code for generating discrete tokens using
[WavTokenizer](https://github.com/Tomiinek/WavTokenizer), depends on `wavtokenizer`.

Here is a record of test setup and relevant results:

```bash
$ pip install scikit-learn==1.5.1 speechtokenizer==1.0.1 beartype==0.19.0 wavtokenizer==1.0.0
$ pytest --cov=speechbrain/integrations/discrete/ --cov-context=test --doctest-modules speechbrain/integrations/discrete/

=================== test session starts =======================
platform linux -- Python 3.12.7, pytest-8.3.4, pluggy-1.5.0
plugins: hypothesis-6.112.0, cov-6.0.0, anyio-4.6.2.post1
collected 3 items

speechbrain/integrations/discrete/kmeans.py .
speechbrain/integrations/discrete/speechtok.py .
speechbrain/integrations/discrete/wavtok.py .

---------- coverage: platform linux, python 3.12.7-final-0 -----------
Name                                             Stmts   Miss  Cover
--------------------------------------------------------------------
speechbrain/integrations/discrete/kmeans.py         51     10    80%
speechbrain/integrations/discrete/speechtok.py      27      3    89%
speechbrain/integrations/discrete/wavtok.py         32      5    84%
--------------------------------------------------------------------
TOTAL                                              110     18    84%
```
