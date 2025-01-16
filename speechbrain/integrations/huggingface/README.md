Huggingface
-----------

In many cases, PyTorch is well-integrated enough that one can use models from
[HuggingFace](https://huggingface.co/) without adding any code to SpeechBrain,
but in some cases, we provide a wrapper to better match SpeechBrain style and
provide utility functions for things like freezing / thawing parts of a model,
or other such quality-of-life stuff.

Here is a record of test setup and relevant results:

```bash
$ pip install transformers==4.47.1
$ pytest --cov=speechbrain/integrations/huggingface/ --cov-context=test --doctest-modules speechbrain/integrations/huggingface/

=================== test session starts =======================
platform linux -- Python 3.12.7, pytest-8.3.4, pluggy-1.5.0
plugins: hypothesis-6.112.0, cov-6.0.0, anyio-4.6.2.post1
collected 20 items

speechbrain/integrations/huggingface/discrete_ssl.py .
speechbrain/integrations/huggingface/encodec.py .
speechbrain/integrations/huggingface/gpt.py .
speechbrain/integrations/huggingface/hubert.py .
speechbrain/integrations/huggingface/huggingface.py .
speechbrain/integrations/huggingface/labse.py .
speechbrain/integrations/huggingface/llama2.py .
speechbrain/integrations/huggingface/mbart.py .
speechbrain/integrations/huggingface/mert.py .
speechbrain/integrations/huggingface/mimi.py .
speechbrain/integrations/huggingface/nllb.py .
speechbrain/integrations/huggingface/textencoder.py .
speechbrain/integrations/huggingface/vocos.py .
speechbrain/integrations/huggingface/wav2vec2.py ..
speechbrain/integrations/huggingface/wavlm.py .
speechbrain/integrations/huggingface/weighted_ssl.py .
speechbrain/integrations/huggingface/whisper.py .
speechbrain/integrations/huggingface/wordemb/transformer.py .
speechbrain/integrations/huggingface/wordemb/util.py .


---------- coverage: platform linux, python 3.12.7-final-0 -----------
Name                                                          Stmts   Miss  Cover
---------------------------------------------------------------------------------
speechbrain/integrations/huggingface/__init__.py                 17      5    71%
speechbrain/integrations/huggingface/discrete_ssl.py            100     12    88%
speechbrain/integrations/huggingface/encodec.py                 108      8    93%
speechbrain/integrations/huggingface/gpt.py                      30      9    70%
speechbrain/integrations/huggingface/hubert.py                    6      0   100%
speechbrain/integrations/huggingface/huggingface.py             119     40    66%
speechbrain/integrations/huggingface/labse.py                    30      7    77%
speechbrain/integrations/huggingface/llama2.py                  113     66    42%
speechbrain/integrations/huggingface/mbart.py                    49     11    78%
speechbrain/integrations/huggingface/mert.py                      6      0   100%
speechbrain/integrations/huggingface/mimi.py                     38      3    92%
speechbrain/integrations/huggingface/nllb.py                      6      0   100%
speechbrain/integrations/huggingface/textencoder.py              22      5    77%
speechbrain/integrations/huggingface/vocos.py                    46      4    91%
speechbrain/integrations/huggingface/wav2vec2.py                 69     15    78%
speechbrain/integrations/huggingface/wavlm.py                     6      0   100%
speechbrain/integrations/huggingface/weighted_ssl.py             29      3    90%
speechbrain/integrations/huggingface/whisper.py                 196     78    60%
speechbrain/integrations/huggingface/wordemb/__init__.py          0      0   100%
speechbrain/integrations/huggingface/wordemb/transformer.py      90     67    26%
speechbrain/integrations/huggingface/wordemb/util.py             11      0   100%
---------------------------------------------------------------------------------
TOTAL                                                          1091    333    69%
```
