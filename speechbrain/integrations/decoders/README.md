Decoders
--------

In ASR, decoding is often done with the help of an n-gram language model,
and we provide integration with a fast implementation through
[KenLM](https://github.com/kpu/kenlm).

Here is a record of test setup and relevant results:

```bash
$ pip install kenlm==0.2.0
$ pytest --cov=speechbrain/integrations/decoders/ --cov-context=test --doctest-modules speechbrain/integrations/decoders/

=================== test session starts =======================
platform linux -- Python 3.12.7, pytest-8.3.4, pluggy-1.5.0
plugins: hypothesis-6.112.0, cov-6.0.0, anyio-4.6.2.post1
collected 1 item

speechbrain/integrations/decoders/kenlm_scorer.py .

---------- coverage: platform linux, python 3.12.7-final-0 -----------
Name                                                Stmts   Miss  Cover
-----------------------------------------------------------------------
speechbrain/integrations/decoders/kenlm_scorer.py     100     62    38%
```
