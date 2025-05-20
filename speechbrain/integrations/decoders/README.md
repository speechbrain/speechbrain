Decoders
--------

In ASR, decoding is often done with the help of an n-gram language model,
and we provide integration with a fast implementation through
[KenLM](https://github.com/kpu/kenlm).

Here is a record of test setup and relevant results:

```bash
$ pip install kenlm==0.3.0 pygtrie==2.5.0
$ pytest --cov=speechbrain/integrations/decoders/ --cov-context=test --doctest-modules speechbrain/integrations/decoders/

=================== test session starts =======================
platform linux -- Python 3.11.11, pytest-7.4.0, pluggy-1.5.0
rootdir: /home/competerscience/Documents/Repositories/speechbrain
configfile: pytest.ini
plugins: anyio-4.8.0, hydra-core-1.3.2, cov-6.1.1, typeguard-4.4.1
collected 2 items

speechbrain/integrations/decoders/kenlm_scorer.py ..

====================== test coverage ==========================
_______ coverage: platform linux, python 3.11.11-final-0 ______

Name                                                Stmts   Miss  Cover
-----------------------------------------------------------------------
speechbrain/integrations/decoders/kenlm_scorer.py     100     29    71%

```
