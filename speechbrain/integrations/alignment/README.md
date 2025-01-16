Alignment
---------

This folder contains code for doing speech alignment using the [CTC Segmentation library](https://github.com/lumaku/ctc-segmentation)

Here is a record of test setup and relevant results:

```bash
$ pip install ctc-segmentation==1.7.4
$ pytest --cov=speechbrain/integrations/alignment/ --cov-context=test --doctest-modules speechbrain/integrations/alignment/

=================== test session starts =======================
platform linux -- Python 3.12.7, pytest-8.3.4, pluggy-1.5.0
plugins: hypothesis-6.112.0, cov-6.0.0, anyio-4.6.2.post1
collected 1 item

speechbrain/integrations/alignment/ctc_seg.py .

---------- coverage: platform linux, python 3.12.7-final-0 -----------
Name                                            Stmts   Miss  Cover
-------------------------------------------------------------------
speechbrain/integrations/alignment/ctc_seg.py     191    143    25%
```
