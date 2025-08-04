Alignment
---------

This folder contains code for doing speech alignment using the [CTC Segmentation library](https://github.com/lumaku/ctc-segmentation)

Here is a record of test setup and relevant results:

```bash
$ pip install ctc-segmentation==1.7.4 numpy<2.0
$ pytest --cov=speechbrain/integrations/alignment/ --cov-context=test --doctest-modules speechbrain/integrations/alignment/

=================== test session starts =======================
platform linux -- Python 3.11.11, pytest-7.4.0, pluggy-1.5.0
configfile: pytest.ini
plugins: anyio-4.8.0, hydra-core-1.3.2, cov-6.1.1, typeguard-4.4.1
collected 9 items

speechbrain/integrations/alignment/ctc_seg.py .
speechbrain/integrations/alignment/diarization.py ........

============================ tests coverage ===========================
__________ coverage: platform linux, python 3.11.11-final-0 ___________

Name                                                Stmts   Miss  Cover
-----------------------------------------------------------------------
speechbrain/integrations/alignment/ctc_seg.py         191     54    72%
speechbrain/integrations/alignment/diarization.py     317    133    58%
-----------------------------------------------------------------------
TOTAL                                                 508    187    63%

```
