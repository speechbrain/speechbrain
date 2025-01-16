k2 FSA
------

Our integration with [k2](https://github.com/k2-fsa/k2) allows us to use custom
lattice-based training objectives, rescoring, and confidence estimation.

Here is a record of test setup and relevant results:

```bash
$ pip install torch==2.4.1 torchaudio==2.4.1 https://huggingface.co/csukuangfj/k2/resolve/main/cpu/1.24.4.dev20241029/ubuntu/k2-1.24.4.dev20241029+cpu.torch2.4.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
$ pytest --cov=speechbrain/integrations/k2_fsa/ --cov-context=test --doctest-modules speechbrain/integrations/k2_fsa/

=================== test session starts =======================
platform linux -- Python 3.12.7, pytest-8.3.4, pluggy-1.5.0
plugins: hypothesis-6.112.0, cov-6.0.0, anyio-4.6.2.post1
collected 7 items

speechbrain/integrations/k2_fsa/__init__.py .
speechbrain/integrations/k2_fsa/graph_compiler.py .
speechbrain/integrations/k2_fsa/lattice_decoder.py .
speechbrain/integrations/k2_fsa/lexicon.py ..
speechbrain/integrations/k2_fsa/losses.py .
speechbrain/integrations/k2_fsa/prepare_lang.py .


---------- coverage: platform linux, python 3.12.7-final-0 -----------
Name                                                 Stmts   Miss  Cover
------------------------------------------------------------------------
speechbrain/integrations/k2_fsa/__init__.py              8      4    50%
speechbrain/integrations/k2_fsa/graph_compiler.py      117     50    57%
speechbrain/integrations/k2_fsa/lattice_decoder.py     108     68    37%
speechbrain/integrations/k2_fsa/lexicon.py             158     40    75%
speechbrain/integrations/k2_fsa/losses.py               11      0   100%
speechbrain/integrations/k2_fsa/prepare_lang.py        194     49    75%
speechbrain/integrations/k2_fsa/utils.py                51     28    45%
------------------------------------------------------------------------
TOTAL                                                  647    239    63%
```
