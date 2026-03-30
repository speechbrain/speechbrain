Numba
-----

This package contains modules that rely on [Numba](https://numba.pydata.org/)
for CUDA-accelerated computations, such as the Transducer loss.

```bash
$ pip install numba
$ pytest --cov=speechbrain/integrations/numba/ --cov-context=test --doctest-modules speechbrain/integrations/numba/
========================================================================= test session starts ==========================================================================
platform linux -- Python 3.12.11, pytest-9.0.2, pluggy-1.6.0
plugins: cov-7.0.0, anyio-4.12.1
collected 1 item

speechbrain/integrations/numba/transducer_loss.py .

___________________________________________________________ coverage: platform linux, python 3.12.11-final-0 ___________________________________________________________

Name                                                Stmts   Miss  Cover
-----------------------------------------------------------------------
speechbrain/integrations/numba/__init__.py              9      5    44%
speechbrain/integrations/numba/transducer_loss.py     121     67    45%
-----------------------------------------------------------------------
TOTAL                                                 130     72    45%
```
