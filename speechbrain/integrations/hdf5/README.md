HDF5 Feature Caching
--------------------

This integration provides a new backend for feature caching based on HDF5,
a high-performance data software library for large datasets.

Here is a record of test setup and relevant results:

```bash
$ pip install h5py==3.12.1
$ pytest --cov=speechbrain/integrations/hdf5/ --cov-context=test --doctest-modules speechbrain/integrations/hdf5/

================================== test session starts ==================================
platform linux -- Python 3.11.11, pytest-7.4.0, pluggy-1.5.0
configfile: pytest.ini
plugins: hydra-core-1.3.2, typeguard-2.13.3, torchtyping-0.1.5, cov-6.1.1, anyio-4.10.0
collected 1 item

speechbrain/integrations/hdf5/cached_item.py .                                     [100%]

==================================== tests coverage =====================================
___________________ coverage: platform linux, python 3.11.11-final-0 ____________________

Name                                                Stmts   Miss  Cover
-----------------------------------------------------------------------
speechbrain/integrations/hdf5/cached_item.py           25      4    84%
-----------------------------------------------------------------------
TOTAL                                                  25      4    84%
=================================== 1 passed in 2.38s ===================================
```
