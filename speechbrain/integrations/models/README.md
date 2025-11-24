Models
------

This folder integrates models with code existing in stand-alone repos (not in SpeechBrain or Huggingface).

* [SGMSE](https://github.com/sp-uhh/sgmse), diffusion-based generative models of speech enhancement.

Here is a record of test setup and relevant results:

```bash
$ pip install git+https://github.com/sp-uhh/sgmse.git@main#egg=sgmse
$ pytest --cov=speechbrain/integrations/models/ --cov-context=test --doctest-modules speechbrain/integrations/models/
================ test session starts ==============================
platform linux -- Python 3.11.11, pytest-7.4.0, pluggy-1.5.0
plugins: anyio-4.8.0, hydra-core-1.3.2, typeguard-2.13.3, torchtyping-0.1.5, cov-6.1.1
collected 1 item

speechbrain/integrations/models/sgmse_plus.py .

========================= tests coverage ==========================
__________ coverage: platform linux, python 3.11.11-final-0 _______

Name                                            Stmts   Miss  Cover
-------------------------------------------------------------------
speechbrain/integrations/models/sgmse_plus.py     202    127    37%
-------------------------------------------------------------------
TOTAL                                             202    127    37%
```
