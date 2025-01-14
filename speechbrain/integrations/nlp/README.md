NLP Tools
---------

This folder integrates NLP tools such as text embeddings, text-tagging models, text metrics, etc.
for a variety of languages. This is useful for e.g. embedding-based WER calculations amongst other things.

* [Flair](https://github.com/flairNLP/flair), a framework for e.g. bert embeddings, POS-tagging.
* [Spacy](https://github.com/explosion/spaCy), a framework for NLP pipelines, from tokenization to lemmatization and beyond.
* [SacreBLEU](https://github.com/mjpost/sacrebleu), a standardized implementation of the BLEU metric.

Here is a record of test setup and relevant results:

```bash
$ pip install flair==0.14.0 spacy==3.8.3 sacrebleu==2.4.3
$ pytest --cov=speechbrain/integrations/nlp/ --cov-context=test --doctest-modules speechbrain/integrations/nlp/

=================== test session starts =======================
platform linux -- Python 3.12.7, pytest-8.3.4, pluggy-1.5.0
plugins: hypothesis-6.112.0, cov-6.0.0, anyio-4.6.2.post1
collected 3 items

speechbrain/integrations/nlp/bleu.py .
speechbrain/integrations/nlp/flair_embeddings.py .
speechbrain/integrations/nlp/spacy_pipeline.py .

---------- coverage: platform linux, python 3.12.7-final-0 -----------
Name                                               Stmts   Miss  Cover
----------------------------------------------------------------------
speechbrain/integrations/nlp/__init__.py               3      0   100%
speechbrain/integrations/nlp/bleu.py                  51      9    82%
speechbrain/integrations/nlp/flair_embeddings.py      27      3    89%
speechbrain/integrations/nlp/flair_tagger.py          18      9    50%
speechbrain/integrations/nlp/spacy_pipeline.py        19      1    95%
----------------------------------------------------------------------
TOTAL                                                118     22    81%
```
