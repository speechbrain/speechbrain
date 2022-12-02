# Ideas for future testing

As mentioned above, this document portrays a point in time.
Naturally for OpenSource, there are so many pathways for CI/CD:
* Tutorial testing
* How to know all GDrive & HF repos are referenced in tests/recipes?
* Pre-trained interfaces & data caching (models & audios)
* Targeted testing tools & workflows: test only what is impacted by a change (and not covered elsewhere already)
* Suggestion tools for default/optional args when using hyperpyyaml
* Community-driven 'full' recipe checks from dataio to few-epoch outcomes
* PR testing automation for a self-service SB community (move beyond checklists -> discuss ideas)
* readthedocs as vivid part in PR drafting; testing & reviewing
* restructure recipe yamls &/or testing scripts for easier parameter override (see `recipes/TIMIT/ASR/seq2seq_knowledge_distillation`) 
* speed-up recipe tests through yaml rewritings (availing necessary overrides)
* Coverage tables (e.g. for README): `Python x PyTorch` versions
