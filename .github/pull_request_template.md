# Contribution in a nutshell
Hey, this could help our community 🌱

# Scope
* [ ] I want to get done ...
* [ ] ... and hope to also achieve ...

# Notes for reviewing (optional)
This change has these implication which might need attention over here; —how should we tackle this?

# Pre-review
* [ ] (if applicable) add an `extra_requirements.txt` file
* [ ] (if applicable) add database preparation scripts & use symlinks for nested folders (to the level of task READMEs)
* [ ] (if applicable) add a recipe test entry in the depending CSV file under: tests/recipes
* [ ] create a fresh testing environment (install SpeechBrain from cloned repo branch of this PR)
* [ ] (if applicable) run a recipe test for each yaml/your recipe dataset
* [ ] check function comments: are there docstrings w/ arguments & returns? If you're not the verbose type, put a comment every three lines of code (better: every line)
* [ ] use CI locally: `pre-commit run -a` to check linters; run `pytest tests/consistency`
* [ ] (optional) run `tests/.run-doctests.sh` & `tests/.run-unittests.sh`
* [ ] exhausted patience before clicking « Ready for review » in the merge box 🍄

---

Note: when merged, we desire to include your PR title in our contributions list, check out one of our past version releases
—https://github.com/speechbrain/speechbrain/releases/tag/v0.5.14

Tip: below, on the « Create Pull Request » use the drop-down to select: « Create Draft Pull Request » – your PR will be in draft mode until you declare it « Ready for review »

