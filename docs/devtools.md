# Development tools

## Linting/formatting/testing

### flake8
- A bit like pycodestyle: make sure the codestyle is according to guidelines.
- Compatible with black, in fact, current flake8 config directly taken from black
- Code compliance can be tested simply with: `flake8 <file-or-directory>`
- You can bypass flake8 for a line with `# noqa: <QA-CODE> E.G. # noqa: E731 to allow lambda assignment`

### pre-commit
- Python tool which takes a configuration file (.pre-commit-config.yaml) and installs the git commit hooks specified in it.
- Git commit hooks are local so all who want to use them need to install them separately. This is done by: `pre-commit install`
- The tool can also install pre-push hooks. This is done separately with: `pre-commit install --hook-type pre-push --config .pre-push-config.yaml`

### the git pre-commit hooks
- Automatically run black
- Automatically fix trailing whitespace, end of file, sort requirements.txt
- Check that no large (>512kb) files are added by accident
- Automatically run flake8
- Automatically run cspell
- NOTE: If the hooks fix something (e.g. trailing whitespace or reformat with black), these changes are not automatically added and committed. You’ll have to add the fixed files again and run the commit again. I guess this is a safeguard: don’t blindly accept changes from git hooks.
- NOTE2: The hooks are only run on the files you git added to the commit. This is in contrast to the CI pipeline, which always tests everything.
- NOTE3: If a word is flagged as a spelling error but it should be kept, you can add the word to `.dict-speechbrain.txt`

### the git pre-push hooks
- Black and flake8 as checks on the whole repo
- Unit-tests and doctests run on the whole repo
- These hooks can only be run in the full environment, so if you install these, you’ll need to e.g. activate virtualenv before pushing.

### pytest doctests
- This is not an additional dependency, but just that doctests are now run with pytest. Use: `pytest --doctest-modules <file-or-directory>`
- Thus you may use some pytest features in docstring examples. Most notably IMO: `tmpdir = getfixture('tmpdir')` which makes a temp dir and gives you a path to it, without needing a `with tempfile.TemporaryDirectory() as tmpdir:`

## Continuous integration

### What is CI?
- loose term for a tight merge schedule
- typically assisted by automated testing and code review tools + practices

### CI / CD Pipelines
- GitHub Actions (and also available as a third-party solution) feature, which automatically runs basically anything in reaction to git events.
- The CI pipeline is triggered by pull requests.
- Runs in a Ubuntu environment provided by GitHub
- GitHub offers a limited amount of CI pipeline minutes for free.
- CD stands for continuous deployment, check out the "Releasing a new version" section.

### Our test suite
- Code linters are run. This means black and flake8. These are run on everything in speechbrain (the library directory), everything in recipes and everything in tests.
- Note that black will only error out if it would change a file here, but won’t reformat anything at this stage. You’ll have to run black on your code and push a new commit. The black commit hook helps avoid these errors.
- All unit-tests and doctests are run. You can check that these pass by running them yourself before pushing, with `pytest tests`  and `pytest --doctest-modules speechbrain`
- Integration tests (minimal examples). The minimal examples serve both to
  illustrate basic tasks and experiment running, but also as integration tests
  for the toolkit. For this purpose, any file which is prefixed with
  `example_` gets collected by pytest, and we add a short `test_` function at
  the end of the minimal examples.
- Currently, these are not run: docstring format tests (this should be added once the docstring conversion is done).
- If all tests pass, the whole pipeline takes a couple of minutes.