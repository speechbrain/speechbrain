# Contributing

The goal is to write a set of libraries that process audio and speech in several ways. It is crucial to write a set of homogeneous libraries that are all compliant with the guidelines described in the following sub-sections.

## Zen of Speechbrain
SpeechBrain could be used for *research*, *academic*, *commercial*, *non-commercial* purposes. Ideally, the code should have the following features:

- **Simple:**  the code must be easy to understand even by students or by users that are not professional programmers or speech researchers. Try to design your code such that it can be easily read. Given alternatives with the same level of performance, code the simplest one. (the most explicit and straightforward manner is preferred)

- **Readable:** SpeechBrain mostly adopts the code style conventions in PEP8. The code written by the users must be compliant with that. We test code style with `flake8`

- **Efficient**: The code should be as efficient as possible. When possible, users should maximize the use of pytorch native operations.  Remember that in generally very convenient to process in parallel multiple signals rather than processing them one by one (e.g try to use *batch_size > 1* when possible). Test the code carefully with your favorite profiler (e.g, torch.utils.bottleneck https://pytorch.org/docs/stable/bottleneck.html ) to make sure there are no bottlenecks in your code.  Since we are not working in *c++* directly, the speed can be an issue. Despite that, our goal is to make SpeechBrain as fast as possible.

- **Modular:** Write your code such that it is very modular and fits well with the other functionalities of the toolkit. The idea is to develop a bunch of models that can be naturally interconnected with each other.

- **Well documented:**  Given the goals of SpeechBrain, writing rich and good documentation is a crucial step.

## How to get your code in SpeechBrain

Practically, development goes as follows:

0. We use git and GitHub.
1. Fork the speechbrain repository (https://github.com/speechbrain/speechbrain)
on GitHub under your own account.
    (This creates a copy of SpeechBrain under your account, and GitHub
    knows where it came from, and we typically call this "upstream".)
2. Clone your own speechbrain repository.
    `git clone https://github.com/<your-account>/speechbrain`
    (This downloads the git repository to your machine, git knows where
    it came from, and calls it "origin".)
3. Create a branch for each specific feature you are developing.
    `git checkout -b your-branch-name`
4. Make + commit changes.
    `git add files-you-changed ...`
    `git commit -m "Short message about what you did"`
5. Push the branch to your GitHub repository.
    `git push origin your-branch-name`
6. Navigate to GitHub, and create a pull request from your branch to the upstream
repository speechbrain/speechbrain, to the "develop" branch.
7. The Pull Request (PR) appears on the upstream repository. Discuss your contribution
there. If you push more changes to your branch on GitHub (on your repository), they are
added to the PR.
8. When the reviewer is satisfied that the code improves repository quality, they can merge.

Note that CI tests will be run when you create a PR. If you want to be sure that your
code will not fail these tests, we have set up pre-commit hooks that you can install.
See the section on pre-commit.

These will automatically check the code when you commit and when you push.

## Python
### Version
SpeechBrain targets Python >= 3.8.

### Formatting
To settle code formatting, SpeechBrain adopts the [black](https://black.readthedocs.io/en/stable/) code formatter. Before submitting  pull requests, please run the black formatter on your code.

In addition, we use [flake8](https://flake8.pycqa.org/en/latest/) to test code
style. Black as a tool does not enforce everything that flake8 tests.

You can run the formatter with: `black <file-or-directory>`. Similarly the
flake8 tests can be run with `flake8 <file-or-directory>`.

### Adding dependencies
In general, we strive to have as few dependencies as possible. However, we will
debate dependencies on a case-by-case basis. We value easy installability via
pip.

In case the dependency is only needed for a specific recipe or specific niche
module, we suggest the extra tools pattern: don't add the dependency to general
requirements, but add it in the extra-requirement.txt file of the specific recipe.

### Testing
We are adopting unit tests using
[pytest](https://docs.pytest.org/en/latest/contents.html).
Run unit tests with `pytest tests`

Additionally, we have runnable doctests, though primarily these serve as
examples of the documented code. Run doctests with
`pytest --doctest-modules <file-or-directory>`

## Documentation
In SpeechBrain, we plan to provide documentation at different levels:

-  **Docstrings**: For each class/function in the repository, there should be a header that properly describes its functionality, inputs, and outputs. It is also crucial to provide an example that shows how it can be used as a stand-alone function. We use [Numpy-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) docstrings. Consistent docstring style enables automatic API documentation. Also note the automatic doctests (see [here](#testing).

-  **Comments**: We encourage developers to write self-documenting code, and use
proper comments where the implementation is surprising (to a Python-literate audience)
and where the implemented algorithm needs clarification.

-  **Website documentation**.  On the SpeechBrain website, you can find detailed documentation for each of the functionalities currently implemented in the toolkit.

-  **Tutorials**:  Tutorials are a good way to familiarize yourself with SpeechBrain with interactive codes and explanations.

## Development tools

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
- NOTE: If the hooks fix something (e.g. trailing whitespace or reformat with black), these changes are not automatically added and committed. You’ll have to add the fixed files again and run the commit again. I guess this is a safeguard: don’t blindly accept changes from git hooks.
- NOTE2: The hooks are only run on the files you git added to the commit. This is in contrast to the CI pipeline, which always tests everything.

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

## Pull Request review guide

This is not a comprehensive code review guide, but some rough guidelines to unify the general review practices across this project.

Firstly, let the review take some time. Try to read every line that was added,
if possible. Try also to run some tests. Read the surrounding context of the code if needed to understand
the changes introduced. Possibly ask for clarifications if you don't understand.
If the pull request changes are hard to understand, maybe that's a sign that
the code is not clear enough yet. However, don't nitpick every detail.

Secondly, focus on the major things first, and only then move on to smaller,
things. Level of importance:
- Immediate deal breakers (code does the wrong thing, or feature shouldn't be added etc.)
- Things to fix before merging (Add more documentation, reduce complexity, etc.)
- More subjective things could be changed if the author also agrees with you.

Thirdly, approve the pull request only once you believe the changes "improve overall code health" as attested to [here](https://google.github.io/eng-practices/review/reviewer/standard.html).
However, this also means the pull request does not have to be perfect. Some features are best implemented incrementally over many pull requests, and you should be more concerned with making sure that the changes introduced lend themselves to painless further improvements.

Fourthly, use the tools that GitHub has: comment on specific code lines, suggest edits, and once everyone involved has agreed that the PR is ready to merge, merge the request and delete the feature branch.

Fifthly, the code review is a place for professional constructive criticism,
a nice strategy to show (and validate) that you understand what the PR is really
doing is to provide some affirmative comments on its strengths.

## Releasing a new version

Here are a few guidelines for when and how to release a new version.
To begin with, as hinted in the "Continuous Integration" section, we would like to follow a
pretty tight release schedule, known as "Continuous Deployment". For us, this means a new
version should be released roughly once a week.

As for how to name the released version, we try to follow semantic versioning for this. More details
can be found at [semver.org](http://semver.org). As it applies to SpeechBrain, some examples
of what this would likely mean:
 * Changes to the Brain class or other core elements often warrant a major version bump (e.g. 1.5.3 -> 2.0.0)
 * Added classes or features warrant a minor version bump. Most weekly updates should fall into this.
 * Patch version bumps should happen only for bug fixes.

When releasing a new version, there are a few user-initiated action that need to occur.
 1. On the `develop` branch, update `speechbrain/version.txt` to say the new version:
    X.Y.Z
 2. Merge the `develop` branch into the `main` branch:
    git checkout main
    git merge develop
 3. Push the `main` branch to github:
    git push
 4. Tag the `main` branch with the new version:
    git tag vX.Y.Z
 5. Push the new tag to github:
    git push --tags

This kicks off an automatic action that creates a draft release with release notes.
Review the notes to make sure they make sense and remove commits that aren't important.
You can then publish the release to make it public.
Publishing a new release kicks off a series of automatic tools, listed below:

 * The `main` branch is checked out and used for building a python package.
 * The built package is uploaded to PyPI and the release is published there.
 * Read the Docs uses Webhooks to get notified when a new version is published.
   Read the Docs then builds the documentation and publishes the new version.

Maintainers of relevant accounts:
 * Mirco Ravanelli maintains the GitHub and PyPI accounts
 * Titouan Parcollet maintains the website at [speechbrain.github.io](speechbrain.github.io)
   as well as accounts at Read the Docs and Discourse
