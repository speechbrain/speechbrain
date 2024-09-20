# Contributing

## Zen of Speechbrain
SpeechBrain is used for *research*, *academic*, *commercial*, *non-commercial* purposes, thus the code should be:

- **Simple:** Straightforward and easy to understand even by students, academics and non-professional programmers. Complex code, when it _must_ exist, should be especially well explained.

- **Readable:** Avoid abstract naming. Link to resources and references to help understand complex topics or implementations. Code style and formatting are automatically enforced.

- **Efficient**: Not _everything_ must be fast, but for what _should_ be, [profile and optimize it](https://speechbrain.readthedocs.io/en/develop/tutorials/advanced/profiling-and-benchmark.html). Operate on batches. Prefer tensor operations over Python-heavy constructs. Avoid CPU/GPU syncs.

- **Modular:** It should be easy to use any of the functionality from the toolkit. Break up functions/classes when it helps. Group functionality logically. Avoid unnecessary coupling.

- **Well documented:** Docs should be complete, easy to navigate and easy to discover. Consider [writing a tutorial](https://github.com/speechbrain/speechbrain/tree/develop/docs#tutorial-integration).

## Creating Pull Requests on GitHub

0. We use git and GitHub.
1. Fork the speechbrain repository (https://github.com/speechbrain/speechbrain)
on GitHub under your own account.
    (This creates a copy of SpeechBrain under your account, and GitHub
    knows where it came from, and we typically call this "upstream".)
2. Clone your own speechbrain repository.
    `git clone https://github.com/ <your-account> /speechbrain`
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

## Important code guidelines

We target a specific range of supported Python versions, which are tested via CI.

### Formatting & linting

Use `pre-commit run -a` to run formatting and linting, using tools like `black`
and `flake8` under the hood (see [`.pre-commit-config.yaml`](../.pre-commit-config.yaml)).
Some passes automatically fix your code, and some may require your intervention.

These checks are run and enforced on the CI.

### Running tests

We use [pytest](https://docs.pytest.org/en/latest/contents.html). Run unit tests
with `pytest tests`

Additionally, we have runnable doctests, though primarily these serve as
examples of the documented code. Run doctests with
`pytest --doctest-modules <file-or-directory>`

These checks are run and enforced on the CI.

### Adding dependencies

In general, we strive to have as few dependencies as possible. However, we will
debate dependencies on a case-by-case basis. We value easy installability via
pip.

In case the dependency is only needed for a specific recipe or specific niche
module, we suggest the extra tools pattern: don't add the dependency to general
requirements, but add it in the `extra-requirements.txt` file of that specific
recipe.

## Important documentation guidelines

In SpeechBrain, we plan to provide documentation at different levels:

-  **Docstrings**: For each class/function in the repository, there should be a header that properly describes its functionality, inputs, and outputs. It is also crucial to provide an example that shows how it can be used as a stand-alone function. We use [Numpy-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) docstrings. Consistent docstring style enables automatic API documentation. Also note the automatic doctests (see [here](#testing)).

-  **Comments**: We encourage developers to write self-documenting code, and use
proper comments where the implementation is surprising (to a Python-literate audience)
and where the implemented algorithm needs clarification.

-  **Website documentation**.  On the SpeechBrain website, you can find detailed documentation for each of the functionalities currently implemented in the toolkit.

-  **Tutorials**:  Tutorials are a good way to familiarize yourself with SpeechBrain with interactive codes and explanations.


## Additional reading

- [Development tools](devtools.md)
- [What testing coverage approaches are needed?](coverage.md)

### Internal contributors

- [Releasing a new version](newversion.md)
- [Reviewing code](codereview.md)