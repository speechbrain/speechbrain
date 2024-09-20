# Releasing a new version

Here are a few guidelines for when and how to release a new version.
To begin with, as hinted in the "Continuous Integration" document, we would like to follow a
pretty tight release schedule, known as "Continuous Deployment". For us, this means a new
version should be released roughly once a week.

As for how to name the released version, we try to follow semantic versioning for this. More details
can be found at [semver.org](http://semver.org). As it applies to SpeechBrain, some examples
of what this would likely mean:
 * Changes to the Brain class or other core elements often warrant a major version bump (e.g. 1.5.3 -> 2.0.0)
 * Added classes or features warrant a minor version bump. Most weekly updates should fall into this.
 * Patch version bumps should happen only for bug fixes.

**[Final pre-release tests](../tests/PRE-RELEASE-TESTS.md) should be performed!** Some of these checks aren't run by the CI.

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
   as well as accounts at Read the Docs
