## Reviewing code

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