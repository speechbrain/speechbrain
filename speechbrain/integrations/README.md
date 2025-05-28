Third-Party Integrations
------------------------

This python module serves to collect all the (non-recipe) SpeechBrain code that relies on
external libraries not present in the `requirements.txt`. By keeping `requirements.txt`
as small as possible we keep SpeechBrain lightweight and easy to maintain.
In addition, this folder makes it easier to keep track of what third-party tools have been
added and apply different rules to the adding and maintenance of new external integrations.

> [!WARNING]
> Since these third-party integrations rely on libraries not part of the core toolkit, we make
> no guarantees as to the proper functioning of these libraries; they may be
> broken on the develop branch at any time. We will check that they function correctly
> only when creating a new release of the toolkit.

In order to minimize the impact of libraries changing and causing the integrations
to stop functioning, we will add additional tests and checks on code in this module.
If the tests are broken, we may remove rather than fix the code in this integration
depending on our capacity.

To add new code to the module, please ensure it contains runnable examples in the docstring
and tests in the `integrations/tests` folder. You can check that all the tests pass by running

```bash
$ sh tests/.third-party-tests.sh
```

In addition we would like new modules to have 80% or greater coverage of the code, evaluated
using the following code, with `pytest-cov` installed:

```bash
$ pytest --cov=speechbrain/integrations --cov-context=test --doctest-modules speechbrain/integrations
```
