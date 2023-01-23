# Guiding contributors, reviewers & maintainers through the complexity of SpeechBrain testing.

SpeechBrain is the name of a speech technology toolkit. It is written in Python and uses an extended YAML (HyperPyYAML) for hyperparameters in recipes, as well as some tutorials and scripts. SpeechBrain (the toolkit) is continuously updated and improved by the SpeechBrain community and the SpeechBrain core team, working together on GitHub. New versions of SpeechBrain (the toolkit) are continuously published by the core team on platforms like PyPI.

If we take a step back, SpeechBrain also refers to a wider ecosystem, which has spread to many different platforms: there's documentation on readthedocs, tutorials on Colab, models on HuggingFace, et cetera. Another important part of SpeechBrain are the recipes. The main GitHub repository houses a set of recipes, which has built up over time.

As SpeechBrain (all of it) is improved and changed, ideally the old, existing parts should continue to work well. However, in reality, changes will break old parts.

The purpose of tests is to ensure that things work, or that at least we know what breaks: for example, SpeechBrain (the toolkit) has unittests which test specific bits of code in the core library. But since SpeechBrain (the ecosystem) is quite wide and spread out, there should also be other types of tests which ensure that the different platforms cooperate and the recipes keep working.

Demonstrating that no harm is done by some given change is a big challenge. Ideally, tests will help in integrating (potentially legacy-breaking) changes without losing the existing achievements.

The following graphics illustrate the different complexities at work when it comes to testing in SpeechBrain.

by Andreas Nautsch, Aku Rouhe, 2022

## Functionality provided on multiple platforms, in the SpeechBrain ecosystem.

```
                  (documentation)           (tutorials)
                  .—————————————.            .———————.
                  | readthedocs |       ‚––> | Colab |
                  \—————————————/      ∕     \———————/
                         ^       ‚––––‘          |
    (release)            |      ∕                v
    .——————.       .———————————. (landing) .———————————.
    | PyPI | –––>  | github.io |  (page)   | templates |   (reference)
    \——————/       \———————————/       ‚–> \———————————/ (implementation)
        |                |        ‚–––‘          |
        v                v       ∕               v
.———————————–—.   .———————————–—.           .—————————.           .~~~~~~~~~~~~~.
| HyperPyYAML |~~~| speechbrain | ––––––––> | recipes | ––––––––> | HuggingFace |
\————————————–/   \————————————–/           \—————————/     ∕     \~~~~~~~~~~~~~/
  (usability)     (source/modules)          (use cases)    ∕    (pretrained models)
                                                          ∕
                        |                        |       ∕               |
                        v                        v      ∕                v
                  .~~~~~~~~~~~~~.            .~~~~~~~~.            .———————————.
                  |   PyTorch   | ––––––––-> | GDrive |            | Inference |
                  \~~~~~~~~~~~~~/            \~~~~~~~~/            \———————————/
                   (checkpoints)             (results)            (code snippets)
```
Each platform/functionality has their own dependencies (which can break) and interfaces (which are specific and can change).

## How is functionality provided?

```
   (imported)        (used in)    (as units in) (integrated by)   (to code)
.——————————————.    .—————————.    .—————————.    .—————————.    .—————————.  |  code & yaml
| dependencies | => | helpers | => | classes | => | modules | => | scripts |  | style checks
\——————————————/    \—————————/    \—————————/    \—————————/    \—————————/  |   (linters)
        |                |              |              |              |
        v                v              v              v              v
 version updates     docstring      unittests     integration      tutorials,
 may change their   examples as      assert       tests ensure     templates,
    interface;     modular tests    expected     working vanilla    recipes &
 latest versions                    behaviour     experiments       snippets
 are controlled by                                               need advanced
requirements configs                                           testing strategies

    [irregular]     [ --- github push workflow actions --- ]  [ hybrid periodicity]
```
Python, business as usual:
* doc tests: one or two examples, that the interface does not crash when being used
* unit tests: set of examples to (more exhaustively) test that function does as should
* integration tests: combination of python snippets, targeted yaml hparams, and minimal examples (audio with text & annotation) to demonstrate a use case for a part of module

1. _Contributor, did you provide a new interface?_ <br/>=> doc test
2. _Contributor, did you improve upon inner workings?_ <br/>=> unit test
3. _Contributor, did you offer new ways to the SpeechBrain community?_ <br/>=> integration test

While one cannot control others (dependencies), CI/CD workflows are periodic actions to assert functionality of the known.
Multi-platform checks, for it goes beyond this repo, is on a hybrid (partly irregular periodicity), i.e., before a future SpeechBrain release.

_Naturally, writing style (linters checks) is a part of functionality._

## How is the SpeechBrain community improving quality, continuously?

```
            .———————————.
            | Closed PR |  (but not merged)
         ‚->\———————————/<-˛
        ∕                   \                      (made it! :)
.——————————.              .—————————.              .———————————.
| Draft PR | –––––––––––> | Open PR | –––––––––––> | Merged PR |
\——————————/              \—————————/              \———————————/
 * create initial          * ensure all             * pre-release
   branch to improve         workflow                 checks (later)
 * state todo list           tests pass             * contribution
   and fulfill it          * collaborate              log entry
 * inquire feedback          on change              * part of next
   early on                  requests                 release tag

     |                          |                     (more below)
     v                          v

To push formatted code:    Review of:
git add ...                 * changes to core modules
pre-commit                  * enhanced testing/documentation
git status                  * contributed tutorial
git add ...                 * new/edited template/tool
git commit -m ...           * added/modified recipe
git push                    * uploaded pretrained model
                            * well-formatted py & yaml files
Missed out on one?
pre-commit run --all-files
git status
git add ...
```

To guide the lifecycle of a PR within the SpeechBrain lifecycle—as contributor and as reviewer—can be demanding to being exhausted.
Test automation (e.g., through github and offline workflows) simplify discussions to the points that are of debate, actually.

## The location of a change foreshadows its integrative complexity.

```
BEFORE
------
           (python)                              (yaml)

def func_sig(x, arg0, arg1=None):    |    my_var: !new:func_sig
    # just to demonstrate changes    |        arg0: 6.28 # tau
    if arg1 is None:                 |
        return x + arg0              |    my_other: !new:func_sig
    else:                            |        arg0: !ref <my_var>
        return x + arg1              |        arg1: 1/137 # fine structure constant


AFTER - A. Changes to function body &/or interface parameterization via YAML
-----
           (python)                              (yaml)

def func_sig(x, arg0,                |    my_arg: !new:func_sig
                arg1=None,):         |        arg0: 6.28
    if arg1 is None:                 |
        return x / arg0              |    my_other: !new:func_sig
    else:                            |        arg0: !ref <my_arg>
        return x - arg1              |        arg1: 0.0073


AFTER - B. Changes to function signature (interface), legacy-preserving
-----
           (python)                              (yaml)

def func_sig(x, arg0, arg1=None,     |    my_arg: !new:func_sig
                arg2=true,):         |        arg0: 6.28
    return next_gen(x, arg0=arg0,    |
                       arg1=arg1,    |    my_other: !new:func_sig
                       arg2=arg2,)   |        arg0: !ref <my_arg>
                                     |        arg1: 0.0073
# the new interface being introduced |
def next_gen(x, arg0=6.28,           |    my_arg_same: !new:next_gen
                arg1=1/137,          |        arg1: None
                arg2=true,):         |
    if !arg2:                        |    my_other_same: !new:next_gen
        return x                     |
    if arg1 is None:                 |    my_new_feature: !new:next_gen
        return x / arg0              |        arg0: 2.718 # e
    else:                            |        arg1: 1.618 # what could it be...
        return x - arg1              |        arg2: false # ;-)


AFTER - C. Changes to function signature (interface), legacy-breaking
-----
           (python)                              (yaml)

def next_gen(x, arg0=6.28,           |    my_arg: !new:next_gen
                arg1=1/137,          |        arg1: None
                arg2=true,):         |
    if !arg2:                        |    my_other: !new:next_gen
        return x                     |
    if arg1 is None:                 |    my_new_feature: !new:next_gen
        return x / arg0              |        arg0: 2.718
    else:                            |        arg1: 1.618
        return x - arg1              |        arg2: false

```
How would you approach testing each of them?
<br/>Such changes happen not only once, but on a regular basis, throughout all core modules.

Changes can be internal to a function &/or alter the function signature:
* function-internal changes are not of concern to other function (so long they do what they should),
* function signature changes impact the overall—the multi-platform ecosystem.

Legacy-breaking changes will impact the outline of all recipes:
<br/>how will all work after a change—, and after the next major refactoring (after that first one)?

## Branch topology: release <- CI/CD <- ecosystem-spanning refactorings.
```
    release | main             | business
      CI/CD |   \--- develop   | as usual
  ecosystem |         \   \<~> testing-refactoring   |  the tricky
refactoring |          \--- unstable <~>/            | bits & pieces
```
The core challenge—to testing SpeechBrain's community-driven development in its multi-platform setting—is tackled through different branches serving each their constructive purpose:
* `main` branch: released on PyPI
* `develop` branch: CI/CD with github workflow; place to merge regular PRs
* `testing-refactoring` branch: [copy of custom interfaces & yaml hparams for pretrained models](https://github.com/speechbrain/speechbrain/tree/testing-refactoring/updates_pretrained_models) hosted on [HuggingFace](https://huggingface.co/speechbrain/)—if changes to the (usually) permanent interface constitution are necessary, we can treat them here (see what happens & improve further)
* `unstable` branch: accumulating legacy-breaking PRs to separate either CI/CD tracks (develop & this one) from one another. When the time of merger comes, the latest `develop` version becomes the final minor release of the passing major version family (e.g., a `0.5.42` before a `0.6.0`). Then, the next lifecycle continues and roots community growth, prepared for new challenges to come.

1. _Contributor, if your change touches upon standing interfaces, then your PR to `develop` or to `unstable` benefits from a companion PR to the `testing-refactoring` branch._<br/>=> Then, reviewers of your main PR is accompanied by provision to also change repos that provide pretrained models.
2. _Contributor, if your idea for change will change function signatures, then your PR strategy needs planning._
   1. _Can the change be split into one legacy-preserving (to `develop`) & anoether legacy-breaking PR (to `unstable`)?_<br/> => Then, reviewers of your legacy-preserving PR can help you with facilitating a smooth transition.
   2. _Can the legacy-breaking PR be tested for its effectiveness with tools available on the `develop` branch?_ <br/> => Then, reviewers will have their time free to discuss with you on improving your change; provide them tools and assistance to engage with your ideas in a way their mind is open to accept your contribution to the SpeechBrain community.

The other files in this folder provide further guidance on where is what configured, and which tools are there to be used.
Keep in mind, the SpeechBrain community is in-flux, so is a constellation of maintainers and reviewers nothing more but a snapshot.

_Note: github workflows take the definition of a PR, what is specified within its branch. We might update our procedures on the `develop` branch (e.g., to meet dependency updates).
Consequentially, PR and `unstable` branches need to fetch from latest `develop` when testing related definitions are updated._
