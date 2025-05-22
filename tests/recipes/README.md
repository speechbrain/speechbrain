# Templates & recipes

Key point: **specify `test_debug_flags` and make sure testing your recipe works before switching from `Draft` to `Ready for review`.**

---

For GPU testing, install all extra requirements:
```
find recipes | grep extra_requirements.txt | xargs cat | sort -u | grep -v \# | xargs -I {} pip install {}
```

---

If you like to test for all recipes belonging to one dataset:
```
python -c 'from tests.utils.recipe_tests import run_recipe_tests; print("TEST FAILED!") if not(run_recipe_tests(filters_fields=["Dataset"], filters=[["CommonLanguage", "LibriSpeech"]], do_checks=False, run_opts="--device=cuda")) else print("TEST PASSED")'
```

You can run the recipe on the CPU just by setting the run_opts properly:
```
python -c 'from tests.utils.recipe_tests import run_recipe_tests; print("TEST FAILED!") if not(run_recipe_tests(filters_fields=["Dataset"], filters=[["CommonLanguage", "LibriSpeech"]], do_checks=False, run_opts="--device=cpu")) else print("TEST PASSED")'
```

In some cases, you might want to test the recipe on a non-default GPU (e.g, cuda:1). This helps detecting issues in recipes where the device was hard-coded. You can do that simply with:

```
python -c 'from tests.utils.recipe_tests import run_recipe_tests; print("TEST FAILED!") if not(run_recipe_tests(filters_fields=["Dataset"], filters=[["CommonLanguage", "LibriSpeech"]], do_checks=False, run_opts="--device=cuda:0")) else print("TEST PASSED")'
```


To target a specific recipe (here by its hparam yaml):
```
python -c 'from tests.utils.recipe_tests import run_recipe_tests; print("TEST FAILED!") if not(run_recipe_tests(filters_fields=["Hparam_file"], filters=[["recipes/TIMIT/ASR/transducer/hparams/train_wav2vec.yaml"]], do_checks=False, run_opts="--device=cuda")) else print("TEST PASSED")'
```

We also support full inference tests, where we download specific data and an output folder, then conduct inference using the downloaded data.

To run full inference tests, please run:

```
python -c 'from tests.utils.recipe_tests import run_recipe_tests; print("TEST FAILED!") if not(run_recipe_tests(filters_fields=["Task"], filters=[["full_inference"]], do_checks=True, run_opts="--device=cuda")) else print("TEST PASSED")'
```

Note that this tests might take a few hours to complete.



Note: the above examples excluded checks for reaching a specific performance criterion. Their scope is: does the data flow break? [yes/no]
<br/> (to that extent, data preparation is ignored)

_These recipe tests rely on minimal & annotated data, as presented in [tests/samples](https://github.com/speechbrain/speechbrain/tree/develop/tests/samples)._

_Contributors, please ensure that your recipes work with minimal data, so we can keep on testing your work before our next future releases._

__Reviewers: please assist the contributors—to you, running one check on minimal data reduces all your workload to the fun stuff in conversational AI.__

---

Let's take a look at recipes: their structural outline & their testing definition.

## A recipe follows this structure:
* `recipes/DATASET/prepare_data.py` – a Data prep file
  > __Reviewer: with OpenData, does the recipe work with `--debug`? If no data is available, skip this preparation and use `tests/samples` data to check if the recipe breaks/not.__
  <br/><hr/>
  > **User:** provide required _test_debug_flags_ for the reviewing task.
* `recipes/DATASET/extra_requirements.txt` – additional dependencies
* `recipes/DATASET/TASK/METHOD/extra_requirements.txt` – particular, additional dependencies
  > _Note: this can lead to conflicting recipes / which need to point to different e.g. HF hub caches to not conflict one another._
* `recipes/DATASET/TASK/METHOD/train.py` – a _Script_file_
* `recipes/DATASET/TASK/METHOD/hparams/hparam.yaml` – a _Hparam_file_
* `recipes/DATASET/TASK/METHOD/README.md` – a _Readme_file_, which points to
  * some GDrive url – a _Result_url_ [optional]
  * some HuggingFace url – a _HF_repo_ [optional], which has
    * pretrained model – `hyperparameters.yaml` to be loaded either by [a pretrained interface](https://github.com/speechbrain/speechbrain/tree/develop/speechbrain/inference) or a custom interface
    * code snippets, for demonstration
  * additional references, incl. further URLs
  > _Note: all URLs references (in .py, .md & .txt files) are checked to be valid._

## Recipe testing mirrors the recipes' structure:
* `tests/recipes/DATASET.csv` – a summary of testing parameters for templates & recipes, including derived pretrained models
  <br/>_(as hinted above; example: [tests/recipes/LibriSpeech.csv](https://github.com/speechbrain/speechbrain/tree/develop/tests/recipes/LibriSpeech.csv):2)_
  * Task
    >_ASR_
  * Dataset
    > _LibriSpeech_
  * Script_file
    > _recipes/LibriSpeech/ASR/CTC/train_with_wav2vec.py_
  * Hparam_file
    > _recipes/LibriSpeech/ASR/CTC/hparams/train_hf_wav2vec.yaml_
  * Data_prep_file
    > _recipes/LibriSpeech/ASR/CTC/librispeech_prepare.py_
  * Readme_file
    > _recipes/LibriSpeech/ASR/CTC/README.md_
  * Result_url (mandatory/optional?)
    > _https://www.dropbox.com/sh/qj2ps85g8oiicrj/AAAxlkQw5Pfo0M9EyHMi8iAra?dl=0_
  * HF_repo (optional)
    > _https://huggingface.co/speechbrain/asr-wav2vec2-librispeech
  * test_debug_flags
    > _--data_folder=tests/samples/ASR/ --train_csv=tests/samples/annotation/ASR_train.csv --valid_csv=tests/samples/annotation/ASR_train.csv --test_csv=[tests/samples/annotation/ASR_train.csv] --number_of_epochs=10 --skip_prep=True --wav2vec2_folder=tests/tmp/wav2vec2_checkpoint_
  * test_debug_checks (optional)
    > _"file_exists=[env.log,hyperparams.yaml,log.txt,train_log.txt,train_with_wav2vec.py,wer_ASR_train.txt,save/label_encoder.txt] performance_check=[train_log.txt, train loss, <3.5, epoch: 10]"_

These testing parameters are used by checks before releases and by checks after each `git push`.

---

These tools help to check all recipes (for maintainers):
```
tests/.run-load-yaml-tests.sh
tests/.run-recipe-tests.sh
tests/.run-HF-checks.sh
tests/.run-url-checks.sh
```
