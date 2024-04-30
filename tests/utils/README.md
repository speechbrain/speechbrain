# With understanding, there is no fear.
```
    release | main             | business
      CI/CD |   \--- develop   | as usual
  ecosystem |         \   \<~> testing-refactoring   |  the tricky
refactoring |          \--- unstable <~>/            | bits & pieces
```

The `testing-refactoring` branch contains this folder:<br/>
https://github.com/speechbrain/speechbrain/tree/hf-interface-testing/updates_pretrained_models

which contains folders identical to the model names uploaded to HuggingFace:<br/>
https://huggingface.co/speechbrain

e.g. [hf-interface-testing/updates_pretrained_models/asr-wav2vec2-librispeech](https://github.com/speechbrain/speechbrain/tree/hf-interface-testing/updates_pretrained_models/asr-wav2vec2-librispeech) outlines testing of the pretrained model for [speechbrain/asr-wav2vec2-librispeech](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech) and the folders can contain:
* `test.yaml` - test definition w/ integrated code [**mandatory**]
* `hyperparams.yaml` - the standing (or updated) specification [**mandatory**]
* `custom_interface.py` - the standing (or updated) custom interface [optional]

_Note: changing parameters mean either a model revision &/or a new model._

While `hyperparams.yaml` & `custom_interface.py` shall be updated through PRs complementary to conventional PRs, `test.yaml` is to be defined once only (and fixed when needed).
Such a complementary PR is for example:
https://github.com/speechbrain/speechbrain/pull/1801

_Note: to update files relevant to the testing as an operation, like `test.yaml`, please create and manage separate PRs for this. Their nature as a PR is of a different kind (not the change of a pretrained model itself)._

Depending on the testing need, `test.yaml` grows - some examples
1. [ssl-wav2vec2-base-librispeech/test.yaml](https://github.com/speechbrain/speechbrain/blob/hf-interface-testing/updates_pretrained_models/ssl-wav2vec2-base-librispeech/test.yaml) - the play between test sample, interface class, and batch function is handled via HF testing in `tests/utils`
   ```yaml
   sample: example.wav # test audio provided via HF repo
   cls: WaveformEncoder # existing speechbrain.inference class
   fnx: encode_batch # it's batch-wise function after audio loading
   ```
2. [asr-wav2vec2-librispeech/test.yaml](https://github.com/speechbrain/speechbrain/blob/hf-interface-testing/updates_pretrained_models/asr-wav2vec2-librispeech/test.yaml) - testing single example & against a dataset test partition
   ```yaml
   sample: example.wav # as above
   cls: EncoderASR # as above
   fnx: transcribe_batch # as above
   dataset: LibriSpeech # which dataset to use -> will create a tests/tmp/LibriSpeech folder
   recipe_yaml: recipes/LibriSpeech/ASR/CTC/hparams/train_hf_wav2vec.yaml # the training recipe for dataloader etc
   overrides: # what of the recipe_yaml needs to be overridden
     output_folder: !ref tests/tmp/<dataset> # the output folder is at the tmp dataset (data prep & eval tasks only)
   dataio: | # which dataio_prepare to import; copy/paste from train_with_wav2vec.py — pay attention to the last line (their dataio_prepare needs to know how to prepare the recipe dataset)
       from recipes.LibriSpeech.librispeech_prepare import prepare_librispeech
       run_on_main(
           prepare_librispeech,
           kwargs={
               "data_folder": recipe_hparams["data_folder"],
               "tr_splits": recipe_hparams["train_splits"],
               "dev_splits": recipe_hparams["dev_splits"],
               "te_splits": recipe_hparams["test_splits"],
               "save_folder": recipe_hparams["output_folder"],
               "merge_lst": recipe_hparams["train_splits"],
               "merge_name": "train.csv",
               "skip_prep": recipe_hparams["skip_prep"],
           },
       )
       from recipes.LibriSpeech.ASR.CTC.train_with_wav2vec import dataio_prepare
   test_datasets: dataio_prepare(recipe_hparams)[2] # where to get the test dataset from that prep pipeline (w/ input args)
   test_loader: test_dataloader_opts # dataloader name as in recipe_yaml
   performance: # which metric classes are used in the training recipe
     CER: # name for testing
       handler: cer_computer # name as in recipe_yaml
       field: error_rate # field/function as used in train script
     WER: # another one
       handler: error_rate_computer # another one
       field: error_rate # another one
   predicted: "[wrd.split(' ') for wrd in predictions[0]]" # what of the forward to use to compute metrics
   targeted: "[wrd.split(' ') for wrd in batch.wrd]" # what of the batch ground-of-truth to use to compute metrics
   to_stats: ids, predicted, targeted # what the metric computation needs from each batch
   ```
3. [emotion-recognition-wav2vec2-IEMOCAP/test.yaml](https://github.com/speechbrain/speechbrain/blob/hf-interface-testing/updates_pretrained_models/emotion-recognition-wav2vec2-IEMOCAP/test.yaml) - custom interfaces
   ```yaml
   sample: anger.wav # as above
   cls: CustomEncoderWav2vec2Classifier # => name of custom class provided through custom interface
   fnx: classify_batch # as above
   foreign: custom_interface.py # name of custom interface availed through HF repo
   dataset: IEMOCAP # as above
   recipe_yaml: recipes/IEMOCAP/emotion_recognition/hparams/train_with_wav2vec2.yaml # as above
   overrides: # as above
     output_folder: !ref tests/tmp/<dataset> # as above
   dataio: | # as above
       from recipes.IEMOCAP.emotion_recognition.iemocap_prepare import prepare_data
       run_on_main(
           prepare_data,
           kwargs={
               "data_original": recipe_hparams["data_folder"],
               "save_json_train": recipe_hparams["train_annotation"],
               "save_json_valid": recipe_hparams["valid_annotation"],
               "save_json_test": recipe_hparams["test_annotation"],
               "split_ratio": [80, 10, 10],
               "different_speakers": recipe_hparams["different_speakers"],
               "test_spk_id": recipe_hparams["test_spk_id"],
               "seed": recipe_hparams["seed"],
           },
       )
       from recipes.IEMOCAP.emotion_recognition.train_with_wav2vec2 import dataio_prep
   test_datasets: dataio_prep(recipe_hparams)["test"] # as above
   test_loader: dataloader_options # as above
   performance: # as above
     ClassError: # as above
       handler: error_stats # as above
       field: average # as above
   predicted: predictions[0] # as above
   targeted: batch.emo_encoded[0] # as above
   to_stats: ids, predicted, targeted, wav_lens # as above
   ```

When testing the HF snippets, use the functions `gather_expected_results()` and `gather_refactoring_results()`.
They will create another yaml in which the gather before/after refactoring test results.
While standing interfaces are drawn from HF repos, their updated/refactored counterparts need to be specified to clone the PR git+branch into, e.g., `tests/tmp/hf_interfaces`. See the default values:
```python
def gather_refactoring_results(
    glob_filter="*",
    new_interfaces_git="https://github.com/speechbrain/speechbrain",  # change to yours
    new_interfaces_branch="hf-interface-testing",  # maybe you have another branch
    new_interfaces_local_dir="tests/tmp/hf_interfaces",  # you can leave this, or put it elsewhere
    yaml_path="tests/tmp/refactoring_results.yaml",  # same here, change only if necessary
):
    ...
```

Examples:
```bash
# expected result(s) for one audio
# git checkout develop
python -c "from tests.utils.refactoring_checks import gather_expected_results;gather_expected_results('asr-wav2vec2-ctc-aishell')"

# result(s) after refactoring
# git checkout refactor_branch
python -c "from tests.utils.refactoring_checks import gather_refactoring_results;gather_refactoring_results('asr-wav2vec2-ctc-aishell')"
```
this will give a warning
>     WARNING - no audio found on HF: asr-wav2vec2-ctc-aishell/example.wav
this means that `tests/samples/single-mic/example1.wav` is taken instead.

---

When testing against a dataset's test partition, this function is used: `test_performance()`.
It will be handled through the main function of `tests/utils/refactoring_checks.py`, which expects its own config e.g.:
`tests/utils/overrides.yaml`.

Example:
```yaml
LibriSpeech_data: !PLACEHOLDER
CommonVoice_EN_data: !PLACEHOLDER
CommonVoice_FR_data: !PLACEHOLDER
IEMOCAP_data: !PLACEHOLDER

new_interfaces_git: https://github.com/speechbrain/speechbrain
new_interfaces_branch: hf-interface-testing
new_interfaces_local_dir: tests/tmp/hf_interfaces

# Filter HF repos (will be used in a local glob dir crawling)
# glob_filter: "*wav2vec2*"
# glob_filter: "*libri*"
glob_filter: "*"

# put False to test 'before' only, e.g. via override
after: True

LibriSpeech:
  data_folder: !ref <LibriSpeech_data>
  skip_prep: True # assuming you know what you do ;)

CommonVoice_EN:
  data_folder: !ref <CommonVoice_EN_data>

CommonVoice_FR:
  data_folder: !ref <CommonVoice_FR_data>

IEMOCAP:
  data_folder: !ref <IEMOCAP_data>
```

Example call:
```
python tests/utils/refactoring_checks.py tests/utils/overrides.yaml --LibriSpeech_data="" --CommonVoice_EN_data="" --CommonVoice_FR_data="" --IEMOCAP_data=""
--glob_filter="*commonvoice*"
```

The use case for this construction is a legacy-preserving refactoring, providing an alternative interface.

_Note: please feel free to create your own derived `overrides.yaml` (for specific cases)._

---

The `unstable` branch serves to collect a series of legacy-breaking PRs before making a major release through develop.

_Note: ofc, the just introduced testing-refactoring strategy is applicable here, also. Especially, as it relaxes testing demands._

