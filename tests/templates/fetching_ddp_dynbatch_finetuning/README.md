# Stuck with DDP and multi-source pretrained locations?

This testing template is an integration test that combines SpeechBrain features which require multiple GPUs
(intermediate integration examples are for single-GPU, so one can go step-by-step).
As such, this test case is supposed to be run offline and serves more as a tool to debug that this works
(outside of GitHub workflows).

Since one can modify this recipe for further use cases, it also serves as a template.
Eventually, code snippets provided here might find proper re-use in recipes and testing of pretrained models,
e.g. when re-running evaluations on int'l research challenges,
(where one needs to obtain scores for fusion &/or calibration of classifiers for submission).

Tested and demonstrated features are:
* DDP
* Dynamic batching
* Fine-tuning of LM only (inexpensive compared to whole ASR system); mock-up code
* Mixed fetching: local; from HuggingFace & from a URL
* Testing using pretrained interface & its `load_audio` function<br/>(minilibrispeech as local dataset w/ file path via data loader from recipe template)

Essentially, the extensive case study in this testing is a continuation of the ASR speech recognition template.
Thus, before getting started, please meet the following assumptions:
1. There is a checkpoint `templates/speech_recognition/ASR/results/CRDNN_BPE_960h_LM/2602/save/CKPT+latest` <br/><br/>
   > This test case seeks to force SpeechBrain's fetching to refer to this local path using a symbolic link from `speechbrain/asr-crdnn-rnnlm-librispeech` (which is also a HuggingFace repository). This is to challenge conflicting names in FetchSources, and to move on properly.
2. OpenRIR & minilibrispeech datasets are located at: `templates/speech_recognition/data` with prepared JSON descriptors in `speechbrain/templates/speech_recognition` (for train; valid & test sets).
   > Simply to avoid re-downloading everything. This should be part of ensuring the above (-> just run the ASR templates before starting here).

Each of the test cases below is of a different nature.

---

## Extensive case: handling on the Python side
The `finetune.py` example serves as a template to an extensive set of all of the above use cases combined.
Accompanied by `finetune.yaml`, a YAML infrastructure is demonstrated that uses inputs from other hparam files.
The benefit is a simple one: the training/fine-tuning hparams share a common set of yaml entries with the pretrained model: `ASR.yaml`.
As such, the pretrained model yaml `source_pretrained/pretrained.ymal` also references inputs from `ASR.yaml` and
provides symlinks in `source_pretrained` to an expected CKPT that is created during the execution of the script.

How to run with DDP:
```shell
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=../../.. torchrun --nproc_per_node=2 finetune.py finetune.yaml
```

## Sanity check: standard model card from one of our HuggingFace repos

How to run on single-GPU:
```shell
PYTHONPATH=../../.. python single_node_pretrained.py
```

## Sanity check: multi-source fetching with handling on the YAML-side

For single-GPU regression testing with a mini recipe:
```shell
cd ../../.. && PYTHONPATH=. python tests/templates/fetching_ddp_dynbatch_finetuning/multisource_mini_recipe.py tests/templates/fetching_ddp_dynbatch_finetuning/multisource_mini_recipe.yaml --debug --debug_persistently; cd -
```

## Extensive case: handling on the YAML-side

How to run with DDP:
```shell
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=../../.. torchrun --nproc_per_node=2 finetune_fetch_once.py finetune_fetch_once.yaml
```


# Note(s)

This study uses a data loader which handles local audio files by their local folders. If you wish to fetch remote audios, please keep track of the folder structure.

DDP is not intended (yet) for SpeechBrain's pretrained interfaces; please use the Brain class.

Please play with the batch size for testing; see `finetune.yaml`:
```yaml
test_dataloader_opts:
    batch_size: !ref <batch_size>
    # batch_size: 1  # to ensure metrics w/o padding (or alike) impacts
    num_workers: !ref <num_workers>
```
You'll get different results depending on which one you take. Here are exemplary logs for `batch_size: 8`:
```
__main__ - Batch from scratch w/ pretrainer_load_audio=True
WER on DDP rank 0: {'WER': 7.366330698040391, 'SER': 40.30534351145038, 'num_edits': 1966, 'num_scored_tokens': 26689, 'num_erraneous_sents': 528, 'num_scored_sents': 1310, 'num_absent_sents': 0, 'num_ref_sents': 1310, 'insertions': 1039, 'deletions': 96, 'substitutions': 831, 'error_rate': 7.366330698040391}
WER on DDP rank 1: {'WER': 6.7949163672886, 'SER': 41.98473282442748, 'num_edits': 1759, 'num_scored_tokens': 25887, 'num_erraneous_sents': 550, 'num_scored_sents': 1310, 'num_absent_sents': 0, 'num_ref_sents': 1310, 'insertions': 756, 'deletions': 145, 'substitutions': 858, 'error_rate': 6.7949163672886}
	Summary: {'WER': 7.084981740718199, 'SER': 41.14503816793893, 'num_edits': 3725, 'num_scored_tokens': 52576, 'num_erraneous_sents': 1078, 'num_scored_sents': 2620, 'num_absent_sents': 0, 'num_ref_sents': 2620, 'insertions': 1795, 'deletions': 241, 'substitutions': 1689, 'error_rate': 7.084981740718199}
	Summary: {'WER': 7.084981740718199, 'SER': 41.14503816793893, 'num_edits': 3725, 'num_scored_tokens': 52576, 'num_erraneous_sents': 1078, 'num_scored_sents': 2620, 'num_absent_sents': 0, 'num_ref_sents': 2620, 'insertions': 1795, 'deletions': 241, 'substitutions': 1689, 'error_rate': 7.084981740718199}
__main__ - WER: 7.084981740718199

__main__ - Batch from scratch w/ pretrainer_load_audio=False
WER on DDP rank 0: {'WER': 7.366330698040391, 'SER': 40.30534351145038, 'num_edits': 1966, 'num_scored_tokens': 26689, 'num_erraneous_sents': 528, 'num_scored_sents': 1310, 'num_absent_sents': 0, 'num_ref_sents': 1310, 'insertions': 1039, 'deletions': 96, 'substitutions': 831, 'error_rate': 7.366330698040391}
WER on DDP rank 1: {'WER': 6.7949163672886, 'SER': 41.98473282442748, 'num_edits': 1759, 'num_scored_tokens': 25887, 'num_erraneous_sents': 550, 'num_scored_sents': 1310, 'num_absent_sents': 0, 'num_ref_sents': 1310, 'insertions': 756, 'deletions': 145, 'substitutions': 858, 'error_rate': 6.7949163672886}
	Summary: {'WER': 7.084981740718199, 'SER': 41.14503816793893, 'num_edits': 3725, 'num_scored_tokens': 52576, 'num_erraneous_sents': 1078, 'num_scored_sents': 2620, 'num_absent_sents': 0, 'num_ref_sents': 2620, 'insertions': 1795, 'deletions': 241, 'substitutions': 1689, 'error_rate': 7.084981740718199}
__main__ - WER: 7.084981740718199

__main__ - Testing w/ asr_brain's eval dataloader
	Summary: {'WER': 7.018411442483262, 'SER': 40.83969465648855, 'num_edits': 3690, 'num_scored_tokens': 52576, 'num_erraneous_sents': 1070, 'num_scored_sents': 2620, 'num_absent_sents': 0, 'num_ref_sents': 2620, 'insertions': 1781, 'deletions': 246, 'substitutions': 1663, 'error_rate': 7.018411442483262}
__main__ - WER: 7.018411442483262
```
(DDP-WER: 7.08% & one GPU WER: 7.02%), and [after commenting-out the `train()` step] with `batch_size: 1` (both WERs: 4.14%):
```
__main__ - Batch from scratch w/ pretrainer_load_audio=True
WER on DDP rank 0: {'WER': 4.087826445352018, 'SER': 39.00763358778626, 'num_edits': 1091, 'num_scored_tokens': 26689, 'num_erraneous_sents': 511, 'num_scored_sents': 1310, 'num_absent_sents': 0, 'num_ref_sents': 1310, 'insertions': 170, 'deletions': 90, 'substitutions': 831, 'error_rate': 4.087826445352018}
WER on DDP rank 1: {'WER': 4.199018812531387, 'SER': 40.68702290076336, 'num_edits': 1087, 'num_scored_tokens': 25887, 'num_erraneous_sents': 533, 'num_scored_sents': 1310, 'num_absent_sents': 0, 'num_ref_sents': 1310, 'insertions': 104, 'deletions': 142, 'substitutions': 841, 'error_rate': 4.199018812531387}
	Summary: {'WER': 4.142574558734023, 'SER': 39.847328244274806, 'num_edits': 2178, 'num_scored_tokens': 52576, 'num_erraneous_sents': 1044, 'num_scored_sents': 2620, 'num_absent_sents': 0, 'num_ref_sents': 2620, 'insertions': 274, 'deletions': 232, 'substitutions': 1672, 'error_rate': 4.142574558734023}
__main__ - WER: 4.142574558734023

__main__ - Batch from scratch w/ pretrainer_load_audio=False
WER on DDP rank 0: {'WER': 4.087826445352018, 'SER': 39.00763358778626, 'num_edits': 1091, 'num_scored_tokens': 26689, 'num_erraneous_sents': 511, 'num_scored_sents': 1310, 'num_absent_sents': 0, 'num_ref_sents': 1310, 'insertions': 170, 'deletions': 90, 'substitutions': 831, 'error_rate': 4.087826445352018}
WER on DDP rank 1: {'WER': 4.199018812531387, 'SER': 40.68702290076336, 'num_edits': 1087, 'num_scored_tokens': 25887, 'num_erraneous_sents': 533, 'num_scored_sents': 1310, 'num_absent_sents': 0, 'num_ref_sents': 1310, 'insertions': 104, 'deletions': 142, 'substitutions': 841, 'error_rate': 4.199018812531387}
	Summary: {'WER': 4.142574558734023, 'SER': 39.847328244274806, 'num_edits': 2178, 'num_scored_tokens': 52576, 'num_erraneous_sents': 1044, 'num_scored_sents': 2620, 'num_absent_sents': 0, 'num_ref_sents': 2620, 'insertions': 274, 'deletions': 232, 'substitutions': 1672, 'error_rate': 4.142574558734023}
	Summary: {'WER': 4.142574558734023, 'SER': 39.847328244274806, 'num_edits': 2178, 'num_scored_tokens': 52576, 'num_erraneous_sents': 1044, 'num_scored_sents': 2620, 'num_absent_sents': 0, 'num_ref_sents': 2620, 'insertions': 274, 'deletions': 232, 'substitutions': 1672, 'error_rate': 4.142574558734023}
__main__ - WER: 4.142574558734023

__main__ - Testing w/ asr_brain's eval dataloader
	Summary: {'WER': 4.142574558734023, 'SER': 39.847328244274806, 'num_edits': 2178, 'num_scored_tokens': 52576, 'num_erraneous_sents': 1044, 'num_scored_sents': 2620, 'num_absent_sents': 0, 'num_ref_sents': 2620, 'insertions': 274, 'deletions': 232, 'substitutions': 1672, 'error_rate': 4.142574558734023}
__main__ - WER: 4.142574558734023
```
Please feel free to use this template for testing & debugging when attempting to fix this, and avail the fix to the core module (compare commented code block in the `make_dataloader` function). At the time of writing, the WER of the last log entry has been equivalent to the outcome of `asr_brain.evaluate()` for both were `run_on_main` (a fix implies larger refactoring).

