# What's the point of this

This testing template is an integration test that combines SpeechBrain features which require multiple GPUs. As such, this test case is supposed to be run offline and serves more as a tool to debug that this works (outside of GitHub workflows). Since one can modify this recipe for further use cases, it also serves as a template.

Tested and demonstrated features are:
* DDP
* Dynamic batching
* Fine-tuning of LM only (inexpensive compared to whole ASR system)
* Mixed fetching: local; from HuggingFace & from a URL
* Testing using pretrained interface & its `load_audio` function<br/>(minilibrispeech as local dataset w/ file path via data loader from recipe template)

Essentially, this template is a continuation of the ASR speech recognition template. Thus, this template makes a few assumptions:
1. There is a checkpoint `templates/speech_recognition/ASR/results/CRDNN_BPE_960h_LM/2602/save/CKPT+latest` <br/><br/>
   > This test case seeks to force SpeechBrain's fetching to refer to this local path using a symbolic link from `speechbrain/asr-crdnn-rnnlm-librispeech` (which is also a HuggingFace repository).
2. OpenRIR & minilibrispeech datasets are located at: `templates/speech_recognition/data` with prepared JSON descriptors in `speechbrain/templates/speech_recognition` (for train; valid & test sets).
   > Simply to avoid re-downloading everything.

---

# How to run

```shell
PYTHONPATH=../../.. python -m torch.distributed.launch --nproc_per_node=2 finetune_LM.py fixed_ASR.yaml --distributed_launch --distributed_backend='nccl'
```

---

# Note

This study uses a data loader which handles local audio files by their local folders. If you wish to fetch remote audios, please keep track of the folder structure.

