# SpeechBrain Performance Report
This document provides an overview of the performance achieved on key datasets and tasks supported by SpeechBrain.

## AISHELL-1 Dataset

### ASR

| Model | Checkpoints | HuggingFace | Test-CER |
| --------| --------| --------| --------|
 | [`recipes/AISHELL-1/ASR/CTC/hparams/train_with_wav2vec.yaml`](recipes/AISHELL-1/ASR/CTC/hparams/train_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/e4bth1bylk7c6h8/AADFq3cWzBBKxuDv09qjvUMta?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-ctc-aishell) | 5.06 |
 | [`recipes/AISHELL-1/ASR/seq2seq/hparams/train.yaml`](recipes/AISHELL-1/ASR/seq2seq/hparams/train.yaml) | [here](https://www.dropbox.com/sh/kefuzzf6jaljqbr/AADBRWRzHz74GCMDqJY9BES4a?dl=0) | - | 7.51 |
 | [`recipes/AISHELL-1/ASR/transformer/hparams/train_ASR_transformer.yaml`](recipes/AISHELL-1/ASR/transformer/hparams/train_ASR_transformer.yaml) | [here](https://www.dropbox.com/sh/tp6tjmysorgvsr4/AAD7KNqi1ot0gR4N406JbKM6a?dl=0) | [here](https://huggingface.co/speechbrain/asr-transformer-aishell) | 6.04 |
 | [`recipes/AISHELL-1/ASR/transformer/hparams/train_ASR_transformer_with_wav2vect.yaml`](recipes/AISHELL-1/ASR/transformer/hparams/train_ASR_transformer_with_wav2vect.yaml) | [here](https://www.dropbox.com/sh/tp6tjmysorgvsr4/AAD7KNqi1ot0gR4N406JbKM6a?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-transformer-aishell) | 5.58 |


## Aishell1Mix Dataset

### Separation

| Model | Checkpoints | HuggingFace | SI-SNRi |
| --------| --------| --------| --------|
 | [`recipes/Aishell1Mix/separation/hparams/sepformer-aishell1mix2.yaml`](recipes/Aishell1Mix/separation/hparams/sepformer-aishell1mix2.yaml) | [here](https://www.dropbox.com/sh/6x9356yuybj8lue/AABPlpS03Vcci_E3jA69oKoXa?dl=0) | - | 13.4dB |
 | [`recipes/Aishell1Mix/separation/hparams/sepformer-aishell1mix3.yaml`](recipes/Aishell1Mix/separation/hparams/sepformer-aishell1mix3.yaml) | [here](https://www.dropbox.com/sh/6x9356yuybj8lue/AABPlpS03Vcci_E3jA69oKoXa?dl=0) | - | 11.2dB |


## BinauralWSJ0Mix Dataset

### Separation

| Model | Checkpoints | HuggingFace | SI-SNRi |
| --------| --------| --------| --------|
 | [`recipes/BinauralWSJ0Mix/separation/hparams/convtasnet-cross.yaml`](recipes/BinauralWSJ0Mix/separation/hparams/convtasnet-cross.yaml) | [here](https://www.dropbox.com/sh/i7fhu7qswjb84gw/AABsX1zP-GOTmyl86PtU8GGua?dl=0) | - | 12.39dB |
 | [`recipes/BinauralWSJ0Mix/separation/hparams/convtasnet-independent.yaml`](recipes/BinauralWSJ0Mix/separation/hparams/convtasnet-independent.yaml) | [here](https://www.dropbox.com/sh/i7fhu7qswjb84gw/AABsX1zP-GOTmyl86PtU8GGua?dl=0) | - | 11.90dB |
 | [`recipes/BinauralWSJ0Mix/separation/hparams/convtasnet-parallel-noise.yaml`](recipes/BinauralWSJ0Mix/separation/hparams/convtasnet-parallel-noise.yaml) | [here](https://www.dropbox.com/sh/i7fhu7qswjb84gw/AABsX1zP-GOTmyl86PtU8GGua?dl=0) | - | 18.25dB |
 | [`recipes/BinauralWSJ0Mix/separation/hparams/convtasnet-parallel-reverb.yaml`](recipes/BinauralWSJ0Mix/separation/hparams/convtasnet-parallel-reverb.yaml) | [here](https://www.dropbox.com/sh/i7fhu7qswjb84gw/AABsX1zP-GOTmyl86PtU8GGua?dl=0) | - | 6.95dB |
 | [`recipes/BinauralWSJ0Mix/separation/hparams/convtasnet-parallel.yaml`](recipes/BinauralWSJ0Mix/separation/hparams/convtasnet-parallel.yaml) | [here](https://www.dropbox.com/sh/i7fhu7qswjb84gw/AABsX1zP-GOTmyl86PtU8GGua?dl=0) | - | 16.93dB |


## CVSS Dataset

### S2ST

| Model | Checkpoints | HuggingFace | Test-sacrebleu |
| --------| --------| --------| --------|
 | [`recipes/CVSS/S2ST/hparams/train_fr-en.yaml`](recipes/CVSS/S2ST/hparams/train_fr-en.yaml) | [here]( https://www.dropbox.com/sh/woz4i1p8pkfkqhf/AACmOvr3sS7p95iXl3twCj_xa?dl=0) | [here]( ) | 24.47 |


## CommonLanguage Dataset

### Language-id

| Model | Checkpoints | HuggingFace | Error |
| --------| --------| --------| --------|
 | [`recipes/CommonLanguage/lang_id/hparams/train_ecapa_tdnn.yaml`](recipes/CommonLanguage/lang_id/hparams/train_ecapa_tdnn.yaml) | [here](https://www.dropbox.com/sh/1fxpzyv67ouwd2c/AAAeMUWYP2f1ycpE1Lp1CwEla?dl=0) | [here](https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa) | 15.1% |


## CommonVoice Dataset

### ASR-seq2seq

| Model | Checkpoints | HuggingFace | Test-WER |
| --------| --------| --------| --------|
 | [`recipes/CommonVoice/ASR/seq2seq/hparams/train_de.yaml`](recipes/CommonVoice/ASR/seq2seq/hparams/train_de.yaml) | [here](https://www.dropbox.com/sh/zgatirb118f79ef/AACmjh-D94nNDWcnVI4Ef5K7a?dl=0) | [here](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-de) | 12.25% |
 | [`recipes/CommonVoice/ASR/seq2seq/hparams/train_en.yaml`](recipes/CommonVoice/ASR/seq2seq/hparams/train_en.yaml) | [here](https://www.dropbox.com/sh/h8ged0yu3ztypkh/AAAu-12k_Ceg-tTjuZnrg7dza?dl=0) | [here](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-en) | 23.88% |
 | [`recipes/CommonVoice/ASR/seq2seq/hparams/train_fr.yaml`](recipes/CommonVoice/ASR/seq2seq/hparams/train_fr.yaml) | [here](https://www.dropbox.com/sh/07a5lt21wxp98x5/AABhNwmWFaNFyA734bNZUO03a?dl=0) | [here](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-fr) | 14.88% |
 | [`recipes/CommonVoice/ASR/seq2seq/hparams/train_it.yaml`](recipes/CommonVoice/ASR/seq2seq/hparams/train_it.yaml) | [here](https://www.dropbox.com/sh/ss59uu0j5boscvp/AAASsiFhlB1nDWPkFX410bzna?dl=0) | [here](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-it) | 17.02% |
 | [`recipes/CommonVoice/ASR/seq2seq/hparams/train_rw.yaml`](recipes/CommonVoice/ASR/seq2seq/hparams/train_rw.yaml) | [here](https://www.dropbox.com/sh/i1fv4f8miilqgii/AAB3gE97kmFDA0ISkIDSUW_La?dl=0) | [here](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-rw) | 29.22% |
 | [`recipes/CommonVoice/ASR/seq2seq/hparams/train_es.yaml`](recipes/CommonVoice/ASR/seq2seq/hparams/train_es.yaml) | [here](https://www.dropbox.com/sh/r3w0b2tm1p73vft/AADCxdhUwDN6j4PVT9TYe-d5a?dl=0) | [here](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-es) | 14.77% |


### ASR-CTC

| Model | Checkpoints | HuggingFace | Test-WER |
| --------| --------| --------| --------|
 | [`recipes/CommonVoice/ASR/CTC/hparams/train_en_with_wav2vec.yaml`](recipes/CommonVoice/ASR/CTC/hparams/train_en_with_wav2vec.yaml) | [here](https://www.dropbox.com/scl/fo/gx0szpbectig2r6r6p9vk/APdoN_wWWq_wP4My7w6SvMo?rlkey=v8fhd887bn947yjb45i99wm8p&st=6muft51b&dl=0) | - | 16.16% |
 | [`recipes/CommonVoice/ASR/CTC/hparams/train_fr_with_wav2vec.yaml`](recipes/CommonVoice/ASR/CTC/hparams/train_fr_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/0i7esfa8jp3rxpp/AAArdi8IuCRmob2WAS7lg6M4a?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-fr) | 9.71% |
 | [`recipes/CommonVoice/ASR/CTC/hparams/train_it_with_wav2vec.yaml`](recipes/CommonVoice/ASR/CTC/hparams/train_it_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/hthxqzh5boq15rn/AACftSab_FM6EFWWPgHpKw82a?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-it) | 7.99% |
 | [`recipes/CommonVoice/ASR/CTC/hparams/train_rw_with_wav2vec.yaml`](recipes/CommonVoice/ASR/CTC/hparams/train_rw_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/4iax0l4yfry37gn/AABuQ31JY-Sbyi1VlOJfV7haa?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-rw) | 22.52% |
 | [`recipes/CommonVoice/ASR/CTC/hparams/train_de_with_wav2vec.yaml`](recipes/CommonVoice/ASR/CTC/hparams/train_de_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/dn7plq4wfsujsi1/AABS1kqB_uqLJVkg-bFkyPpVa?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-de) | 8.39% |
 | [`recipes/CommonVoice/ASR/CTC/hparams/train_ar_with_wav2vec.yaml`](recipes/CommonVoice/ASR/CTC/hparams/train_ar_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/7tnuqqbr4vy96cc/AAA_5_R0RmqFIiyR0o1nVS4Ia?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-ar) | 28.53% |
 | [`recipes/CommonVoice/ASR/CTC/hparams/train_es_with_wav2vec.yaml`](recipes/CommonVoice/ASR/CTC/hparams/train_es_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/ejvzgl3d3g8g9su/AACYtbSWbDHvBr06lAb7A4mVa?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-es) | 12.67% |
 | [`recipes/CommonVoice/ASR/CTC/hparams/train_pt_with_wav2vec.yaml`](recipes/CommonVoice/ASR/CTC/hparams/train_pt_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/80wucrvijdvao2a/AAD6-SZ2_ZZXmlAjOTw6fVloa?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-pt) | 21.69% |
 | [`recipes/CommonVoice/ASR/CTC/hparams/train_zh-CN_with_wav2vec.yaml`](recipes/CommonVoice/ASR/CTC/hparams/train_zh-CN_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/2bikr81vgufoglf/AABMpD0rLIaZBxjtwBHgrNpga?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-zh-CN) | 23.17% |


### ASR-transformer

| Model | Checkpoints | HuggingFace | Test-WER |
| --------| --------| --------| --------|
 | [`recipes/CommonVoice/ASR/transformer/hparams/train_hf_whisper.yaml`](recipes/CommonVoice/ASR/transformer/hparams/train_hf_whisper.yaml) | - | - | 16.96% |


## DNS Dataset

### Enhancement

| Model | Checkpoints | HuggingFace | valid-PESQ | test-SIG | test-BAK | test-OVRL |
| --------| --------| --------| --------| --------| --------| --------|
 | [`recipes/DNS/enhancement/hparams/sepformer-dns-16k.yaml`](recipes/DNS/enhancement/hparams/sepformer-dns-16k.yaml) | [here](https://www.dropbox.com/sh/d3rp5d3gjysvy7c/AACmwcEkm_IFvaW1lt2GdtQka?dl=0) | [here](https://huggingface.co/speechbrain/sepformer-dns4-16k-enhancement) | 2.06 | 2.999 | 3.076 | 2.437 |


## DVoice Dataset

### ASR-CTC

| Model | Checkpoints | HuggingFace | Test-WER |
| --------| --------| --------| --------|
 | [`recipes/DVoice/ASR/CTC/hparams/train_amh_with_wav2vec.yaml`](recipes/DVoice/ASR/CTC/hparams/train_amh_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/pyu40jq1ebv6hcc/AADQO_lAD-F9Q0vlVq8KoXHqa?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-amharic) | 24.92% |
 | [`recipes/DVoice/ASR/CTC/hparams/train_dar_with_wav2vec.yaml`](recipes/DVoice/ASR/CTC/hparams/train_dar_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/pyu40jq1ebv6hcc/AADQO_lAD-F9Q0vlVq8KoXHqa?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-darija) | 18.28% |
 | [`recipes/DVoice/ASR/CTC/hparams/train_fon_with_wav2vec.yaml`](recipes/DVoice/ASR/CTC/hparams/train_fon_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/pyu40jq1ebv6hcc/AADQO_lAD-F9Q0vlVq8KoXHqa?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-fongbe) | 9.00% |
 | [`recipes/DVoice/ASR/CTC/hparams/train_sw_with_wav2vec.yaml`](recipes/DVoice/ASR/CTC/hparams/train_sw_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/pyu40jq1ebv6hcc/AADQO_lAD-F9Q0vlVq8KoXHqa?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-swahili) | 23.16% |
 | [`recipes/DVoice/ASR/CTC/hparams/train_wol_with_wav2vec.yaml`](recipes/DVoice/ASR/CTC/hparams/train_wol_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/pyu40jq1ebv6hcc/AADQO_lAD-F9Q0vlVq8KoXHqa?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-wolof) | 16.05% |


### Multilingual-ASR-CTC

| Model | Checkpoints | HuggingFace | WER-Darija | WER-Swahili | WER-Fongbe | Fongbe-Wolof | WER-Amharic |
| --------| --------| --------| --------| --------| --------| --------| --------|
 | [`recipes/DVoice/ASR/CTC/hparams/train_multi_with_wav2vec.yaml`](recipes/DVoice/ASR/CTC/hparams/train_multi_with_wav2vec.yaml) | [here](https://www.dropbox.com/sh/pyu40jq1ebv6hcc/AADQO_lAD-F9Q0vlVq8KoXHqa?dl=0) | - | 13.27% | 29.31% | 10.26% | 21.54% | 31.15% |


## ESC50 Dataset

### SoundClassification

| Model | Checkpoints | HuggingFace | Accuracy |
| --------| --------| --------| --------|
 | [`recipes/ESC50/classification/hparams/cnn14.yaml`](recipes/ESC50/classification/hparams/cnn14.yaml) | [here](https://www.dropbox.com/sh/fbe7l14o3n8f5rw/AACABE1BQGBbX4j6A1dIhBcSa?dl=0) | - | 82% |
 | [`recipes/ESC50/classification/hparams/conv2d.yaml`](recipes/ESC50/classification/hparams/conv2d.yaml) | [here](https://www.dropbox.com/sh/tl2pbfkreov3z7e/AADwwhxBLw1sKvlSWzp6DMEia?dl=0) | - | 75% |


## Fisher-Callhome-Spanish Dataset

### Speech_Translation

| Model | Checkpoints | HuggingFace | Test-sacrebleu |
| --------| --------| --------| --------|
 | [`recipes/Fisher-Callhome-Spanish/ST/transformer/hparams/transformer.yaml`](recipes/Fisher-Callhome-Spanish/ST/transformer/hparams/transformer.yaml) | [here](https://www.dropbox.com/sh/tmh7op8xwthdta0/AACuU9xHDHPs8ToxIIwoTLB0a?dl=0) | - | 47.31 |
 | [`recipes/Fisher-Callhome-Spanish/ST/transformer/hparams/conformer.yaml`](recipes/Fisher-Callhome-Spanish/ST/transformer/hparams/conformer.yaml) | [here](https://www.dropbox.com/sh/tmh7op8xwthdta0/AACuU9xHDHPs8ToxIIwoTLB0a?dl=0) | - | 48.04 |


## Google-speech-commands Dataset

### Command_recognition

| Model | Checkpoints | HuggingFace | Test-accuracy |
| --------| --------| --------| --------|
 | [`recipes/Google-speech-commands/hparams/xvect.yaml`](recipes/Google-speech-commands/hparams/xvect.yaml) | [here](https://www.dropbox.com/sh/9n9q42pugbx0g7a/AADihpfGKuWf6gkwQznEFINDa?dl=0) | [here](https://huggingface.co/speechbrain/google_speech_command_xvector) | 97.43% |
 | [`recipes/Google-speech-commands/hparams/xvect_leaf.yaml`](recipes/Google-speech-commands/hparams/xvect_leaf.yaml) | [here](https://www.dropbox.com/sh/r63w4gytft4s1x6/AAApP8-pp179QKGCZHV_OuD8a?dl=0) | - | 96.79% |


## IEMOCAP Dataset

### Emotion_recognition

| Model | Checkpoints | HuggingFace | Test-Accuracy |
| --------| --------| --------| --------|
 | [`recipes/IEMOCAP/emotion_recognition/hparams/train_with_wav2vec2.yaml`](recipes/IEMOCAP/emotion_recognition/hparams/train_with_wav2vec2.yaml) | [here](https://www.dropbox.com/sh/lmebg4li83sgkhg/AACooPKbNlwd-7n5qSJMbc7ya?dl=0) | [here](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP/) | 65.7% |
 | [`recipes/IEMOCAP/emotion_recognition/hparams/train.yaml`](recipes/IEMOCAP/emotion_recognition/hparams/train.yaml) | [here](https://www.dropbox.com/sh/ke4fxiry97z58m8/AACPEOM5bIyxo9HxG2mT9v_aa?dl=0) | - | 77.0% |


## IWSLT22_lowresource Dataset

### Speech_Translation

| Model | Checkpoints | HuggingFace | Test-BLEU |
| --------| --------| --------| --------|
 | [`recipes/IWSLT22_lowresource/AST/transformer/hparams/train_w2v2_mbart_st.yaml`](recipes/IWSLT22_lowresource/AST/transformer/hparams/train_w2v2_mbart_st.yaml) | [here](https://www.dropbox.com/sh/xjo0ou739oksnus/AAAgyrCwywmDRRuUiDnUva2za?dl=0) | - | 7.73 |
 | [`recipes/IWSLT22_lowresource/AST/transformer/hparams/train_w2v2_nllb_st.yaml`](recipes/IWSLT22_lowresource/AST/transformer/hparams/train_w2v2_nllb_st.yaml) | [here](https://www.dropbox.com/sh/spp2ijgfdbzuz26/AABkJ97e72D7aKzNLTm1qmWEa?dl=0) | - | 8.70 |
 | [`recipes/IWSLT22_lowresource/AST/transformer/hparams/train_samu_mbart_st.yaml`](recipes/IWSLT22_lowresource/AST/transformer/hparams/train_samu_mbart_st.yaml) | [here](https://www.dropbox.com/sh/98s1xyc3chreaw6/AABom3FnwY5SsIvg4en9tWC2a?dl=0) | - | 10.28 |
 | [`recipes/IWSLT22_lowresource/AST/transformer/hparams/train_samu_nllb_st.yaml`](recipes/IWSLT22_lowresource/AST/transformer/hparams/train_samu_nllb_st.yaml) | [here](https://www.dropbox.com/sh/ekkpl9c3kxsgllj/AABa0q2LrJe_o7JF-TTbfxZ-a?dl=0) | - | 11.32 |


## KsponSpeech Dataset

### ASR

| Model | Checkpoints | HuggingFace | clean-WER | others-WER |
| --------| --------| --------| --------| --------|
 | [`recipes/KsponSpeech/ASR/transformer/hparams/conformer_medium.yaml`](recipes/KsponSpeech/ASR/transformer/hparams/conformer_medium.yaml) | [here](https://www.dropbox.com/sh/uibokbz83o8ybv3/AACtO5U7mUbu_XhtcoOphAjza?dl=0) | [here](https://huggingface.co/speechbrain/asr-conformer-transformerlm-ksponspeech) | 20.78% | 25.73% |


## LibriMix Dataset

### Separation

| Model | Checkpoints | HuggingFace | SI-SNR |
| --------| --------| --------| --------|
 | [`recipes/LibriMix/separation/hparams/sepformer-libri2mix.yaml`](recipes/LibriMix/separation/hparams/sepformer-libri2mix.yaml) | [here](https://www.dropbox.com/sh/skkiozml92xtgdo/AAD0eJxgbCTK03kAaILytGtVa?dl=0) | - | 20.4dB |
 | [`recipes/LibriMix/separation/hparams/sepformer-libri3mix.yaml`](recipes/LibriMix/separation/hparams/sepformer-libri3mix.yaml) | [here](https://www.dropbox.com/sh/kmyz7tts9tyg198/AACsDcRwKvelXxEB-k5q1OaIa?dl=0) | - | 19.0dB |


## LibriParty Dataset

### VAD

| Model | Checkpoints | HuggingFace | Test-Precision | Recall | F-Score |
| --------| --------| --------| --------| --------| --------|
 | [`recipes/LibriParty/VAD/hparams/train.yaml`](recipes/LibriParty/VAD/hparams/train.yaml) | [here](https://www.dropbox.com/sh/6yguuzn4pybjasd/AABpUF8LAQ8d2TJyC8aK2OBga?dl=0 ) | [here](https://huggingface.co/speechbrain/vad-crdnn-libriparty) | 0.9518 | 0.9437 | 0.9477 |


## LibriSpeech Dataset

### ASR-Transformers

| Model | Checkpoints | HuggingFace | Test_clean-WER | Test_other-WER |
| --------| --------| --------| --------| --------|
 | [`recipes/LibriSpeech/ASR/transformer/hparams/conformer_small.yaml`](recipes/LibriSpeech/ASR/transformer/hparams/conformer_small.yaml) | [here](https://www.dropbox.com/sh/s0x6ni124858b8i/AAALaCH6sGTMRUVTjh8Tm8Jwa?dl=0) | [here](https://huggingface.co/speechbrain/asr-conformersmall-transformerlm-librispeech) | 2.49% | 6.10% |
 | [`recipes/LibriSpeech/ASR/transformer/hparams/transformer.yaml`](recipes/LibriSpeech/ASR/transformer/hparams/transformer.yaml) | [here](https://www.dropbox.com/sh/653kq8h2k87md4p/AAByAaAryXtQKpRzYtzV9ih5a?dl=0) | [here](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech) | 2.27% | 5.53% |
 | [`recipes/LibriSpeech/ASR/transformer/hparams/conformer_large.yaml`](recipes/LibriSpeech/ASR/transformer/hparams/conformer_large.yaml) | [here](https://www.dropbox.com/scl/fo/9we244tgdf47ay20hrdoz/AKnoqQ13nLwSv1ITeJEQ3wY?rlkey=05o5jiszr8rhj6dlprw87t2x4&st=u2odesyk&dl=0) | - | 2.01% | 4.52% |
 | [`recipes/LibriSpeech/ASR/transformer/hparams/branchformer_large.yaml`](recipes/LibriSpeech/ASR/transformer/hparams/branchformer_large.yaml) | [here](https://www.dropbox.com/scl/fo/qhtds5rrdvhhhjywa7ovw/AMiIL5YvQENw5JKVpzXlP5o?rlkey=hz8vlpy3qf9kcyfx0cox089e6&st=ufckv6tb&dl=0) | - | 2.04% | 4.12% |
 | [`recipes/LibriSpeech/ASR/transformer/hparams/hyperconformer_22M.yaml`](recipes/LibriSpeech/ASR/transformer/hparams/hyperconformer_22M.yaml) | [here](https://www.dropbox.com/sh/30xsmqj13jexzoh/AACvZNtX1Fsr0Wa1Z3C9rHLXa?dl=0) | - | 2.23% | 4.54% |
 | [`recipes/LibriSpeech/ASR/transformer/hparams/hyperconformer_8M.yaml`](recipes/LibriSpeech/ASR/transformer/hparams/hyperconformer_8M.yaml) | [here](https://www.dropbox.com/sh/8jc96avmivr8fke/AABrFEhtWy_3-Q7BHhkh0enwa?dl=0) | - | 2.55% | 6.61% |
 | [`recipes/LibriSpeech/ASR/transformer/hparams/hyperbranchformer_25M.yaml`](recipes/LibriSpeech/ASR/transformer/hparams/hyperbranchformer_25M.yaml) | - | - | 2.36% | 6.89% |
 | [`recipes/LibriSpeech/ASR/transformer/hparams/hyperbranchformer_13M.yaml`](recipes/LibriSpeech/ASR/transformer/hparams/hyperbranchformer_13M.yaml) | - | - | 2.54% | 6.58% |
 | [`recipes/LibriSpeech/ASR/transformer/hparams/train_hf_whisper.yaml`](recipes/LibriSpeech/ASR/transformer/hparams/train_hf_whisper.yaml) | - | - |  |
 | [`recipes/LibriSpeech/ASR/transformer/hparams/bayesspeech.yaml`](recipes/LibriSpeech/ASR/transformer/hparams/bayesspeech.yaml) | [here](https://www.dropbox.com/scl/fo/cdken4jqfj96ev1v84jxm/h?rlkey=25eu1ytgm5ac51zqj8p65zwxd&dl=0) | - | 2.84% | 6.27% |


### ASR-Transducers

| Model | Checkpoints | HuggingFace | Test_clean-WER | Test_other-WER |
| --------| --------| --------| --------| --------|
 | [`recipes/LibriSpeech/ASR/transducer/hparams/conformer_transducer.yaml`](recipes/LibriSpeech/ASR/transducer/hparams/conformer_transducer.yaml) | [here](https://www.dropbox.com/scl/fo/kl1eikmoauygwqcx8ok4r/AMkreKLzHtxPtqnoXzUerko?rlkey=juk374k210b76lbnblh7or95d&st=1ugwe9e3&dl=0) | - | 2.72% | 6.47% |


### ASR-CTC

| Model | Checkpoints | HuggingFace | Test_clean-WER | Test_other-WER |
| --------| --------| --------| --------| --------|
 | [`recipes/LibriSpeech/ASR/CTC/hparams/train_hf_wav2vec.yaml`](recipes/LibriSpeech/ASR/CTC/hparams/train_hf_wav2vec.yaml) | [here](https://www.dropbox.com/sh/qj2ps85g8oiicrj/AAAxlkQw5Pfo0M9EyHMi8iAra?dl=0) | [here](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech) | 1.65% | 3.67% |
 | [`recipes/LibriSpeech/ASR/CTC/hparams/train_hf_wav2vec_transformer_rescoring.yaml`](recipes/LibriSpeech/ASR/CTC/hparams/train_hf_wav2vec_transformer_rescoring.yaml) | [here](https://www.dropbox.com/sh/ijqalvre7mm08ng/AAD_hsN-8dBneUMMkELsOOxga?dl=0) | - | 1.57% | 3.37% |


### G2P

| Model | Checkpoints | HuggingFace | PER-Test |
| --------| --------| --------| --------|
 | [`recipes/LibriSpeech/G2P/hparams/hparams_g2p_rnn.yaml`](recipes/LibriSpeech/G2P/hparams/hparams_g2p_rnn.yaml) | [here](https://www.dropbox.com/sh/qmcl1obp8pxqaap/AAC3yXvjkfJ3mL-RKyAUxPdNa?dl=0) | - | 2.72% |
 | [`recipes/LibriSpeech/G2P/hparams/hparams_g2p_transformer.yaml`](recipes/LibriSpeech/G2P/hparams/hparams_g2p_transformer.yaml) | [here](https://www.dropbox.com/sh/zhrxg7anuhje7e8/AADTeJtdsja_wClkE2DsF9Ewa?dl=0) | [here](https://huggingface.co/speechbrain/soundchoice-g2p) | 2.89% |


### ASR-Seq2Seq

| Model | Checkpoints | HuggingFace | Test_clean-WER | Test_other-WER |
| --------| --------| --------| --------| --------|
 | [`recipes/LibriSpeech/ASR/seq2seq/hparams/train_BPE_5000.yaml`](recipes/LibriSpeech/ASR/seq2seq/hparams/train_BPE_5000.yaml) | [here](https://www.dropbox.com/sh/1ycv07gyxdq8hdl/AABUDYzza4SLYtY45RcGf2_0a?dl=0) | [here](https://huggingface.co/speechbrain/asr-crdnn-transformerlm-librispeech) | 2.89% | 8.09% |


## MEDIA Dataset

### ASR

| Model | Checkpoints | HuggingFace | Test-ChER | Test-CER |
| --------| --------| --------| --------| --------|
 | [`recipes/MEDIA/ASR/CTC/hparams/train_hf_wav2vec.yaml`](recipes/MEDIA/ASR/CTC/hparams/train_hf_wav2vec.yaml) | - | [here](https://huggingface.co/speechbrain/asr-wav2vec2-ctc-MEDIA) | 7.78% | 4.78% |


### SLU

| Model | Checkpoints | HuggingFace | Test-ChER | Test-CER | Test-CVER |
| --------| --------| --------| --------| --------| --------|
 | [`recipes/MEDIA/SLU/CTC/hparams/train_hf_wav2vec_full.yaml`](recipes/MEDIA/SLU/CTC/hparams/train_hf_wav2vec_full.yaml) | - | [here](https://huggingface.co/speechbrain/slu-wav2vec2-ctc-MEDIA-relax) | 7.46% | 20.10% | 31.41% |
 | [`recipes/MEDIA/SLU/CTC/hparams/train_hf_wav2vec_relax.yaml`](recipes/MEDIA/SLU/CTC/hparams/train_hf_wav2vec_relax.yaml) | - | [here](https://huggingface.co/speechbrain/slu-wav2vec2-ctc-MEDIA-full) | 7.78% | 24.88% | 35.77% |


## MultiWOZ Dataset

### Response-Generation

| Model | Checkpoints | HuggingFace | Test-PPL | Test_BLEU-4 |
| --------| --------| --------| --------| --------|
 | [`recipes/MultiWOZ/response_generation/gpt/hparams/train_gpt.yaml`](recipes/MultiWOZ/response_generation/gpt/hparams/train_gpt.yaml) | [here](https://www.dropbox.com/sh/vm8f5iavohr4zz9/AACrkOxXuxsrvJy4Cjpih9bQa?dl=0) | [here](https://huggingface.co/speechbrain/MultiWOZ-GPT-Response_Generation) | 4.01 | 2.54e-04 |
 | [`recipes/MultiWOZ/response_generation/llama2/hparams/train_llama2.yaml`](recipes/MultiWOZ/response_generation/llama2/hparams/train_llama2.yaml) | [here](https://www.dropbox.com/sh/d093vsje1d7ijj9/AAA-nHEd_MwNEFJfBGLmXxJra?dl=0) | [here](https://huggingface.co/speechbrain/MultiWOZ-Llama2-Response_Generation) | 2.90 | 7.45e-04 |


## REAL-M Dataset

### Sisnr-estimation

| Model | Checkpoints | HuggingFace | L1-Error |
| --------| --------| --------| --------|
 | [`recipes/REAL-M/sisnr-estimation/hparams/pool_sisnrestimator.yaml`](recipes/REAL-M/sisnr-estimation/hparams/pool_sisnrestimator.yaml) | [here](https://www.dropbox.com/sh/n55lm8i5z51pbm1/AABHfByOEy__UP_bmT4GJvSba?dl=0) | [here](https://huggingface.co/speechbrain/REAL-M-sisnr-estimator) | 1.71dB |


## RescueSpeech Dataset

### ASR+enhancement

| Model | Checkpoints | HuggingFace | SISNRi | SDRi | PESQ | STOI | WER |
| --------| --------| --------| --------| --------| --------| --------| --------|
 | [`recipes/RescueSpeech/ASR/noise-robust/hparams/robust_asr_16k.yaml`](recipes/RescueSpeech/ASR/noise-robust/hparams/robust_asr_16k.yaml) | [here](https://www.dropbox.com/sh/kqs2ld14fm20cxl/AACiobSLdNtXhm-4Y3IIbTeia?dl=0) | [here](https://huggingface.co/sangeet2020/noisy-whisper-resucespeech) | 7.482 | 8.011 | 2.083 | 0.854 | 45.29% |


## SLURP Dataset

### SLU

| Model | Checkpoints | HuggingFace | scenario-accuracy | action-accuracy | intent-accuracy |
| --------| --------| --------| --------| --------| --------|
 | [`recipes/SLURP/NLU/hparams/train.yaml`](recipes/SLURP/NLU/hparams/train.yaml) | [here](https://www.dropbox.com/scl/fo/c0rm2ja8oxus8q27om8ve/h?rlkey=irxzl1ea8g7e6ipk0vuc288zh&dl=0 ) | - | 90.81% | 88.29% | 87.28% |
 | [`recipes/SLURP/direct/hparams/train.yaml`](recipes/SLURP/direct/hparams/train.yaml) | [here](https://www.dropbox.com/scl/fo/c0rm2ja8oxus8q27om8ve/h?rlkey=irxzl1ea8g7e6ipk0vuc288zh&dl=0 ) | - | 81.73% | 77.11% | 75.05% |
 | [`recipes/SLURP/direct/hparams/train_with_wav2vec2.yaml`](recipes/SLURP/direct/hparams/train_with_wav2vec2.yaml) | [here](https://www.dropbox.com/scl/fo/c0rm2ja8oxus8q27om8ve/h?rlkey=irxzl1ea8g7e6ipk0vuc288zh&dl=0 ) | [here](https://huggingface.co/speechbrain/SLU-direct-SLURP-hubert-enc) | 91.24% | 88.47% | 87.55% |


## Switchboard Dataset

### ASR

| Model | Checkpoints | HuggingFace | Swbd-WER | Callhome-WER | Eval2000-WER |
| --------| --------| --------| --------| --------| --------|
 | [`recipes/Switchboard/ASR/CTC/hparams/train_with_wav2vec.yaml`](recipes/Switchboard/ASR/CTC/hparams/train_with_wav2vec.yaml) | - | [here](https://huggingface.co/speechbrain/asr-wav2vec2-switchboard) | 8.76% | 14.67% | 11.78% |
 | [`recipes/Switchboard/ASR/seq2seq/hparams/train_BPE_2000.yaml`](recipes/Switchboard/ASR/seq2seq/hparams/train_BPE_2000.yaml) | - | [here](https://huggingface.co/speechbrain/asr-crdnn-switchboard) | 16.90% | 25.12% | 20.71% |
 | [`recipes/Switchboard/ASR/transformer/hparams/transformer.yaml`](recipes/Switchboard/ASR/transformer/hparams/transformer.yaml) | - | [here](https://huggingface.co/speechbrain/asr-transformer-switchboard) | 9.80% | 17.89% | 13.94% |


## TIMIT Dataset

### ASR

| Model | Checkpoints | HuggingFace | Test-PER |
| --------| --------| --------| --------|
 | [`recipes/TIMIT/ASR/CTC/hparams/train.yaml`](recipes/TIMIT/ASR/CTC/hparams/train.yaml) | [here](https://www.dropbox.com/sh/059jnwdass8v45u/AADTjh5DYdYKuZsgH9HXGx0Sa?dl=0) | - | 14.78% |
 | [`recipes/TIMIT/ASR/seq2seq/hparams/train.yaml`](recipes/TIMIT/ASR/seq2seq/hparams/train.yaml) | [here](https://www.dropbox.com/sh/059jnwdass8v45u/AADTjh5DYdYKuZsgH9HXGx0Sa?dl=0) | - | 14.07% |
 | [`recipes/TIMIT/ASR/seq2seq/hparams/train_with_wav2vec2.yaml`](recipes/TIMIT/ASR/seq2seq/hparams/train_with_wav2vec2.yaml) | [here](https://www.dropbox.com/sh/059jnwdass8v45u/AADTjh5DYdYKuZsgH9HXGx0Sa?dl=0) | - | 8.04% |
 | [`recipes/TIMIT/ASR/transducer/hparams/train.yaml`](recipes/TIMIT/ASR/transducer/hparams/train.yaml) | [here](https://www.dropbox.com/sh/059jnwdass8v45u/AADTjh5DYdYKuZsgH9HXGx0Sa?dl=0) | - | 14.12% |
 | [`recipes/TIMIT/ASR/transducer/hparams/train_wav2vec.yaml`](recipes/TIMIT/ASR/transducer/hparams/train_wav2vec.yaml) | [here](https://www.dropbox.com/sh/059jnwdass8v45u/AADTjh5DYdYKuZsgH9HXGx0Sa?dl=0) | - | 8.91% |


## Tedlium2 Dataset

### ASR

| Model | Checkpoints | HuggingFace | Test-WER_No_LM |
| --------| --------| --------| --------|
 | [`recipes/Tedlium2/ASR/transformer/hparams/branchformer_large.yaml`](recipes/Tedlium2/ASR/transformer/hparams/branchformer_large.yaml) | [here](https://www.dropbox.com/sh/el523uofs96czfi/AADgTd838pKo2aR8fhqVOh-Oa?dl=0) | [here](https://huggingface.co/speechbrain/asr-branchformer-large-tedlium2) | 8.11% |


## UrbanSound8k Dataset

### SoundClassification

| Model | Checkpoints | HuggingFace | Accuracy |
| --------| --------| --------| --------|
 | [`recipes/UrbanSound8k/SoundClassification/hparams/train_ecapa_tdnn.yaml`](recipes/UrbanSound8k/SoundClassification/hparams/train_ecapa_tdnn.yaml) | [here](https://www.dropbox.com/sh/f61325e3w8h5yy2/AADm3E3PXFi1NYA7-QW3H-Ata?dl=0 ) | [here](https://huggingface.co/speechbrain/urbansound8k_ecapa) | 75.4% |


## Voicebank Dataset

### Dereverberation

| Model | Checkpoints | HuggingFace | PESQ |
| --------| --------| --------| --------|
 | [`recipes/Voicebank/dereverb/MetricGAN-U/hparams/train_dereverb.yaml`](recipes/Voicebank/dereverb/MetricGAN-U/hparams/train_dereverb.yaml) | [here](https://www.dropbox.com/sh/r94qn1f5lq9r3p7/AAAZfisBhhkS8cwpzy1O5ADUa?dl=0 ) | - | 2.07 |
 | [`recipes/Voicebank/dereverb/spectral_mask/hparams/train.yaml`](recipes/Voicebank/dereverb/spectral_mask/hparams/train.yaml) | [here](https://www.dropbox.com/sh/pw8aer8gcsrdbx7/AADknh7plHF5GBeTRK9VkIKga?dl=0 ) | - | 2.35 |


### ASR

| Model | Checkpoints | HuggingFace | Test-PER |
| --------| --------| --------| --------|
 | [`recipes/Voicebank/ASR/CTC/hparams/train.yaml`](recipes/Voicebank/ASR/CTC/hparams/train.yaml) | [here](https://www.dropbox.com/sh/w4j0auezgmmo005/AAAjKcoJMdLDp0Pqe3m7CLVaa?dl=0) | - | 10.12% |


### ASR+enhancement

| Model | Checkpoints | HuggingFace | PESQ | COVL | test-WER |
| --------| --------| --------| --------| --------| --------|
 | [`recipes/Voicebank/MTL/ASR_enhance/hparams/robust_asr.yaml`](recipes/Voicebank/MTL/ASR_enhance/hparams/robust_asr.yaml) | [here](https://www.dropbox.com/sh/azvcbvu8g5hpgm1/AACDc6QxtNMGZ3IoZLrDiU0Va?dl=0) | [here](https://huggingface.co/speechbrain/mtl-mimic-voicebank) | 3.05 | 3.74 | 2.80 |


### Enhancement

| Model | Checkpoints | HuggingFace | PESQ |
| --------| --------| --------| --------|
 | [`recipes/Voicebank/enhance/MetricGAN/hparams/train.yaml`](recipes/Voicebank/enhance/MetricGAN/hparams/train.yaml) | [here](https://www.dropbox.com/sh/n5q9vjn0yn1qvk6/AAB-S7i2-XzVm6ux0MrXCvqya?dl=0 ) | [here](https://huggingface.co/speechbrain/metricgan-plus-voicebank) | 3.15 |
 | [`recipes/Voicebank/enhance/SEGAN/hparams/train.yaml`](recipes/Voicebank/enhance/SEGAN/hparams/train.yaml) | [here](https://www.dropbox.com/sh/ez0folswdbqiad4/AADDasepeoCkneyiczjCcvaOa?dl=0 ) | - | 2.38 |
 | [`recipes/Voicebank/enhance/spectral_mask/hparams/train.yaml`](recipes/Voicebank/enhance/spectral_mask/hparams/train.yaml) | [here](https://www.dropbox.com/sh/n5q9vjn0yn1qvk6/AAB-S7i2-XzVm6ux0MrXCvqya?dl=0 ) | - | 2.65 |


## VoxCeleb Dataset

### Speaker_recognition

| Model | Checkpoints | HuggingFace | EER |
| --------| --------| --------| --------|
 | [`recipes/VoxCeleb/SpeakerRec/hparams/train_ecapa_tdnn.yaml`](recipes/VoxCeleb/SpeakerRec/hparams/train_ecapa_tdnn.yaml) | [here](https://www.dropbox.com/sh/ab1ma1lnmskedo8/AADsmgOLPdEjSF6wV3KyhNG1a?dl=0) | [here](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | 0.80% |
 | [`recipes/VoxCeleb/SpeakerRec/hparams/train_x_vectors.yaml`](recipes/VoxCeleb/SpeakerRec/hparams/train_x_vectors.yaml) | [here](https://www.dropbox.com/sh/ab1ma1lnmskedo8/AADsmgOLPdEjSF6wV3KyhNG1a?dl=0) | [here](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) | 3.23% |
 | [`recipes/VoxCeleb/SpeakerRec/hparams/train_resnet.yaml`](recipes/VoxCeleb/SpeakerRec/hparams/train_resnet.yaml) | [here](https://www.dropbox.com/sh/ab1ma1lnmskedo8/AADsmgOLPdEjSF6wV3KyhNG1a?dl=0) | [here](https://huggingface.co/speechbrain/spkrec-resnet-voxceleb) | 0.95% |


## VoxLingua107 Dataset

### Language-id

| Model | Checkpoints | HuggingFace | Accuracy |
| --------| --------| --------| --------|
 | [`recipes/VoxLingua107/lang_id/hparams/train_ecapa.yaml`](recipes/VoxLingua107/lang_id/hparams/train_ecapa.yaml) | [here](https://www.dropbox.com/sh/72gpuic5m4x8ztz/AAB5R-RVIEsXJtRH8SGkb_oCa?dl=0 ) | [here](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) | 93.3% |


## VoxPopuli Dataset

## WHAMandWHAMR Dataset

### Separation

| Model | Checkpoints | HuggingFace | SI-SNR |
| --------| --------| --------| --------|
 | [`recipes/WHAMandWHAMR/separation/hparams/sepformer-wham.yaml`](recipes/WHAMandWHAMR/separation/hparams/sepformer-wham.yaml) | [here](https://www.dropbox.com/sh/sfrgb3xivri432e/AACQodNmiDIKrB9vCeCFUDWUa?dl=0) | [here](https://huggingface.co/speechbrain/sepformer-whamr) | 16.5 |
 | [`recipes/WHAMandWHAMR/separation/hparams/sepformer-whamr.yaml`](recipes/WHAMandWHAMR/separation/hparams/sepformer-whamr.yaml) | [here](https://www.dropbox.com/sh/1sia32z01xbfgvu/AADditsqaTyfN3N6tzfEFPica?dl=0) | [here](https://huggingface.co/speechbrain/sepformer-wham) | 14.0 |


### Enhancement

| Model | Checkpoints | HuggingFace | SI-SNR | PESQ |
| --------| --------| --------| --------| --------|
 | [`recipes/WHAMandWHAMR/enhancement/hparams/sepformer-wham.yaml`](recipes/WHAMandWHAMR/enhancement/hparams/sepformer-wham.yaml) | [here](https://www.dropbox.com/sh/pxz2xbj76ijd5ci/AAD3c3dHyszk4oHJaa26K1_ha?dl=0) | [here](https://huggingface.co/speechbrain/sepformer-wham-enhancement) | 14.4 | 3.05 |
 | [`recipes/WHAMandWHAMR/enhancement/hparams/sepformer-whamr.yaml`](recipes/WHAMandWHAMR/enhancement/hparams/sepformer-whamr.yaml) | [here](https://www.dropbox.com/sh/kb0xrvi5k168ou2/AAAPB2U6HyyUT1gMoUH8gxQCa?dl=0) | [here](https://huggingface.co/speechbrain/sepformer-whamr-enhancement) | 10.6 | 2.84 |


## WSJ0Mix Dataset

### Separation (2mix)

| Model | Checkpoints | HuggingFace | SI-SNRi |
| --------| --------| --------| --------|
 | [`recipes/WSJ0Mix/separation/hparams/convtasnet.yaml`](recipes/WSJ0Mix/separation/hparams/convtasnet.yaml) | [here](https://www.dropbox.com/sh/hdpxj47signsay7/AABbDjGoyQesnFxjg0APxl7qa?dl=0) | - | 14.8dB |
 | [`recipes/WSJ0Mix/separation/hparams/dprnn.yaml`](recipes/WSJ0Mix/separation/hparams/dprnn.yaml) | [here](https://www.dropbox.com/sh/o8fohu5s07h4bnw/AADPNyR1E3Q4aRobg3FtXTwVa?dl=0) | - | 18.5dB |
 | [`recipes/WSJ0Mix/separation/hparams/resepformer.yaml`](recipes/WSJ0Mix/separation/hparams/resepformer.yaml) | [here](https://www.dropbox.com/sh/obnu87zhubn1iia/AAAbn_jzqzIfeqaE9YQ7ujyQa?dl=0) | [here](https://huggingface.co/speechbrain/resepformer-wsj02mix) | 18.6dB |
 | [`recipes/WSJ0Mix/separation/hparams/sepformer.yaml`](recipes/WSJ0Mix/separation/hparams/sepformer.yaml) | [here](https://www.dropbox.com/sh/9klsqadkhin6fw1/AADEqGdT98rcqxVgFlfki7Gva?dl=0 ) | [here](https://huggingface.co/speechbrain/sepformer-wsj02mix) | 22.4dB |
 | [`recipes/WSJ0Mix/separation/hparams/skim.yaml`](recipes/WSJ0Mix/separation/hparams/skim.yaml) | [here](https://www.dropbox.com/sh/zy0l5rc8abxdfp3/AAA2ngB74fugqpWXmjZo5v3wa?dl=0) | [here](https://huggingface.co/speechbrain/resepformer-wsj02mix ) | 18.1dB |


## ZaionEmotionDataset Dataset

### Emotion_Diarization

| Model | Checkpoints | HuggingFace | EDER |
| --------| --------| --------| --------|
 | [`recipes/ZaionEmotionDataset/emotion_diarization/hparams/train.yaml`](recipes/ZaionEmotionDataset/emotion_diarization/hparams/train.yaml) | [here](https://www.dropbox.com/sh/woudm1v31a7vyp5/AADAMxpQOXaxf8E_1hX202GJa?dl=0) | [here](https://huggingface.co/speechbrain/emotion-diarization-wavlm-large) | 30.2% |


## fluent-speech-commands Dataset

### SLU

| Model | Checkpoints | HuggingFace | Test-accuracy |
| --------| --------| --------| --------|
 | [`recipes/fluent-speech-commands/direct/hparams/train.yaml`](recipes/fluent-speech-commands/direct/hparams/train.yaml) | [here](https://www.dropbox.com/sh/wal9ap0go9f66qw/AADBVlGs_E2pEU4vYJgEe3Fba?dl=0) | - | 99.60% |


## timers-and-such Dataset

### SLU

| Model | Checkpoints | HuggingFace | Accuracy-Test_real |
| --------| --------| --------| --------|
 | [`recipes/timers-and-such/decoupled/hparams/train_TAS_LM.yaml`](recipes/timers-and-such/decoupled/hparams/train_TAS_LM.yaml) | [here](https://www.dropbox.com/sh/gmmum179ig9wz0x/AAAOSOi11yVymGXHp9LzYNrqa?dl=0) | - | 46.8% |
 | [`recipes/timers-and-such/direct/hparams/train.yaml`](recipes/timers-and-such/direct/hparams/train.yaml) | [here](https://www.dropbox.com/sh/gmmum179ig9wz0x/AAAOSOi11yVymGXHp9LzYNrqa?dl=0) | [here](https://huggingface.co/speechbrain/slu-timers-and-such-direct-librispeech-asr) | 77.5% |
 | [`recipes/timers-and-such/direct/hparams/train_with_wav2vec2.yaml`](recipes/timers-and-such/direct/hparams/train_with_wav2vec2.yaml) | [here](https://www.dropbox.com/sh/gmmum179ig9wz0x/AAAOSOi11yVymGXHp9LzYNrqa?dl=0) | - | 94.0% |
 | [`recipes/timers-and-such/multistage/hparams/train_TAS_LM.yaml`](recipes/timers-and-such/multistage/hparams/train_TAS_LM.yaml) | [here](https://www.dropbox.com/sh/gmmum179ig9wz0x/AAAOSOi11yVymGXHp9LzYNrqa?dl=0) | - | 72.6% |


