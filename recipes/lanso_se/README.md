# VoiceBank Speech Enhancement with Spectral Mask
This recipe implements a speech enhancement system based on spectral mask
with the VoiceBank dataset.

!!Add downloading instructions!!

# How to run
python train.py hparams/train.yaml --data_parallel_backend



## train command

```
python train.py hparams/train_crn_dataset1.yaml --data_folder /home/wangwei/work/corpus/asr/MiniLibriSpeech --rir_folder /home/wangwei/work/corpus/RIR --data_parallel_backend
```



# Results

| Release | hyperparams file | Test WER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 | train.yaml | -.-- | Not Available | 1xV100 32GB |

| model | valid      | test | seed | cmd                                                          |
| ----- | ---------- | ---- | ---- | ------------------------------------------------------------ |
| crn   | 2.51/0.923 |      | 4247 | python train.py hparams/train_crn_dataset1.yaml --data_folder /home/wangwei/work/corpus/asr/MiniLibriSpeech --rir_folder /home/wangwei/work/corpus/RIR --data_parallel_backend |
| crn2  |            |      | 4248 |                                                              |
| crn3  | 2.45/0.918 |      | 4249 |                                                              |



# Training Time

About N for each epoch with a  TESLA V100.

