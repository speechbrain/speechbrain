#### prepare kws

```
python prepare_kws.py
```

#### train

```
# commands should exist under data_folder and rirs_noises.zip should under rir_folder
python train.py hparams/crn.yaml --data_folder=/home/wangwei/work/corpus/kws/lanso/LS-ASR-data --rir_folder /home/wangwei/work/corpus/rir --data_parallel_backend
```

#### test

```

python test.py hparams/crn.yaml --data_folder=/home/wangwei/work/corpus/kws/lanso/LS-ASR-data --rir_folder /home/wangwei/work/corpus/rir
```

#### listen

```
python listen.py hparams/crn.yaml --data_folder=/home/wangwei/diskdata/corpus/kws/mobvoi_hotword_dataset/mobvoi_hotword_dataset --rir_folder E:/work/corpus/rir --device cpu --seed 19994
```



#### tensorboard

```
tensorboard --logdir logs --port 6006
ssh -L 8008:localhost:6006 wangwei@192.168.1.9
```

# keywords

| keyword  | samples | speakers |
| -------- | ------- | -------- |
| 柏丽柏丽 | 59863   | 500      |
| 小蓝小蓝 | 60119   | 501      |
| 安尼安尼 | 59735   | 498      |
| 物业物业 | 60114   | 501      |
| 德禄德禄 | 60087   | 501      |
| 管家管家 | 60187   | 500      |
| 曼格曼格 | 60187   | 501      |
| 小蓝管家 | 359     | 3        |



## target words

| keyword  | samples |
| -------- | ------- |
| 小蓝小蓝 | 60119   |
| 管家管家 | 59871   |
| 物业物业 | 60107   |
| unknown  | 240230  |



# Performance summary

[Command accuracy on mobvoihotwords]

| System | Valid ErrorRate | Test ErrorRate | seed |
|----------------- | ------------ |----------------- |----------------- |
| BIGRU(60 epoch) | 1.74e-3 | 2.91e-3 | 1988 |
| resnet(200 epoch) | 1.37e-3 | 2.86e-3 | 1999 |
| resnet+augment(200 epoch) | 8.05e-04 | 1.93e-3 | 19992 |
| CRN+augment | 5.97e-4 | 1.18e-3 | 19993 |
| crn_att | 1.19e-3 |  | 191 |
| crn_att+dropout(0.25 rnn) | 1.04e-3 | | 193 |
| crn_att+dropout | 9.60e-4 | 1.72e-3 | 194 |
| CRDNN |  |  | 197 |

+augment

| model           | Valdi ErrorRate | Test ErrorRate | seed  |                                                              |
| --------------- | --------------- | -------------- | ----- | ------------------------------------------------------------ |
| CRN             | 5.97e-4         | 1.18e-3        | 19993 |                                                              |
| CRN_GRU         | 3.63e-4         |                | 19994 |                                                              |
| CRN_GRU+dropout | 5.45e-4         | 9.26e-4        | 19997 |                                                              |
| tcresnet-m1     | 2.88e-3         |                | 19998 |                                                              |
| tcresnet-m3     | 2.93e-3         |                | 19999 |                                                              |
| crn_att         | 5.71e-4         | 1.01e-3        | 192   |                                                              |
| crn_att+dropout | 6.49e-4         | 1.13e-3        | 195   |                                                              |
| CRN3(BGRU)      | 4.67e-4         | 7.9e-4         | 19995 |                                                              |
| CRDNN           | 5.71e-4         |                | 198   | python train.py hparams/crdnn.yaml --data_folder=/home/wangwei/work/corpus/kws/mobvoi_hotword_dataset/mobvoi_hotword_dataset --rir_folder /home/wangwei/work/corpus/RIR --data_parallel_backend --seed 198 --apply_data_augmentation True --batch_size 64 |
| CRDNN_ATT       |                 |                | 199   |                                                              |



FNR：拒识率

FPR：虚警率

| FPR/FNR | 嗨小问          | 你好问问      |
| ------- | --------------- | ------------- |
| kaldi   | 0.00371/0.00376 | 0.0025/0.0026 |
| pytorch | 0.00590/0.00319 | 0.0078/0.0038 |
|         |                 |               |

CNN-LSTM

![image-20210428170830871](C:/Users/Administrator/AppData/Roaming/Typora/typora-user-images/image-20210428170830871.png)

CNN-GRU

![image-20210428170927322](C:/Users/Administrator/AppData/Roaming/Typora/typora-user-images/image-20210428170927322.png)