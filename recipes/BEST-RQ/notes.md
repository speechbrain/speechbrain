




## Table of results

### WER
Model & \# Params. & WER clean (w/lm) $\downarrow$  & WER other (w/lm) $\downarrow$ & Train Time & \% speed up & ASV $\downarrow$ & IC $\uparrow$ \\

\hline
Wav2Vec2 Base         & 90.9M & 16.26 (10.63) & 40.17 (30.83) &  16:14 & - \\
BEST-RQ               & 83.0M & 16.79 (10.79) & 38.09 (28.31) & ~~6:43 & 2.41x \\
% BRQ HyperConf         & 71.3M & 18.47 (11.59) & 41.80 (31.17) & ~~5:56 & 2.73x \\
% BRQ HyperBranch       & 46.5M & 32.65 (18.33) & 58.10 (42.90) & ~~4:43 & 3.44x \\

## Best-RQ HyperBase Test w/LM
100%|██████████████████████████████████████████████████████████████████████████████████████████| 655/655 [01:22<00:00,  7.99it/s]
speechbrain.utils.train_logger - Epoch loaded: 20 - test loss: 3.47e-01, test CER: 3.81, test WER: 10.79
speechbrain.utils.checkpoints - Loading a checkpoint from results/LibriSpeech/brqb/1000/save/CKPT+2023-12-02+16-30-11+00
100%|██████████████████████████████████████████████████████████████████████████████████████████| 735/735 [01:49<00:00,  6.71it/s]
speechbrain.utils.train_logger - Epoch loaded: 20 - test loss: 1.03, test CER: 12.27, test WER: 28.31

## Best-RQ HyperConformer Test w/LM
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 655/655 [01:13<00:00,  8.97it/s]
speechbrain.utils.train_logger - Epoch loaded: 20 - test loss: 3.72e-01, test CER: 4.22, test WER: 11.59
speechbrain.utils.checkpoints - Loading a checkpoint from results/LibriSpeech/brq_hyperconf/2000/save/CKPT+2023-12-02+16-38-45+00
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 735/735 [01:46<00:00,  6.90it/s]
speechbrain.utils.train_logger - Epoch loaded: 20 - test loss: 1.13, test CER: 13.92, test WER: 31.18

## Best-RQ HyperBranchformer Test w/LM
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 655/655 [02:02<00:00,  5.33it/s]
speechbrain.utils.train_logger - Epoch loaded: 20 - test loss: 4.70e-01, test CER: 7.26, test WER: 18.33
speechbrain.utils.checkpoints - Loading a checkpoint from results/LibriSpeech/brq_hyperbranch/2000/save/CKPT+2023-12-02+16-24-51+00
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 735/735 [03:12<00:00,  3.81it/s]
speechbrain.utils.train_logger - Epoch loaded: 20 - test loss: 1.20, test CER: 20.60, test WER: 42.90

## Wav2Vec2 Test w/LM
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 655/655 [01:54<00:00,  5.74it/s]
speechbrain.utils.train_logger - Epoch loaded: 20 - test loss: 3.16e-01, test CER: 3.80, test WER: 10.63
speechbrain.utils.checkpoints - Loading a checkpoint from results/LibriSpeech/w2v2/1000/save/CKPT+2023-12-02+23-18-44+00
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 735/735 [02:31<00:00,  4.84it/s]
speechbrain.utils.train_logger - Epoch loaded: 20 - test loss: 1.10, test CER: 13.52, test WER: 30.83
