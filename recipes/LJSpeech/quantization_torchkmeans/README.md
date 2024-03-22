# Semantic Discrete Audio Representations

This recipe includes scripts to train semantic discrete audio representations for the LJSpeech dataset
based on the technique described in https://arxiv.org/abs/2312.09747.
According to this paper, speaker information can be preserved by mapping discrete representations back
to continuous ones and training a vocoder on top of these.

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

Open a terminal and run:

```bash
pip install -r extra-requirements.txt
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Open a terminal and run:

```bash
python train_<module>.py hparams/<module>/<config>.yaml --data_folder <path-to-data-folder>
```

### Examples

```bash
python train_quantizer.py hparams/quantizer/wavlm.yaml --data_folder data/LJSpeech-1.1
```

```bash
python train_dequantizer.py hparams/dequantizer/wavlm.yaml --data_folder data/LJSpeech-1.1 \
--quantizer_path results/quantizer/wavlm/0/save/CKPT+2024-02-15+09-15-35+00/quantizer.ckpt
```

```bash
python train_decoder.py hparams/decoder/wavlm.yaml --data_folder data/LJSpeech-1.1 \
--quantizer_path results/quantizer/wavlm/0/save/CKPT+2024-02-15+09-15-35+00/quantizer.ckpt \
--dequantizer_path results/dequantizer/wavlm/0/save/CKPT+2024-02-15+11-16-24+00/dequantizer.ckpt
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
