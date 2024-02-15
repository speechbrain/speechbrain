# Semantic Discrete Audio Representations

This recipe includes scripts to train semantic discrete audio representations for the LJSpeech dataset (see https://arxiv.org/abs/2312.09747).

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
python train_<module>.py hparams/<module>/<quantizer>_<model>.yaml --data_folder <path-to-data-folder>
```

### Examples

```bash
python train_quantizer.py hparams/quantizer/kmeans_wavlm.yaml --data_folder data/LJSpeech-1.1
```

```bash
python train_dequantizer.py hparams/dequantizer/kmeans_wavlm.yaml --data_folder data/LJSpeech-1.1 \
--quantizer_path results/quantizer/kmeans_wavlm/0/save/CKPT+2024-02-15+09-15-35+00/quantizer.ckpt
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
