# SpeechBrain-MOABB: A User-Friendly Benchmark for Neural EEG Tasks

This repository provides a set of recipes for processing electroencephalographic (EEG) signals based on the popular [Mother of all BCI Benchmarks (MOABB)](https://github.com/NeuroTechX/moabb), seamlessly integrated with SpeechBrain. The benchmark facilitates the integration of new models and their evaluation on all supported tasks. It not only offers an interface for easy model integration and testing but also proposes a fair and robust method for comparing different architectures.

For detailed information, please refer to [our paper](link_to_our_paper).

## ⚡ Datasets and Recipes

The benchmark leverages datasets supported by [MOABB](https://neurotechx.github.io/moabb/datasets.html). Specifically, it comes with recipes for the following datasets:

| Dataset ID | Task |
|------------|-------------|
|[BNCI2014001](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2014001.html#moabb.datasets.BNCI2014001) | Motor Imagery |
|[BNCI2014004](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2015004.html#moabb.datasets.BNCI2015004) | Motor Imagery |
|[BNCI2015001](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2015001.html#moabb.datasets.BNCI2015001) | Motor Imagery |  
|[Zhou2016](https://neurotechx.github.io/moabb/generated/moabb.datasets.Zhou2016.html#moabb.datasets.Zhou2016) | Motor Imagery | 
|[BNCI2014009](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2014009.html#moabb.datasets.BNCI2014009) | P300 | 
|[EPFLP300](https://neurotechx.github.io/moabb/generated/moabb.datasets.EPFLP300.html#moabb.datasets.EPFLP300) | P300 | 
|[Lee2019_ERP](https://neurotechx.github.io/moabb/generated/moabb.datasets.Lee2019_ERP.html#moabb.datasets.Lee2019_ERP) | P300 | 
|[bi2015a](https://neurotechx.github.io/moabb/generated/moabb.datasets.Lee2019_ERP.html#moabb.datasets.Lee2019_ERP) | P300 | 
|[Nakanishi2015](https://neurotechx.github.io/moabb/generated/moabb.datasets.Nakanishi2015.html#moabb.datasets.Nakanishi2015) | SSVEP | 

The EEG datasets are automatically downloaded when running the provided recipes. Furthermore, the code is designed to allow easy integration of any other dataset from MOABB, as well as the ability to plug and test various deep neural networks. The benchmark includes recipes for using the datasets mentioned above with popular models such as EEGNET, ShallowConvNet, Bioformer, and PodNet.

## 🛠️ Installation

To set up the benchmark, follow these steps:

1. Install SpeechBrain:
   ```shell
   pip install speechbrain
   ```

2. Clone the benchmark repository:
   ```shell
   git clone https://github.com/speechbrain/benchmarks/
   ```

3. Navigate to `<path-to-repository>/benchmarks/CL_MASR` in your file system, open a terminal, and run the following commands:

   ```shell
   pip install -r ../../requirements.txt    # Install base dependencies
   pip install -r extra-requirements.txt    # Install additional dependencies
   ```

   These commands will install the necessary dependencies for the benchmark, including both the base requirements and the additional requirements.

### Notes on MNE

The benchmark relies on MNE, which, by default, stores a config file at `$HOME/.mne/mne-python.json` and downloads data to `$HOME/mne-data`. However, in some cases, the home directory may not exist, have storage limitations, or be on a shared filesystem where data operations are restricted by the system admin.

To set up a different folder for MNE, follow these steps:

1. Open your bash shell and execute the following command to set the environment variable `_MNE_FAKE_HOME_DIR` to your preferred folder:

   ```bash
   export _MNE_FAKE_HOME_DIR='your/folder'
   ```

2. Launch a Python session and import the MNE module. This action will create a new configuration file at `your/folder/.mne/mne-python.json`.

3. Open the newly created file (`your/folder/.mne/mne-python.json`) and set the `MNE_DATA` and `MNE_DATASETS_BNCI_PATH` variables to the folders you want to use for MNE data and MOABB datasets, respectively.

By following these steps, you can ensure that MNE uses the specified folder for configuration and data storage.

## ▶️ Quickstart

Before running an experiment, ensure that you have installed the extra dependencies listed in the `extra_requirements.txt` file.

### Running a Single Experiment

To train a neural network for decoding single EEG trials, run the following code:

```bash
python train.py hparams/EEGNet_BNCI2014001.yaml --data_folder '/path/to/BNCI2014001'
```

Replace `hparams/EEGNet_BNCI2014001.yaml` with the desired hyperparameter file and `'path/to/BNCI2014001'` with the folder where data will be automatically downloaded.

Training employs different strategies: leave-one-session-out and leave-one-subject-out. These strategies are defined as follows:
* Leave-one-session-out (within-subject, cross-session, and session-agnostic decoders):
  For each subject, one session is reserved as a test set, and the remaining sessions are used for training neural networks.

* Leave-one-subject-out (cross-subject, cross-session, and subject-agnostic decoders):
  One subject is reserved as a test set, and the remaining subjects are used for training neural networks.

The yaml files contain hyperparameters selected as the best after hyperparameter tuning.

After training, aggregate and visualize the performance of all experiments with:
```bash
python parse_results.py results/MOABB/EEGNet_BNCI2014001/ test_metrics.pkl acc f1
```

To see the results on the validation set, use:

```bash
python parse_results.py results/MOABB/EEGNet_BNCI2014001/ test_metrics.pkl acc f1
```

### Hyperparameter Tuning

For proposing new models, performing hyperparameter tuning is essential. We support hyperparameter tuning with [Orion](https://orion.readthedocs.io/en/stable/).
To run hyperparameter tuning, follow the instructions in the [Add here instructions on how to do hparam tuning] section.

The default hyperparameter optimization uses the [Tree-structured Parzen Estimator (TPE) algorithm](https://orion.readthedocs.io/en/stable/user/algorithms.html#tpe) as it has shown the best performance on the addressed tasks. For more details, refer to our [paper](link_to_our_paper) on the validation of the proposed experimental protocol.

## 📈️ Results

Here are some results obtained with a leave-one-session-out strategy. Performance metrics were computed on each held-out session (stored in the metrics.pkl file) and reported for each held-out session as an average value ± standard deviation across subjects.

To ensure transparency and reproducibility, we release the output folder containing model checkpoints and training logs [here](add_link).

| Release | Task | Hyperparams file | Training strategy | Session | Key loaded model | Performance (test set) |  GPUs | 
|:-------------:|:-------------:|:---------------------------:|:---------------------------:|  -----:|-----:| -----:| :-----------:|
| 23-07-31 | Motor imagery | EEGNet_BNCI2014001.yaml | leave-one-session-out | session_E | 'acc'| 0.7465±0.0660 | 1xNVIDIA A100 (40 GB) |
| 23-07-31 | Motor imagery | EEGNet_BNCI2014001.yaml | leave-one-session-out | session_T | 'acc'| 0.7585±0.0710 | 1xNVIDIA A100 (40 GB) |
| 23-07-31 | P300 | EEGNet_EPFLP300.yaml | leave-one-session-out | session_1 | 'f1'| 0.6332±0.1146 | 1xNVIDIA A100 (40 GB) | 
| 23-07-31 | P300 | EEGNet_EPFLP300.yaml | leave-one-session-out | session_2 | 'f1'| 0.6566±0.0944 | 1xNVIDIA A100 (40 GB) | 
| 23-07-31 | P300 | EEGNet_EPFLP300.yaml | leave-one-session-out | session_3 | 'f1'| 0.6600±0.1242 | 1xNVIDIA A100 (40 GB) |
| 23-07-31 | P300 | EEGNet_EPFLP300.yaml | leave-one-session-out | session_4 | 'f1'| 0.6526±0.1218 | 1xNVIDIA A100 (40 GB) | 
| 23-07-31 | SSVEP | EEGNet_Lee2019_SSVEP.yaml | leave-one-session-out | session_1 | 'acc'| 0.9370±0.1170 | 1xNVIDIA A100 (40 GB) |
| 23-07-31 | SSVEP | EEGNet_Lee2019_SSVEP.yaml | leave-one-session-out | session_2 | 'acc'| 0.9287±0.1157 | 1xNVIDIA A100 (40 GB) | 

Note that the experiments runs with any GPU with memory >= 12 GB.

## 📧 Contact

For any questions or inquiries, feel free to reach out to [davide.borra2@unibo.it](mailto:davide.borra2@unibo.it).

## **Citing**

If you use the SpeechBrainMOABB benchmark, please cite:

[The link to the official paper will be available soon]

Please also cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
