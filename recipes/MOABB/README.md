# SpeechBrain-MOABB: A User-Friendly Benchmark for Neural EEG Tasks

This repository provides a set of recipes for processing electroencephalographic (EEG) signals based on the popular [Mother of all BCI Benchmarks (MOABB)](https://github.com/NeuroTechX/moabb), seamlessly integrated with SpeechBrain. The benchmark facilitates the integration of new models and their evaluation on all supported tasks. It not only offers an interface for easy model integration and testing but also proposes a fair and robust method for comparing different architectures.

For detailed information, please refer to [our paper](link_to_our_paper).

## ⚡ Datasets and Recipes

The benchmark leverages datasets supported by [MOABB](https://neurotechx.github.io/moabb/datasets.html). Specifically, it comes with recipes for the following datasets:

| Dataset ID | Task | nsbj | nsess |
|------------|-------------|-----|-----|
|[BNCI2014001](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2014001.html#moabb.datasets.BNCI2014001) | Motor Imagery | 9 | 2 |
|[BNCI2014004](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2015004.html#moabb.datasets.BNCI2015004) | Motor Imagery | - | - |
|[BNCI2015001](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2015001.html#moabb.datasets.BNCI2015001) | Motor Imagery | - | - |
|[BNCI2015004](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2015004.html#moabb.datasets.BNCI2015004) | Motor Imagery | - | - |
|[Zhou2016](https://neurotechx.github.io/moabb/generated/moabb.datasets.Zhou2016.html#moabb.datasets.Zhou2016) | Motor Imagery | - | - |
|[BNCI2014009](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2014009.html#moabb.datasets.BNCI2014009) | P300 | - | - |
|[EPFLP300](https://neurotechx.github.io/moabb/generated/moabb.datasets.EPFLP300.html#moabb.datasets.EPFLP300) | P300 | - | - |
|[Lee2019_ERP](https://neurotechx.github.io/moabb/generated/moabb.datasets.Lee2019_ERP.html#moabb.datasets.Lee2019_ERP) | P300 | - | - |
|[bi2015a](https://neurotechx.github.io/moabb/generated/moabb.datasets.Lee2019_ERP.html#moabb.datasets.Lee2019_ERP) | P300 | - | - |
|[Nakanishi2015](https://neurotechx.github.io/moabb/generated/moabb.datasets.Nakanishi2015.html#moabb.datasets.Nakanishi2015) | SSVEP | - | - |

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


## Training Strategies
EEG recordings involve recording brain activity from a subject using multiple EEG sensors placed on their head, resulting in a multi-channel signal (one for each sensor). These recordings can be performed while the subject is engaged in specific tasks, such as motor imagery, where they are asked to think about a particular movement. Multiple recordings, each involving the same subject undertaking the same task, are typically conducted. These recordings are referred to as *sessions*.

One of the distinctive features of EEG tasks compared to other popular machine learning tasks, such as speech processing or computer vision, is the relatively low amount of data available for each subject. Additionally, due to the cost of recording brain activity, the number of subjects is not extensive.

Normally, two common strategies are used during the training phase: Leave-One-Session-Out and Leave-One-Subject-Out cross-validation.

* **Leave-One-Session-Out**:
  For each subject, we reserve one session as a test set and use the remaining sessions for training neural networks. We thus train different neural networks, each excluding a different session. We repeat this process for all subjects and then average the performance to asses the final performance of our models.

* **Leave-One-Subject-Out**:
  In this challenging condition, we reserve one subject as the test set while training using the data from all the other subjects. This approach is challenging because each subject has a unique brain activity pattern, making it difficult to successfully leverage data from other subjects at the time of writing.



## ▶️ Quickstart

Before proceeding with the experiments, make sure that you have installed the additional dependencies listed in the `extra_requirements.txt` file. Please, read the content above as well.

### Training for a Specific Subject and Session

Let's now dive into how to train a model using data from a single subject and session. Follow the steps below to run this experiment:

```bash
python train.py hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder=eeg_data --cached_data_folder=eeg_pickled_data --target_subject_idx=0 --target_session_idx=0 --data_iterator_name=leave-one-session-out
```

In this example, we will train EEGNET for Motor Imagery using the BNCI2014001 dataset. Specifically, we will train the model using data from subject 0. The  data recorded in session 0 will be used for testing, while all the other sessions will be used for training.

The data will be automatically downloaded to the specified `data_folder`, and a cached version of the data will be stored in `cached_data_folder` for future reuse.

The results, including training logs and checkpoints, will be availabe in the output folder specified in the hparam file.


## Run a Training Experiment on a Given Dataset

To perform training experiments using the Leave-One-Subject-Out and Leave-One-Session-Out approaches, we need to train different models on various subjects and sessions. Ultimately, we need to average their performance to get more reliable results.

To simplify the process for our users, we have developed a convenient bash script called `run_experiment.sh`, which orchestrates the necessary loops.


To run an experiment, execute the following code:

```bash
./run_experiments.sh --hparams=hparams/MotorImagery/BNCI2014001/EEGNet.yaml \
   --data_folder=eeg_data \
   --cached_data_folder=eeg_pickled_data \
   --output_folder=results/MotorImagery/BNCI2014001/EEGNet \
   --nsbj=9 \
   --nsess=2 \
   --seed=1986 \
   --nruns=2 \
   --eval_metric=acc \
   --metric_file=valid_metrics.pkl \
   --do_leave_one_subject_out=false \
   --do_leave_one_session_out=true
```

In this example, the script will run the Leave-One-Session-Out training on the BNCI2014001 dataset for Motor Imagery using the EEGNet.yaml configuration file. The experiments will iterate over the 9 subjects and 2 sessions. Each experiment will be repeated 2 times (`--nruns=2`) with different initialization seeds. Conducting multiple experiments with various seeds and averaging their performance enhances the statistical significance of the results. The evaluation metric used here is accuracy, and the validation metrics will be stored in `valid_metrics.pkl`.


The experiment results will be available in the specified output folder. For the final aggregated performance, please refer to the `aggregated_performance.txt` file.

**Note:** The number of subjects (`--nsbj`) and sessions (`--nsess`) varies depending on the dataset used. Please, take a look at the dataset table above for knowing the number of subjects and sessions in each dataset.



## Run a training experiment on a given dataset
For either the Leave-One-Subject-Out and Leave-One-Session-Out, we need to train different models using different sujects and sessions. At the end we need to average their performance.
To make it easier for our users, we created a simple bash script that orchestrates the needed loops: run_experiment.sh

For instance, run the following code:

./run_experiments.sh --hparams=hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder=eeg_data --cached_data_folder=eeg_pickled_data \
   --output_folder=results/MotorImagery/BNCI2014001/EEGNet --nsbj=9 --nsess=2 --seed=1986 --nruns=2 --eval_metric=acc --metric_file=valid_metrics.pkl \
   --do_leave_one_subject_out=false --do_leave_one_session_out=true

This script will run leave_one_session_out training on the BNCI2014001 dataset for Motor Imagery using EEGNet.yaml. we will loop over the 9 subjects and 2 sessions.
We run each experiments 2 times (--nruns=2) with different initialization seeds. Running multiple experiments using different seeds and averaging their performance is 
a good practice to improve the significance of the results. The evaluation metric is accuracy and the validation metrics are stored in valid_metrics.pkl.

The results are available in the specified output folder. For the final aggregated performance, take a look at the aggregated_performance.txt file.

**Note:** The number of subjects (--nsbj) and sessions (--nsess) is dataset dependent. 


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
