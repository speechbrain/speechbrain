# SpeechBrain-MOABB: A User-Friendly Benchmark for Neural EEG Tasks

This repository provides a set of recipes for processing electroencephalographic (EEG) signals based on the popular [Mother of all BCI Benchmarks (MOABB)](https://github.com/NeuroTechX/moabb), seamlessly integrated with SpeechBrain. The benchmark facilitates the integration of new models and their evaluation on all supported tasks. It not only offers an interface for easy model integration and testing but also proposes a fair and robust method for comparing different architectures.

For detailed information, please refer to [The link to the official paper will be available soon].

## ⚡ Datasets and Recipes

The benchmark leverages datasets supported by [MOABB](https://neurotechx.github.io/moabb/datasets.html). Specifically, it comes with recipes for the following datasets:

[Link Text](#dataset-table)

| Dataset ID | Task | nsbj | nsess |
|------------|-------------|-----|-----|
|[BNCI2014001](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2014001.html#moabb.datasets.BNCI2014001) | Motor Imagery | 9 | 2 |
|[BNCI2014004](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2014004.html#moabb.datasets.BNCI2014004) | Motor Imagery | 9 | 5 |
|[BNCI2015001](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2015001.html#moabb.datasets.BNCI2015001) | Motor Imagery | 12 | 2 |
|[BNCI2015004](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2015004.html#moabb.datasets.BNCI2015004) | Motor Imagery | 9 | 2 |
|[Lee2019_MI](https://neurotechx.github.io/moabb/generated/moabb.datasets.Lee2019_MI.html#moabb.datasets.Lee2019_MI) | Motor Imagery | 54 | 2 |
|[Zhou2016](https://neurotechx.github.io/moabb/generated/moabb.datasets.Zhou2016.html#moabb.datasets.Zhou2016) | Motor Imagery | 4 | 3 |
|[BNCI2014009](https://neurotechx.github.io/moabb/generated/moabb.datasets.BNCI2014009.html#moabb.datasets.BNCI2014009) | P300 | 10 | 3 |
|[EPFLP300](https://neurotechx.github.io/moabb/generated/moabb.datasets.EPFLP300.html#moabb.datasets.EPFLP300) | P300 | 8 | 4 |
|[Lee2019_ERP](https://neurotechx.github.io/moabb/generated/moabb.datasets.Lee2019_ERP.html#moabb.datasets.Lee2019_ERP) | P300 | 54 | 2 |
|[bi2015a](https://neurotechx.github.io/moabb/generated/moabb.datasets.bi2015a.html#moabb.datasets.bi2015a) | P300 | 43 | 3 |
|[Lee2019_SSVEP](https://neurotechx.github.io/moabb/generated/moabb.datasets.Lee2019_SSVEP.html#moabb.datasets.Lee2019_SSVEP) | SSVEP | 54 | 2 |

The EEG datasets are automatically downloaded when running the provided recipes. Furthermore, the code is designed to allow easy integration of any other dataset from MOABB, as well as the ability to plug and test various deep neural networks. The benchmark includes recipes for using the datasets mentioned above with popular models such as:
- [EEGNET](https://arxiv.org/pdf/1611.08024.pdf)
- [ShallowConvNet](https://arxiv.org/pdf/1703.05051.pdf)
- [EEGConformer](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9991178)

Users can easily integrate their own PyTorch models into our benchmark by following the instructions provided in the [Incorporating Your Model](#incorporating-your-model) section below.

## 🛠️ Installation

To set up the benchmark, follow these steps:

1. Install SpeechBrain:
   ```shell
   pip install speechbrain
   ```

**Note:** Before the release of SpeechBrain v1.0, you'll need to clone and install the SpeechBrain repository as follows:
    ```shell
    git clone https://github.com/speechbrain/speechbrain/
    cd speechbrain
    pip install -r requirements.txt
    pip install -e .
    cd ..
    ```


2. Clone the benchmark repository:
   ```shell
   git clone https://github.com/speechbrain/benchmarks/
   ```

3. Navigate to `<path-to-repository>/benchmarks/MOABB` in your file system, open a terminal, and run the following commands:

   ```shell
   pip install -r ../../requirements.txt    # Install base dependencies
   pip install -r extra-requirements.txt    # Install additional dependencies
   ```

   These commands will install the necessary dependencies for the benchmark, including both the base requirements and the additional requirements.

### Notes on MNE

The benchmark relies on [MNE](https://mne.tools/stable/index.html), which, by default, stores a config file at `$HOME/.mne/mne-python.json` and downloads data to `$HOME/mne-data`. However, in some cases, the home directory may not exist, have storage limitations, or be on a shared filesystem where data operations are restricted by the system admin.

To set up a different folder for MNE, follow these steps:

1. Open your bash shell and execute the following command to set the environment variable `_MNE_FAKE_HOME_DIR` to your preferred folder:

   ```bash
   export _MNE_FAKE_HOME_DIR='your/folder'
   ```

2. Launch a Python session and import the MNE module. This action will create a new configuration file at `your/folder/.mne/mne-python.json`.

3. Open the newly created file (`your/folder/.mne/mne-python.json`) and set the `MNE_DATA` and `MNE_DATASETS_BNCI_PATH` variables to the folders you want to use for MNE data and MOABB datasets, respectively.

By following these steps, you can ensure that MNE uses the specified folder for configuration and data storage.


## Training Strategies
EEG recordings involve recording brain activity from a subject using multiple EEG sensors placed on their head, resulting in a multi-channel signal (one for each sensor).

These recordings can be performed while the subject is engaged in specific tasks, such as *motor imagery*, where they are asked to think about a particular movement.

Multiple recordings, each involving the same subject undertaking the same task, are typically conducted. These recordings are referred to as *sessions*.

One of the distinctive features of EEG tasks compared to other popular machine learning tasks, such as speech processing or computer vision, is the relatively low amount of data available for each subject. Additionally, due to the cost of recording brain activity, the number of subjects is not large.

Normally, two common strategies are used during the training phase: *Leave-One-Session-Out* and *Leave-One-Subject-Out* cross-validation.

* **Leave-One-Session-Out**:
  For each subject, we reserve one session as a test set and use the remaining sessions for training neural networks. We thus train different neural networks, each excluding a different session. We repeat this process for all subjects and then average the performance to asses the final performance of our models.

* **Leave-One-Subject-Out**:
  In this challenging condition, we reserve one subject as the test set while training using the data from all the other subjects. This approach is challenging because each subject has a unique brain activity pattern, making it difficult to successfully leverage data from other subjects.



## ▶️ Quickstart

**Note:** Before proceeding with the experiments, make sure that you have installed the additional dependencies listed in the `extra_requirements.txt` file. Please, read the content above as well.

### Training for a Specific Subject and Session

Let's now dive into how to train a model using data from a single subject and session. Follow the steps below to run this experiment:

```bash
python train.py hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder=eeg_data --cached_data_folder=eeg_pickled_data --target_subject_idx=0 --target_session_idx=1 --data_iterator_name=leave-one-session-out
```

In this example, we will train EEGNET for Motor Imagery using the BNCI2014001 dataset. Specifically, we will train the model using data from *subject 0*. The  data recorded in *session 1* of  *subject 0* will be used for testing, while all the other sessions will be used for training.

The data will be automatically downloaded to the specified `data_folder`, and a cached version of the data will be stored in `cached_data_folder` for future reuse.

Ensure that your local machine has internet access. If it does not, you can download the data in advance and store it in the specified data_folder.

The results, including training logs and checkpoints, will be available in the output folder specified in the hparam file.


### Run a Training Experiment on a Given Dataset

To train models using either the *Leave-One-Subject-Out* or *Leave-One-Session-Out* approach and then average their performance, we have developed a convenient bash script called `run_experiment.sh`. This script orchestrates the necessary loops for easy execution.

To run a training experiment, use the following command:

```bash
./run_experiments.sh --hparams hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder eeg_data --output_folder results/MotorImagery/BNCI2014001/EEGNet --nsbj 9 --nsess 2 --nruns 10 --train_mode leave-one-session-out --number_of_epochs 250 --device=cpu
```

This command will execute the `leave_one_session_out` training on the BNCI2014001 dataset for Motor Imagery using the EEGNet.yaml configuration.

The script will loop over 9 subjects and 2 sessions, running the experiment 10 times (--nruns 10) with different initialization seeds to ensure robustness.

Running multiple experiments with varied seeds and averaging their performance is a recommended practice to improve result significance.

The evaluation metric is accuracy, and the validation metrics are stored in `valid_metrics.pkl`.

The results of each experiment are saved in the specified output folder. To view the final aggregated performance, refer to the `aggregated_performance.txt` file.


**Default Values:**
- By default, the training modality is set to `leave_one_session_out`. If you prefer to use `leave_one_subject_out`, simply add the flag `--train_mode=leave_one_subject_out`.
- The default evaluation metric is accuracy (acc). If you wish to use F1 score instead, use the flag `--eval_metric=f1`.
- By default, the evaluation is conducted on the test set. To use the dev set instead, use the flag `--eval_set=dev`.
- Without specifying the `--seed flag`, a random seed is used.
- Beyond the flags expected by the `./run_experiments.sh` script, you can use additional flags to override any value declared in the hparam file. In the example above, we changed the number of epochs to 2.

**Note**: This script operates under the assumption that you are utilizing a Linux-based system. In this scenario, we offer a bash script instead of a Python script due to its inherent suitability for effectively orchestrating multiple training loops across various subjects and sessions.

**Important:** The number of subjects (`--nsbj`) and sessions (`--nsess`) is dataset dependent. Refer to the dataset [dataset table above](#link-to-dataset-table) for these details. When executing a training experiment on a different dataset or model, please modify both the hparam file and adjust the subject and session counts accordingly.

### Hyperparameter Tuning

Efficient hyperparameter tuning is paramount when introducing novel models or experimenting with diverse datasets. Our benchmark establishes a standardized protocol for hyperparameter tuning, utilizing [Orion](https://orion.readthedocs.io/en/stable/) to ensure fair model comparisons.

#### **Overview**

Hyperparameter tuning is orchestrated through the `./run_hparam_optimization.sh` script, which oversees the execution of multiple hyperparameter trials via `run_experiments.sh`. This script supports leave-one-subject-out and leave-one-session-out training.

Please keep in mind the following points:
- In certain scenarios, you may find it advantageous to retain separate experiment folders for each hyperparameter trial. You can achieve this by using the `--store_all True` flag. Conversely, setting it to false will condense results within a singular folder, a space-saving measure.
- The script effectively manages all essential phases for executing multi-step hyperparameter tuning. It further assesses the final performance on the test set using the optimal hyperparameters, with performance being averaged across `--nruns_eval` iterations to enhance result significance.

#### **Incorporating Orion Flags in Hparam Files**

The script assumes that Orion flags are directly included in the specified YAML hparam file using comments. To optimize, for instance, the dropout parameter within a defined range, you need to have the following line in the YAML file:

```yaml
dropout: 0.1748  # @orion_step1: --dropout~"uniform(0.0, 0.5)"
```

#### **Multi-Step Hyperparameter Optimization**

Our method supports multi-step hyperparameter optimization.

In practice, we can optimize a subset of hyperparameters while keeping the others fixed.

After finding their optimal values, we utilize them as a foundation for optimizing another set of hyperparameters.

This approach has consistently demonstrated superior results in our benchmark, especially when distinguishing between training and architectural hyperparameters versus data augmentation hyperparameters.

We thus propose a two-phase hyperparameter optimization strategy: in phase 1, we optimize hyperparameters related to the neural network architecture, while in phase 2, our focus shifts exclusively to data augmentation hyperparameters.


To optimize a hyperparameter in a second step, follow this syntax in the YAML file:

```yaml
snr_white_low: 9.1 # @orion_step2: --snr_white_low~"uniform(0.0, 15, precision=2)"
```

Users have the flexibility to define multiple optimization steps based on their experimental protocol, although two steps, as recommended, often suffice.

#### **Workflow of the Script**

The script operates as follows:

1. Scans the specified hparam file for Orion flags.
2. Executes hyperparameter tuning using the `orion-hunt` command.
3. Captures and saves the best hyperparameters for reference via `torch-info`.
4. Continues until flags like `@orion_step<stepid>` are encountered in the YAML file.

#### **Running Hyperparameter Optimization**

You can conduct hyperparameter optimization with commands similar to the following:

```bash
./run_hparam_optimization.sh --exp_name 'EEGNet_BNCI2014001_hopt' \
                             --output_folder results/MotorImagery/BNCI2014001/EEGNet/hopt \
                             --data_folder eeg_data/ \
                             --hparams hparams/MotorImagery/BNCI2014001/EEGNet.yaml \
                             --nsbj 9 --nsess 2 \
                             --nruns 1 \
                             --nruns_eval 10 \
                             --eval_metric acc \
                             --train_mode leave-one-session-out \
                             --exp_max_trials 50
```

Note that hyperparameter tuning may take several hours (or days) depending on the model complexity and dataset.

As evident from the example, you need to configure the hyperparameter file, specify the number of subjects (nsbj), and set the number of sessions (nsess).

The [table above](#link-to-dataset-table) provides these values for each compatible dataset.

When it comes to training the model utilizing the leave_one_subject_out approach, simply employ the `--train_mode leave-one-subject-out` flag.

By default trainings are performed on gpu. However, in case you do not have any gpu available on your machine, you can train models on cpu by specifying the `--device cpu` flag.

**Note:**
- To monitor the status of the hyperparameter optimization, simply enter the following command: `orion status --all`. Ensure that you have added the necessary variables required by orion to your bash environment. You can achieve this by executing the following code within your terminal:

```bash
export ORION_DB_ADDRESS=results/MotorImagery/BNCI2014001/EEGNet/hopt/EEGNet_BNCI2014001_hopt.pkl
export ORION_DB_TYPE=pickleddb
```

Please note that the value of the `ORION_DB_ADDRESS` variable will vary depending on the experiment. Adjust it accordingly.

- If needed, you can interrupt the code at any point, and it will resume from the last successfully completed trial.

#### **Output Structure**

Results are saved within the specified output folder (`--output_folder`):

- The optimal hyperparameters are stored in `best_hparams.yaml`.
- The outcomes of individual optimization steps are stored within the subfolders `step1` and `step2`. When the `--store_all True` flag is employed, all hyperparameter trials are saved within the `exp` folder, each contained in subfolders with random names.
- To circumvent the generation of excessive files and folders within the `exp` directory, which can be an issue on certain HPC clusters due to file quantity restrictions, consider activating the `--compress_exp True` option.
- The "best" subfolder contains performance metrics on test sets using the best hyperparameters. Refer to `aggregated_performance.txt` for averaged results across multiple runs.

#### **Model Comparison**

Our protocol ensures a model comparison that is as fair as possible. All reported results reported below are achieved with the same hyperparameter tuning methodology, enabling fair assessments across diverse models.

For further details on arguments and customization options, consult `./run_hparam_optimization.sh`.

#### **Additional Notes:**

- The quantities of subjects (`--nsbj`) and sessions (`--nsess`) are dataset-dependent. Please consult the [table above](#link-to-dataset-table) for this information. When conducting a hyperparameter optimization experiment using an alternative dataset or model, kindly adjust both the hparam file and the subject/session counts accordingly.

- If you intend to perform multiple repetitions of the same hparam optimization, it is necessary to modify the `--exp_name`.

- This script is designed for a Linux-based system. In this context, we provide a bash script instead of a Python script due to its natural ability of orchestrating diverse training loops across various subjects and sessions.


## Incorporating Your Model
[Link Text](#incorporating-your-model)

Let's bow assume you've designed a neural network in PyTorch and wish to integrate it into our benchmark.
 we've streamlined the process for your convenience. Follow these steps:

1. Write your model's code in a Python library saved in `benchmarks/MOABB/models` (e.g., `benchmarks/MOABB/models/my_model.py`). Ensure that your model is compatible with the EGG task, considering varying input channels and variable-length inputs across different datasets.

2. Create a YAML file for each dataset you want to experiment with. Thankfully, you don't have to start from scratch. For example, if you're working with BNCI2014001 (Motor Imagery/), copy `benchmarks/MOABB/hparams/MotorImagery/BNCI2014001/EEGNet.yaml` and save it in the same folder with a different name (e.g., `my_model.yaml`).

3. Edit the relevant section of your `my_model.yaml`. Redefine the `model:` to reference your custom model (e.g., `model: !new:models.my_model.my_model`).

4. Ensure you include the hyperparameters specific to your model, along with the ranges you'd like to explore during hyperparameter tuning.

5. Now, follow the instructions above to run an experiment and perform the necessary hyperparameter tuning.

**Note**: If you're not familiar with YAML, you can refer to our [HyperPyYAML tutorial](https://speechbrain.github.io/tutorial_basics.html) on the SpeechBrain website for guidance.



## Incorporating Your Model
[Link Text](#incorporating-your-model)

Let's now assume that you designed your fancy neural network in pytorch and you would like to plug it in our benchmark.
You're in luck because we've made this step as simple as possible for you!

Here are the steps you should follow:

1- Write the code of your model in a python library saved in `benchmarks/MOABB/models` (e.g., `benchmarks/MOABB/models/my_model.py`). Note that you have to design the model such that it is compliant with the EGG task (e.g, we have might have different input channels in different datasets and the inputs are of variable length).
2- You have now to create a yaml file for each dataset you would like to try. Fortunately, you don't have to do it from scratch.  For instance, if you want to run your model into with
`BNCI2014001` (MotorImagery), just copy  `benchmarks/MOABB/hparams/MotorImagery/BNCI2014001/EEGNet.yaml` and save it in the same folder with another name (e.g., `my_model.yaml`).
3- Now, you can just edit the relevant part of your my_model.yaml. In particular, you have to redefine the `model:` such that you fetch your own model (e.g, `model: !new:models.my_model.my_model`).
4- Make sure you add the hyperparameters specific to your model along with the ranges you would like to explore in the hparam tuning phase.
5- At this point, you can follow the instructions above for runnning and experiment and the needed hparam tuning phase.

**Note**: If you feel unfamiliar with YAML, you can take a look at our [HyperPyYAML tutorial](https://speechbrain.github.io/tutorial_basics.html) in the speechbrain website.



## 📈️ Results

Here are some results obtained with a leave-one-session-out strategy.

Performance metrics were computed on each held-out session (stored in the metrics.pkl file) and reported averaged across sessions and subjects as an average value ± standard deviation across 10 random seeds.

To ensure transparency and reproducibility, we release the output folder containing model checkpoints and training logs [here](add_link).

| Release | Task | Hyperparams file | Training strategy | Key loaded model | Performance (test set) |  GPUs |
|:-------------:|:-------------:|:---------------------------:|:---------------------------:|  -----:| -----:| :-----------:|
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2014001/EEGNet.yaml | leave-one-session-out |  'acc'| 0.731559±0.003888 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2014001/ShallowConvNet.yaml | leave-one-session-out |  'acc'| 0.695795±0.003748 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2014001/EEGConformer.yaml | leave-one-session-out |  'acc'| 0.675810±0.006926 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2014004/EEGNet.yaml | leave-one-session-out |  'acc'| 0.812062±0.001888 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2014004/ShallowConvNet.yaml | leave-one-session-out |  'acc'| 0.784813±0.003038 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2015001/EEGNet.yaml | leave-one-session-out |  'acc'| 0.810646±0.002113 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2015001/ShallowConvNet.yaml | leave-one-session-out |  'acc'| 0.828646±0.003781 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2015001/EEGConformer.yaml | leave-one-session-out |  'acc'| 0.751688±0.009589 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2015004/EEGNet.yaml | leave-one-session-out |  'acc'| 0.591925±0.011688 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2015004/ShallowConvNet.yaml | leave-one-session-out |  'acc'| 0.523690±0.012165 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2015004/EEGConformer.yaml | leave-one-session-out |  'acc'| 0.597778±0.008762 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Lee2019_MI/EEGNet.yaml | leave-one-session-out |  'acc'| 0.694278±0.003121 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Lee2019_MI/ShallowConvNet.yaml | leave-one-session-out |  'acc'| 0.657500±0.004488 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Lee2019_MI/EEGConformer.yaml | leave-one-session-out |  'acc'| 0.651333±0.008495 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Zhou2016/EEGNet.yaml | leave-one-session-out |  'acc'| 0.843619±0.005637 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Zhou2016/ShallowConvNet.yaml | leave-one-session-out |  'acc'| 0.826854±0.005277 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Zhou2016/EEGConformer.yaml | leave-one-session-out |  'acc'| 0.839601±0.005769 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | P300 | /P300/EPFLP300/EEGNet.yaml | leave-one-session-out |  'f1'| 0.634613±0.003840 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | P300 | /P300/BNCI2014009/EEGNet.yaml | leave-one-session-out |  'f1'| 0.754958±0.001643 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | SSVEP | /SSVEP/Lee2019_SSVEP/EEGNet.yaml | leave-one-session-out |  'acc'| 0.916148±0.002436 | 1xNVIDIA V100 (16 GB) |

Note that the experiments run with any GPU with memory >= 12 GB.

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
