# Sound Event Detection DCASE2019 Task4

This folder is an attempt to solve the Task4 of the DCASE2019 using SpeechBrain toolkit.
The system is based on the baseline from 2019, created by Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah [1] but with much better performance.
The system uses a CRDNN model + a Mean Teacher self-supervised technique proposed by Curious AI Company [2]

These are the results achieved with this code with the configuration set in `train.yaml` compare to the baseline from the year 2019. System performance should be somewhat reproducible (although not guarantee with GPU). F-scores reported are macro averaged. Event-based F-score are computed with a 200ms collar for onsets and max(200ms, 0.2* event_length) for offsets. Segment-based F-scores are computed with a time resolution of 1 sec.

 <table class="table table-striped">
 <thead>
 <tr>
 <td colspan="3">F-score metrics (macro averaged)</td>
 </tr>
 </thead>
 <tbody>
 <tr>
 <td></td>
 <td colspan="2">Public evaluation 2019 (Youtube)</td>
 <td colspan="2">Validation 2019</td>
 </tr>
 <td></td>
 <td>Baseline</td>
 <td>Speechbrain</td>
 <td>Baseline</td>
 <td>Speechbrain</td>
 <tr>
 <td>Event-based</td>
 <td>29.0 %</td>
 <td><strong>42.5 %<strong></td>
 <td>23.7 %</td>
 <td><strong>37.8 %<strong></td>
 </tr>
 <tr>
 <td>Segment-based</td>
 <td> 58.54 %</td>
 <td><strong>62.7 %<strong></td>
 <td>55.2 %</td>
 <td><strong>58.8 %<strong></td>
 </tr>
 </tbody>
 </table>


## Dependencies

Given that speechbrain is already installed and you run this on a Python Environment (Python 3.8+),
these are the missing dependencies

youtube-dl >= 2019.4.30, dcase_util >= 0.2.5, sed-eval >= 0.2.1, pandas > =1.2.4, requests>=2.26.0, googledrivedownloader>=0.4
*pandas, requests, youtube-dl and googledrivedownloader for data download only*

`requirements.txt` contains the correct dependencies. To install the dependencies execute the following on the command-line:
```
pip install -r requirements.txt
```

## Dataset Download & Structure

The Domestic Environment Sound Event Detection dataset is composed of multiple subsets. Here is the step to download:

1. **(Public evaluation set: Youtube subset)** download at: [evaluation dataset](https://zenodo.org/record/3588172).(Real recordings)
[ Need to run the download script more than once if the all the files are not downloaded to video not existing on Youtube]
2. (Synthetic clips) download at : [synthetic_dataset](https://doi.org/10.5281/zenodo.2583796).
3. (Weak + Unlabel extraction) run
```
python train.py train.yaml
```
and make sure `download: 1` in `train.yaml`.

The download is tedious and can take several hours (+12 hours), be patient.
After the download the structure of the directory should be like (if not make the adjustments)

```
dataset root
└───metadata			              (directories containing the annotations files)
│   │
│   └───train			              (annotations for the training sets)
│   │     weak.tsv                    (weakly labeled training set list and annotations)
│   │     unlabel_in_domain.tsv       (unlabeled in domain training set list)
│   │     synthetic.tsv               (synthetic data training set list and annotations)
│   │
│   └───validation			          (annotations for the test set)
│   │     validation.tsv                (validation set list with strong labels)
│   │     test_2018.tsv                  (test set list with strong labels - DCASE 2018)
│   │     eval_2018.tsv                (eval set list with strong labels - DCASE 2018)
│   │
│   └───eval			              (annotations for the public eval set (Youtube in papers))
│         public.tsv
└───audio					          (directories where the audio files will be downloaded)
    └───train			              (audio files for the training sets)
    │   └───weak                      (weakly labeled training set)
    │   └───unlabel_in_domain         (unlabeled in domain training set)
    │   └───synthetic                 (synthetic data training set)
    │
    └───validation
    └───eval
        └───public
```

## Training + Evaluation
There are four files here:

* `requirements.txt`: to install requirements.
* `train.py`: the main code file, outlines the entire training process.
* `train.yaml`: the hyperparameters file, sets all parameters of execution.
* `download_data.py`: a file containing the data download process (from baseline [1]).
* `utils_dcase2019.py`: contains utils functions.
* `prepare_dcase2019_task4.py`: prepare the metadata into json file.
* `data_label_pipeline.py`: create Dynamic Datasets.
* `CRNN_baseline.py`: model.
* `CRDNN_custom.py`: model.

To initialize the training, run this following command:
```bash
python train.py train.yaml
```

This will skip download (if `download: 0` in `train.yaml`), prepare the metadata for the datasets, train de model and print out the final evaluation metrics using SED eval [3].

### References
 - [1] Turpault, Nicolas and Serizel, Romain and Parag Shah, Ankit and Salamon, Justin: Sound event detection in domestic environments with weakly labeled data and soundscape synthesis. DCASE 2019 Baseline. October, 2019. https://hal.inria.fr/hal-02160855
 - [2] Tarvainen, A. and Valpola, H., 2017.
 Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Advances in neural information processing systems (pp. 1195-1204).https://arxiv.org/pdf/1703.01780.pdf
 - [3] Mesaros, Annamaria and Heittola, Toni and Virtanen, Tuomas, 2016. Metrics for polyphonic sound event detection. Applied Sciences, 6(6):162.https://www.mdpi.com/2076-3417/6/6/162
