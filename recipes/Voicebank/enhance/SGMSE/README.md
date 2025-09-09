# VoiceBank Speech Enhancement with SGMSE
This recipe implements a speech enhancement system based on the SGMSE architecture using the VoiceBank dataset (based on the paper: [https://arxiv.org/abs/2208.05830](https://arxiv.org/abs/2208.05830)).

## Results

Experiment Date | PESQ | SI-SDR | STOI
-|-|-|-
2025-07-24 | 2.78 | 17.8 | 95.7

You can find the full experiment folder (i.e., checkpoints, logs, etc) here:
https://www.dropbox.com/scl/fo/bi8sln2de6ep8nrv38jt5/ACWQAOAIsYSMyjhcu2ZSavc?rlkey=xtqlon9xjcy43ghncnlbtruii&st=sql8s5r8&dl=0

## How to Run
### Training

To train the SGMSE speech enhancement model, execute:

```bash
python recipes/Voicebank/enhance/SGMSE/train.py recipes/Voicebank/enhance/SGMSE/hparams.yaml
```

This will:

* Prepare the VoiceBank dataset automatically (if not already prepared).
* Train the model based on hyperparameters defined in `hparams.yaml`.
* Create a `run_name`, unique to each run.
* Store checkpoints, logs, and validation / testing samples in `output_dir/run_name` (specified within the `hparams.yaml` file).

### Resume Training from a previous run

Point --resume to the existing run directory (the folder that contains hyperparams.yaml and checkpoints):

```bash
python recipes/Voicebank/enhance/SGMSE/train.py --resume path/to/results/run_YYYY-MM-DD_HH-MM-SS
```

When --resume is provided:

*	The script loads hyperparams.yaml from the given run directory and uses that saved configuration.
*	Training continues from the latest checkpoint in that directory (if present), keeping the same run_name.
*	CLI overrides still work, but a new run_name is not generated.


### Inference (Speech Enhancement)
You can enhance single audio files or entire directories using a trained model:

* **Single-file enhancement:**

```bash
python recipes/Voicebank/enhance/SGMSE/enhancement.py --run_dir /path/to/trained_model noisy_audio.wav
```

* **Batch enhancement (whole directory):**

```bash
python recipes/Voicebank/enhance/SGMSE/enhancement.py --run_dir /path/to/trained_model /path/to/noisy_directory
```

Enhanced audio files will be stored in a newly created subdirectory specified in `inference_dir` within the `hparams.yaml` file, preserving the original filenames.

## Results and Outputs
During training, all results and model checkpoints are saved in:

```
<output_dir>/<run_name>/
```

During inference, enhanced audio outputs are saved in:

```
<output_dir>/<run_name>/<inference_dir>/
```

## About SpeechBrain
* Website: [https://speechbrain.github.io/](https://speechbrain.github.io/)
* Code: [https://github.com/speechbrain/speechbrain/](https://github.com/speechbrain/speechbrain/)
* HuggingFace: [https://huggingface.co/speechbrain/](https://huggingface.co/speechbrain/)

## Citing SGMSE
```bibtex
@article{richter2023speech,
  title={Speech enhancement and dereverberation with diffusion-based generative models},
  author={Richter, Julius and Welker, Simon and Lemercier, Jean-Marie and Lay, Bunlong and Gerkmann, Timo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={31},
  pages={2351--2364},
  year={2023},
  publisher={IEEE}
}
```
