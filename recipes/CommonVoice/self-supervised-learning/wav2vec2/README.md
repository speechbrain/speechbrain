# wav2vec 2.0 pretraining with SpeechBrain and HuggingFace <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="drawing" width="40"/>
This folder contains the scripts to train a wav2vec2 based system using CommonVoice. It can be adapted to any dataset as long as you provide the csv or json files as with other recipes. No other adaptation will be required apart from controlling the sequence length to avoid out of memory issues.

# Requirements
The HuggingFace *transformers* library must be installed first.
`pip install -r extra_requirements.txt`

# Principle
The idea is extremely simple. <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="drawing" width="40"/> provides a wav2vec 2.0 loss calculation. In practice, it means that forwarding throughout their wav2vec 2.0 models returns the loss. Hence, we simply use this interface as a lobes wrapper in SpeechBrain so anyone can fully pretrain a wav2vec 2.0 model.

At a high level, the steps of this integration are:
1. Indicate a <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="drawing" width="40"/> repository that stores the wav2vec 2.0 config file. This is necessary to determine the architecture of the model that will be instantiated (see `wav2vec2_hub` in the yaml). You can browse all the existing HUggingFace architectures online and use them! In practice, SpeechBrain will download the configuration file corresponding (or load it locally), and instantiate in PyTorch the wav2vec 2.0 model.
2. Train it using our wrapper and this recipe.
3. Save it to be reused as a finetunable or frozen encoder with SpeechBrain recipes (as we already have for several task).

# Go !
Simply type:
`python train.py hparams/wav2vec2_base.yaml`

Do not forget to replace the `!PLACEHOLDER` variables in the yaml corresponding to your local path to the data.

# Advices
Training wav2vec 2.0 models is crazy w.r.t compute resources. For instance, this recipe only trains a BASE wav2vec 2.0 architecture, and it already requires from 20 to 32 Tesla V100 for 30 to 48 hours. Of course, you can scale this to your needs (e.g., you can work with 2 GPUs only), but it will take ages! Welcome to the wav2vec 2.0 world!

You will find different advices in the yaml, but just in case, here is a list of the most important ones:
- To train w2v2 model, we recommand to have the effective batch_size higher than 200 (batch_size * nb_gpu * gradient_accumulation). Examples are: 32 Tesla V100 32GB — 12 * 32 * 1.
- Do not train on sequences longer than 20s. This will blow your VRAM up.
- Set the `n_warmup_steps` steps in such a way that it corresponds to 10% of the total training steps. The number of steps correspond to the actual number of call to .backward w.r.t the batch size. You may want to compute it in advance, but in practice, you can estimate it once you ran for a entire epoch as the number of steps per epoch is reported in the log. It is currently set to work best with CommonVoice EN.
- Normalize your input signal (here we do sentence-wise normalization).

# Results and comparison with Fairseq
We compared our model to one trained with Fairseq on the exact same condition. Our model obtained better performance than the Fairseq implementation on three downstream tasks: speech recognition, emotion recognition and speaker verification. No worries, it works well. Our results will be similar to the ones obtained with the HuggingFace implementation, as they are equivalent.
