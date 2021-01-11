## Multi-teacher Knowledge Distillation for CTC/Att models
This is the implementation of multi-teacher distillation methods to
joint ctc-attention end-to-end ASR systems. The proposed approaches integrate
the error rate metric to the teacher selection rather than solely focusing on the observed losses.
This way, we directly distillate and optimize the student toward the relevant metric for speech recognition.
For details please refer to: https://arxiv.org/abs/2005.09310


### Training steps
To speed up student distillation from multiple teachers, we separate the whole procedure into
three parts: teacher model training, inference running on teacher models, student distillation.

#### 1. Teacher model training
Before doing distillation, we require finishing N teacher models training. Here, we propose to set N=10 as in the referenced paper.

Models training can be done in parallel using `experiment_teacher.py`.

Example:
```
python experiment_teacher.py hyperparams/teachers/tea0.yaml --data_folder /path-to/data_folder --seed 1234
```

#### 2. Run inference on all teacher models
This part run inference on all teacher models and store them on disk using `experiment_save_teachers.py`. It is only required that you setup the `tea_models_dir` variable corresponding to the path to a txt file. The latter txt file needs to contain 
a list of paths pointing to each teacher model.ckpt. We decided to work with a file so it can easily scale to hundreds of teachers. 

Example:
```
python experiment_save_teachers.py hyperparams/augment_CRDNN_save_teachers.yaml --data_folder /path-to/data_folder --seed 1234
```

#### 3. Student distillation
This is the main part for distillation using `experiment_kd.py`. Here, the variable `pretrain` might be used to use a pre-trained teacher as the student. Note that if set to `True`, a path to the corresponding `model.ckpt` must be given in `pretrain_st_dir` 

Example:
```
python experiment_kd.py hyperparams/augment_CRDNN.yaml --data_folder /path-to/data_folder --seed 1234
```

### Distillation strategies
There are three strategies in current version setting by the option `strategy` in `hyperparams/augment_CRDNN.yaml`.

- **average**: average losses of teachers when doing distillation.
- **best**: choosing the best teacher based on WER.
- **weighted**: assigning weights to teachers based on WER.
