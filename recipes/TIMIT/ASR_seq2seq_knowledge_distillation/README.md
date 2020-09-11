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
Before doing distillation, we require finishing 10 teacher models training.

Please go to directory `teacher_models_training`. Models training could be done in parallel.
using `experiment_teacher.py`.

Example:
```
python experiment_teacher.py hyperparams/tea0.yaml --data_folder /path-to/data_folder --seed 1234
```

#### 2. Run inference on all teacher models
This part run inference on all teacher models and store them on disk using `experiment_save_teachers.py`.

Example:
```
python experiment_save_teachers.py hyperparams/augment_CRDNN_save_teachers.yaml --data_folder /path-to/data_folder --seed 1234
```

#### 3. Student distillation
This is the main part for distillation using `experiment_kd.py`

Example:
```
python experiment_kd.py hyperparams/augment_CRDNN.yaml --data_folder /path-to/data_folder --seed 1234
```

### Distillation strategies
There are three strategies in current version setting by the option `strategy` in `hyperparams/augment_CRDNN.yaml`.

- **average**: average losses of teachers when doing distillation.
- **best**: choosing the best teacher based on WER.
- **weighted**: assigning weights to teachers based on WER.
