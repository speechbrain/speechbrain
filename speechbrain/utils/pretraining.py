""" Libriries implementing pre-training utilities.

Authors
 * Mirco Ravanelli 2020
"""
import torch


# I think it has to be a class because we have to call the method
# only after initializing the brain class (that performs the model init)
def PreTrainer(models, files):
    """Perform pre-training using the specified models and files.

    Arguments
    ---------
    models : list
        List of neural model objects.
    files : list
        List of files to use for pre-training the given models.
    """

    def __init__(self, models, files):
        super().__init__()
        self.models = models
        self.files = files

    def __call__(self):
        for model, file in zip(self.models, self.files):
            model.load_state_dict(torch.load(file), strict=True)
