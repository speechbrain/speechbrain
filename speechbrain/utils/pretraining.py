"""Pre-training utilities.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
"""
import functools
from speechbrain.utils.parameter_transfer import torch_parameter_transfer


# I think it has to be a class because we have to call the method
# only after initializing the brain class (that performs the model init)
def load_after_init(model, path, loader=torch_parameter_transfer):
    """Load parameters from the specified file.

    Arguments
    ---------
    model : object
        The model to load parameters for
    path : path
        Path to file containing the pretrained parameters to load
    loader : function
        Function that takes an object and a path to a parameter file and loads
        the parameters from that file.
    """

    @functools.wraps(model.forward)
    def patched_forward(self, *args, **kwargs):
        out = self._saved_forward(*args, **kwargs)
        loader(self, path)
        if "init_params" in kwargs:
            kwargs["init_params"] = False
        out = self._saved_forward(*args, **kwargs)
        self.forward = self._saved_forward
        return out

    model._saved_forward = model.forward
    model.forward = patched_forward.__get__(model)
    return model
