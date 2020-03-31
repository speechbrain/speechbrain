import torch
from speechbrain.utils.data_utils import load_extended_yaml


class VGG2_BLSTM_MLP(torch.nn.Module):
    """A VGG2 + Bi-directional LSTM + MLP model.

    Args:
        overrides: changes to the defaults in the yaml files

    Shape:
        - wav: [batch, time_steps] or [batch, channels, time_steps]
        - output: [batch, time_steps / hop, phonemes]

    Example:
        >>> import torch
        >>> inputs = torch.rand([10, 16000])
        >>> model = VGG2_BLSTM_MLP()
        >>> outputs = model(inputs)
        >>> outputs.shape
        torch.Size([10, 101, 40])

    """
    def __init__(self, **overrides):

        # This file specifies overall model parameters
        params_filename = 'recipes/CTC/TIMIT/VGG2_BLSTM_MLP/model_params.yaml'
        self.params = load_extended_yaml(open(params_filename), overrides)

        function_list = [
            'features', 'norm', 'VGG2', 'RNN', 'MLP', 'out', 'log_softmax',
        ]
        class_list = [self.params[fn] for fn in function_list]
        self.model = torch.nn.Sequential(*class_list)

    def forward(wav):
        return self.model(wav)
