import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.pretrained import SpeechSynthesizer
from speechbrain.utils.data_pipeline import takes, provides
from speechbrain.dataio.encoder import TextEncoder
from torch import nn

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!-'

TEST_HPARAMS = '''
model: !new:speechbrain.lobes.models.synthesis.framework.TestModel
model_input_keys: ['input']
model_output_keys: ['output']
'''

class TestModel(nn.Module):
    def forward(self, input):
        return input + 1.


def test_synthesizer():
    # Note: the unit test is done with a fake model
    encoder = TextEncoder()
    encoder.update_from_iterable(ALPHABET)
    encoder.add_unk()

    @takes('txt')
    @provides('txt_encoded')
    def encode_text(txt):
        return encoder.encode_sequence_torch(txt.upper())

    @takes('txt_encoded')
    @provides('input')
    def model_input(txt_encoded):
        print(txt_encoded)
        assert txt_encoded.size(-1) == 4
        return torch.tensor([[1., 2., 3.]])

    @takes('output')
    @provides('wav')
    def decode_waveform(model_output):
        return model_output + torch.tensor([1., 2., 3.])

    test_hparams = {
        'model': TestModel(),
        'encode_pipeline': {
            'batch': False,
            'steps': [
                encode_text,
                model_input
            ],
            'output_keys': ['input']
        },
        'decode_pipeline': {
            'steps': [
                decode_waveform
            ]
        },
        'model_input_keys': ['input'],
        'model_output_keys': ['output']
    }

    synthesizer = SpeechSynthesizer(
        hparams=test_hparams
    )
    output = synthesizer('test')
    assert torch.isclose(output, torch.tensor([3., 5., 7.])).all()