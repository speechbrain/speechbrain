import torch
from speechbrain.lobes.models.synthesis.framework import SpeechSynthesizer
from speechbrain.utils.data_pipeline import takes, provides
from speechbrain.dataio.encoder import TextEncoder
from hyperpyyaml import load_hyperpyyaml


ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!-'

TEST_HPARAMS = '''
model: !new:speechbrain.lobes.models.synthesis.framework.TestModel
model_input_keys: ['input']
model_output_keys: ['output']
'''

def test_synthesizer():
    # Note: the unit test is done with a fake model
    encoder = TextEncoder()
    encoder.update_from_iterable(ALPHABET)
    encoder.add_unk()
    encoder.add_bos_eos()

    @takes('txt')
    @provides('txt_encoded')
    def encode_text(txt):
        return encoder.encode_sequence_torch(txt)
    
    @takes('txt_encoded')
    @provides('input')
    def model_input(txt_encoded):
        return torch.tensor([[1., 2., 3.]])

    @takes('output')
    @provides('wav')
    def decode_waveform(model_output):
        return model_output + torch.tensor([1., 2., 3.])

    synthesizer = SpeechSynthesizer(
        hparams=load_hyperpyyaml(TEST_HPARAMS),
        encode_pipeline=[
            encode_text,
            model_input
        ],
        decode_pipeline=[
            decode_waveform
        ]
    )
    output = synthesizer('test')    
    assert torch.isclose(output, torch.tensor([3., 5., 7.])).all()