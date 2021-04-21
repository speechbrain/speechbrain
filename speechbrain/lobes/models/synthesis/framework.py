# TODO: Remove this file - it has been moved to pretrained.py
# TODO: (previously it did not exist)
import torch
from speechbrain.utils.data_pipeline import DataPipeline
from types import SimpleNamespace



class SpeechSynthesizer:
    INPUT_STATIC_KEYS = ['txt']
    OUTPUT_KEYS = ['wav']

    """
    A friendly wrapper for speech synthesis models

    Arguments
    ---------
    hparams
        Hyperparameters (from HyperPyYAML)
    encode_pipeline: list
        A pipeline of items to encode the text for the model. It should 
        take a single item called 'txt'
    decode_pipeline: list

    """
    def __init__(self, hparams, encode_pipeline, decode_pipeline):
        self.hparams = SimpleNamespace(**hparams)
        self.encode_pipeline = DataPipeline(
            static_data_keys=self.INPUT_STATIC_KEYS,
            dynamic_items=encode_pipeline,
            output_keys=self.hparams.model_input_keys
        )
        self.decode_pipeline = DataPipeline(
            static_data_keys=self.hparams.model_output_keys,
            dynamic_items=decode_pipeline,
            output_keys=self.OUTPUT_KEYS
        )

    def tts(self, text):
        """
        Computes the waveform for the provided example or batch
        of examples

        Arguments
        ---------
        text: str or List[str]
            the text to be translated into speech
        
        Returns
        -------
        a single waveform if a single example is provided - or
        a list of waveform tensors if multiple examples are provided
        """
        # Create a single batch if provided a single example
        single = isinstance(text, str)
        pipeline_input = {
            'txt': [text] if single else text
        }
        model_input = self.encode_pipeline(pipeline_input)
        model_output = self.compute_forward(model_input)
        decoded_output = self.decode_pipeline(model_output)
        waveform = decoded_output.get('wav')
        if waveform is None:
            raise ValueError("The output pipeline did not output a waveform")
        if single:
            waveform = waveform[0]
        return waveform

    def __call__(self, text):
        """
        Calls tts(text)
        """
        return self.tts(text)
        
    def compute_forward(self, data):
        """
        Computes the forward pass of the model. This method can be overridden
        in the implementation, if needed

        Arguments:
        ----------
        data
            the raw inputs to the model
        """
        return self.hparams.model(**data)

class TestModel:
    def __call__(self, input):
        return {'output': input + 1.}