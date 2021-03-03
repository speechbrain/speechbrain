import speechbrain as sb
import torch
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.encoder import TextEncoder
from typing import Collection


from datasets.common import audio_pipeline, mel_spectrogram, resample


#TODO: This is temporary. Add support for different characters for different languages
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!-'

def padded_positions(item_len, max_len):
    """
    :param max_len:  the maximum length of a sequence
    :param item_len: the total length pof the sequence
    """
    positions = torch.zeros(max_len)
    positions[:item_len] = torch.arange(1, item_len+1)
    return positions


def text_encoder(max_input_len=128, tokens=None):
    """
    Configures and returns a text encoder function for use with the deepvoice3 model
    wrapped in a SpeechBrain pipeline function

    :param max_input_len: the maximum allowed length of an input sequence
    :param tokens: a collection of tokens
    """

    encoder = TextEncoder()
    encoder.update_from_iterable(tokens)
    encoder.add_unk()
    encoder.add_bos_eos()

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("text_sequences", "input_lengths", "text_positions")
    def f(label):
        text_sequence = encoder.encode_label_torch(label)
        text_sequence_eos = encoder.append_eos_index(text_sequence)
        input_length = len(label)
        yield text_sequence_eos
        yield input_length
        yield padded_positions(item_len=input_length, max_len=max_input_len)
        
    return f


def frame_positions(max_output_len=128):
    """
    Returns a pipeline element that outputs frame positions within the spectrogram
    :param max_output_len: the maximum length of the spectrogram
    """
    range_tensor = torch.arange(1, max_output_len+1)
    @sb.utils.data_pipeline.provides("frame_positions")
    def f():
        return range_tensor
    return f


OUTPUT_KEYS = ["text_sequences", "mel", "input_lengths", "text_positions", "frame_positions"]


def data_prep(datasets: Collection[DynamicItemDataset], max_input_len=128, max_output_len=512, tokens=None,
              mel_dim: int=80, n_fft:int=512, sample_rate=22050):
    """
    Prepares one or more datasets for use with deepvoice.

    In order to be usable with the DeepVoice model, a dataset needs to contain
    the following keys

    'wav': a file path to a .wav file containing the utterance
    'label': The raw text of the label

    :param datasets: a collection or datasets
    :return: the original dataset enhanced
    """

    if not tokens:
        tokens = ALPHABET
    
    pipeline = [
        audio_pipeline,
        resample(new_freq=sample_rate),
        mel_spectrogram(takes="sig_resampled", n_mels=mel_dim, n_fft=n_fft),
        text_encoder(max_input_len=max_input_len, tokens=tokens),
        frame_positions(max_output_len=max_output_len)]

    for element in pipeline:
        sb.dataio.dataset.add_dynamic_item(datasets, element)
        sb.dataio.dataset.set_output_keys(datasets, OUTPUT_KEYS)
    return datasets