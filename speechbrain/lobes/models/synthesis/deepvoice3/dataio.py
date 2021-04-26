import speechbrain as sb

from speechbrain.dataio.encoder import TextEncoder
from torchaudio import transforms
from speechbrain.lobes.models.synthesis.dataio import audio_pipeline, spectrogram, resample, mel_spectrogram


import math
import torch
from torch.nn import functional as F


DB_BASE = 10.
DB_MULTIPLIER = 0.05


def text_encoder(max_input_len=128, tokens=None, takes="label"):
    """
    Configures and returns a text encoder function for use with the deepvoice3 model
    wrapped in a SpeechBrain pipeline function

    Arguments
    ---------
    max_input_len: int
        The maximum allowed length of an input sequence
    tokens: iterable
        a collection of tokens
    
    Returns
    -------
    item: DynamicItem
        A wrapped transformation function
    """

    encoder = TextEncoder()
    encoder.update_from_iterable(tokens)
    encoder.add_unk()
    encoder.add_bos_eos()

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides("text_sequences", "input_lengths", "text_positions")
    def f(label):
        text_sequence = encoder.encode_sequence_torch(label.upper())
        text_sequence_eos = encoder.append_eos_index(text_sequence)
        input_length = len(label)
        yield text_sequence_eos.long()
        yield input_length + 1
        yield torch.arange(1, input_length + 2, dtype=torch.long)
        
    return f


def downsample_spectrogram(takes, provides, downsample_step=4):
    """
    A pipeline function that downsamples a spectrogram

    Arguments
    ---------
    downsample_step: int
        the number of steps by which to downsample the target spectrograms
    
    Returns
    -------
    item: DynamicItem
        A wrapped transformation function        
    """
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(spectrogram):
        spectrogram = spectrogram[:, :, 0::downsample_step].contiguous()
        return spectrogram
    return f


def pad(takes, provides, length):
    """
    A pipeline function that pads an arbitrary
    tensor to the specified length

    Arguments
    ---------
    takes: str
        the source pipeline element
    provides: str
        the pipeline element to output
    length: int
        the length to which the tensor will be padded

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function
    """
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(x):
        return F.pad(x, (0, length - x.size(-1)))
    return f


def trim(takes, provides, threshold=-30.):
    """
    Returns a pipeline element that removes silence from the beginning and
    the end of the threshold

    Arguments
    ---------
    takes: str
        the source pipeline element
    provides: str
        the pipeline element to output    
    threshold: float
        the threshold, in decibels, below which samples will be considered
        as silence
    
    Returns
    -------
    item: DynamicItem
        A wrapped transformation function
    """

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(wav):
        threshold_amp = math.pow(DB_BASE, threshold * DB_MULTIPLIER)
        sound_pos = (wav > threshold_amp).nonzero()
        first, last = sound_pos[0], sound_pos[-1]
        return wav[first:last]
    return f


def done(takes="target_lengths", provides="done", 
         outputs_per_step=1, downsample_step=4):
    """
    Returns a generator of "done" tensors where 0 indicates
    that decoding is still in progress, 1 indicates that decoding
    is completed

    Arguments
    ---------
    takes: str
        the source pipeline element
    provides: str
        the pipeline element to output        
    outputs_per_step: int
        the number of outputs per step
    downsample_step: int
        the number of downsampling steps

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function
    """
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(target_lengths, padding=2):
        batch_size = target_lengths.size(0)
        done_lengths = target_lengths // outputs_per_step // downsample_step 
        done = torch.zeros(
            batch_size, done_lengths.max() + padding,
            device=target_lengths.device)
        ranges = torch.cat(
            batch_size 
            * [torch.arange(done.size(-1), device=target_lengths.device)
               .unsqueeze(0)])
        offsets = done_lengths.unsqueeze(-1)
        done[ranges >= offsets] = 1.
        return done.unsqueeze(-1)
    
    return f


def frame_positions(takes="mel", provides="frame_positions",
                    max_output_len=1024):
    """
    Returns a pipeline element that outputs frame positions within
    the spectrogram

    Arguments
    ---------
    max_output_len: int
        the maximum length of the spectrogram        

    Returns
    -------
    takes: str
        the source pipeline element
    provides: str
        the pipeline element to output        
    item: DynamicItem
        A wrapped transformation function
    """
    range_tensor = torch.arange(1, max_output_len+1)
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(mel):
        return range_tensor[:mel.size(-1)]
    return f


LOG_10 = math.log(10)

def normalize_spectrogram(takes, provides, min_level_db, ref_level_db, absolute=False):
    """
    Normalizes the spectrogram for DeepVoice3
    
    Arguments
    ---------
    takes: str
        the name of the input dynamic item
    provides: str
        the name of the output dynamic item
    min_level_db: float
        the minimum volume level, in decibels
    ref_level_db: float
        the reference decibel level
    absolute: bool
        where to apply the absolute value function
    
    Returns
    -------
    item: DynamicItem
        A wrapped transformation function
    """
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(linear):
        if absolute:
            linear = (linear**2).sum(dim=-1).sqrt()
        min_level = torch.tensor(math.exp(min_level_db / ref_level_db * LOG_10)).to(linear.device)
        linear_db = ref_level_db * torch.log10(torch.maximum(min_level, linear)) - ref_level_db
        normalized = torch.clip(
            (linear_db - min_level_db) / -min_level_db,
            min=0.,
            max=1.
        )
        return normalized

    return f


def target_lengths(takes='mel_raw', provides='target_lengths'):
    """
    Returns a transformation function that computes the target lengths

    Arguments
    ---------
    takes: str
        the source pipeline element
    provides: str
        the pipeline element to output      

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function           
    """
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(mel):
        nonzero = mel[:, 0, :].nonzero()    
        _, counts = nonzero[:, 0].unique(return_counts=True)
        return counts
    return f


def pad_to_length(tensor, length, value=0.):
    """
    Pads the last dimension of a tensor to the specified length,
    at the end
    
    Arguments
    ---------
    tensor: torch.Tensor
        the tensor
    length: int
        the target length along the last dimension
    value: float
        the value to pad it with

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function        
    """
    padding = length - tensor.size(-1)
    return F.pad(tensor, (0, padding), value=value)


def pad_spectrogram(takes="linear", provides="linear_pad",
                    outputs_per_step=1, downsample_step=5):
    """
    Pads the last dimension of a tensor to the specified length,
    at the end

    Arguments
    ---------
    takes: str
        the source pipeline element
    provides: str
        the pipeline element to output
    outputs_per_step: int
        the number of outputs per step
    downsample_step: int
        the number of downsampling steps

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function        
    
    """

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(tensor: torch.Tensor):
        padding = (
            outputs_per_step,
            downsample_step * outputs_per_step,
            0, 0)
        return F.pad(tensor, padding)        
    return f


def denormalize_spectrogram(takes="linear", provides="linear_denorm",
                            min_level_db=-100, ref_level_db=20):
    """
    Reverses spectrogram normalization and converts it
    to a waveform

    Arguments
    ---------
    takes: str
        the source pipeline element
    provides: str
        the pipeline element to output
    min_db_level: float
        the minimum decibel level
    ref_db_level: float
        the reference decibel level

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function    
    """
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(spectrogram):
        x = torch.clip(spectrogram, 0., 1.) * -min_level_db + min_level_db
        x += ref_level_db
        x = torch.pow(DB_BASE, x * DB_MULTIPLIER)
        return x
    return f


