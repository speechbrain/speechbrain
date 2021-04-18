import speechbrain as sb

from speechbrain.dataio.encoder import TextEncoder
from torchaudio import transforms
from speechbrain.lobes.models.synthesis.dataio import audio_pipeline, spectrogram, resample, mel_spectrogram


import math
import torch
from torch.nn import functional as F

def text_encoder(max_input_len=128, tokens=None):
    """
    Configures and returns a text encoder function for use with the deepvoice3 model
    wrapped in a SpeechBrain pipeline function

    Arguments
    ---------
    max_input_len
        the maximum allowed length of an input sequence
    tokens
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

    @sb.utils.data_pipeline.takes("label")
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
    downsample_step
        the number of steps by which to downsample the target spectrograms
    
    Returns
    -------
    item: DynamicItem
        A wrapped transformation function        
    """
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(spectrogram):
        spectrogram = spectrogram[:, 0::downsample_step].contiguous()
        return spectrogram
    return f


def pad(takes, provides, length):
    """
    A pipeline function that pads an arbitrary
    tensor to the specified length

    Arguments
    ---------
    takes
        the source pipeline element
    provides
        the pipeline element to output
    length
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


def trim(takes, provides, sample_rate=22050, trigger_level=9.,
         *args, **kwargs):
    vad = transforms.Vad(
        sample_rate=48000, trigger_level=10.)
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)    
    def f(wav):
        x = vad(wav.squeeze())
        x = vad(x.flip(0)).flip(0)
        return x
    return f


def done(outputs_per_step=1, downsample_step=4):
    """
    Returns a generator of "done" tensors where 0 indicates
    that decoding is still in progress, 1 indicates that decoding
    is completed

    Arguments
    ---------
    outputs_per_step: int
        the number of outputs per step
    downsample_step
        the number of downsampling steps

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function
    """
    @sb.utils.data_pipeline.takes("target_lengths")
    @sb.utils.data_pipeline.provides("done")
    def f(target_length):
        done_length = target_length // outputs_per_step // downsample_step + 2
        done = torch.zeros(done_length)
        done[-2:] = 1.
        return done.unsqueeze(-1)
    
    return f


def frame_positions(max_output_len=1024):
    """
    Returns a pipeline element that outputs frame positions within the spectrogram

    Arguments
    ---------
    max_output_len
        the maximum length of the spectrogram

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function
    """
    range_tensor = torch.arange(1, max_output_len+1)
    @sb.utils.data_pipeline.takes("mel")
    @sb.utils.data_pipeline.provides("frame_positions")
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



@sb.utils.data_pipeline.takes("mel_raw")
@sb.utils.data_pipeline.provides("target_lengths")
def target_lengths(mel):
    return mel.size(-1)


def pad_to_length(tensor: torch.Tensor, length: int, value: int=0.):
    """
    Pads the last dimension of a tensor to the specified length,
    at the end
    
    Arguments
    ---------
    tensor
        the tensor
    length
        the target length along the last dimension
    value
        the value to pad it with

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function        
    """
    padding = length - tensor.size(-1)
    return F.pad(tensor, (0, padding), value=value)


def pad_spectrogram(takes: str, provides: str, outputs_per_step: int, downsample_step: int):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(tensor: torch.Tensor):
        padding = (
            outputs_per_step,
            downsample_step * outputs_per_step,
            0, 0)
        return F.pad(tensor, padding)        
    return f

def build_encoder_pipeline(hparams):
    pipeline = [
        audio_pipeline,
        resample(
            orig_freq=hparams['source_sample_rate'],
            new_freq=hparams['sample_rate']),
        trim(takes="sig_resampled", provides="sig_trimmed"),
        mel_spectrogram(
            takes="sig_trimmed",
            provides="mel_raw",
            hop_length=hparams['hop_length'],
            n_mels=hparams['mel_dim'],
            n_fft=hparams['n_fft'],
            power=1,
            sample_rate=hparams['sample_rate']),
        normalize_spectrogram(
            takes="mel_raw",
            provides="mel_norm",
            min_level_db=hparams['min_level_db'],
            ref_level_db=hparams['ref_level_db']),
        pad_spectrogram(
            takes="mel_norm",
            provides="mel_pad",
            outputs_per_step=hparams["outputs_per_step"],
            downsample_step=hparams['mel_downsample_step']),
        downsample_spectrogram(
            takes="mel_pad",
            provides="mel",
            downsample_step=hparams['mel_downsample_step']),
        text_encoder(max_input_len=hparams['max_input_len'], tokens=hparams['tokens']),
        frame_positions(
            max_output_len=hparams['max_mel_len'] * hparams['mel_downsample_step']),
        spectrogram(
            n_fft=hparams['n_fft'],
            hop_length=hparams['hop_length'],
            power=1,
            takes="sig_trimmed",
            provides="linear_raw"),
        normalize_spectrogram(
            takes="linear_raw",
            provides="linear_norm",
            min_level_db=hparams['min_level_db'],
            ref_level_db=hparams['ref_level_db']),
        pad_spectrogram(
            takes="linear_norm",
            provides="linear",
            outputs_per_step=hparams["outputs_per_step"],
            downsample_step=hparams['mel_downsample_step']),
        done(downsample_step=hparams['mel_downsample_step'],
                outputs_per_step=hparams['outputs_per_step']),
        target_lengths
    ]
    return pipeline