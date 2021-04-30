import speechbrain as sb
from torchaudio import transforms
import math
from scipy.signal import firwin, lfilter
import torch
import numpy as np
from torch.nn import functional as F
from datasets.vctk import VCTK
from speechbrain.dataio.dataset import DynamicItemDataset


def mulaw_quantize(sig,mu):
    # mulaw transform
    product = mu*sig.abs()
    mulaw = sig.sign() * product.log1p() / np.log1p(mu)

    # mulaw quantize
    mulaw_quantize = ((mulaw + 1) / 2 * mu).long()

    return mulaw_quantize

def start_and_end_indices(quantized, silence_threshold=2):

    for start in range(quantized.size(0)):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size(0) - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end

def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)

@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("signal")
def read_wav(file_name: str):
    """
    A pipeline function that reads a single file and reads
    audio from it into a tensor
    """
    sig = sb.dataio.dataio.read_audio(file_name)
    return sig

def prepare_signal(takes, provides, orig_freq, fs, threshold, cutoff,
    mu, silence_threshold):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(sig):

        sig_resampled = transforms.Resample(orig_freq=orig_freq,new_freq=fs)(sig)

        # remove leading and trailing silence
        # amp = 10**(0.05*dB)

        threshold_amp = math.pow(10.0,threshold*0.05)
        sound_pos = (sig_resampled.abs() > threshold_amp).nonzero()
        first, last = sound_pos[0], sound_pos[-1]
        sig_trimmed = sig_resampled[first:last]

        # low-cut filter
        #https://github.com/kan-bayashi/PytorchWaveNetVocoder
        if cutoff > 0.0:
            nyquist = fs // 2
            norm_cutoff = cutoff / nyquist 
            fil = firwin(255, norm_cutoff, pass_zero=False)
            lcf_x = lfilter(fil, 1, sig_trimmed)
            sig_lowcut = torch.from_numpy(lcf_x)
        else:
            sig_lowcut = sig_trimmed
    
        # trim mulaw domain silence
        out = mulaw_quantize(sig_lowcut, mu-1)
        start, end = start_and_end_indices(out, silence_threshold)
        signal_cut = sig_lowcut[start:end]

        return signal_cut.float()
    return f

def prepare_data(takes, provides, fs, n_mels, hop_length, n_fft, 
    min_level_db, ref_level_db, cin_channels, cin_pad, max_t_sec,
    max_t_steps, num_classes):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(sig):

        mel_raw = transforms.MelSpectrogram(sample_rate=fs, n_mels=n_mels, 
        hop_length=hop_length, n_fft = n_fft, power=1)(sig)

        sig_mulaw = mulaw_quantize(sig,255)
 
        # adjust time resolution between audio and mel-spectrogram
        constant_values = mulaw_quantize(torch.tensor(0.0), 255)
        l, r = (0,n_fft)
        if l > 0 or r > 0:
            sig_padded = F.pad(sig_mulaw, (l, r), mode="constant", value=constant_values)
        N = mel_raw.size(1)

        assert len(sig_padded) >= N * hop_length

        # normalize mel
        LOG_10 = math.log(10)
        absolute=False
        if absolute:
            linear = (mel_raw**2).sum(dim=-1).sqrt()
        else:
            linear=mel_raw
        min_level = torch.tensor(math.exp(min_level_db / ref_level_db * LOG_10)).to(linear.device)
        linear_db = ref_level_db * torch.log10(torch.maximum(min_level, linear)) - ref_level_db
        mel_norm = torch.clip(
            (linear_db - min_level_db) / -min_level_db,
            min=0.,
            max=1.
        )

        # time resolution adjustment
        # ensure length of raw audio is multiple of hop_size so that we can use
        # transposed convolution to upsample
        #print(sig_padded.min(),sig_padded.max())
        #print(sig_padded.size())
        out = sig_padded[:N * hop_length]
        assert out.size(0) % hop_length == 0

        local_conditioning = cin_channels>0

        if max_t_sec is not None:
            max_time_steps = int(max_t_sec*fs)
        elif max_t_steps is not None:
            max_time_steps = max_t_steps
        else:
            max_time_steps = None
          
        # Time resolution adjustment
        if local_conditioning:

            assert out.size(0) == (mel_norm.size(1)) * hop_length

            if max_time_steps is not None:

                max_steps = ensure_divisible(max_time_steps, hop_length, True)

                if out.size(0) > max_steps:
                    max_time_frames = max_steps // hop_length

                    s = np.random.randint(cin_pad, mel_norm.size(1) - max_time_frames - cin_pad)

                    ts = s * hop_length
                    out = out[ts:ts + hop_length * max_time_frames]
                    mel_out = mel_norm[:,s - cin_pad:s + max_time_frames + cin_pad]
                    assert out.size(0) == (mel_out.size(1) - 2 * cin_pad) * hop_length

            #Converts a class vector (integers) to 1-hot encodes a tensor
            out_quantized = torch.from_numpy(np.asarray(np.eye(num_classes, dtype='float32')[out]))

            return (out,out_quantized,mel_out) 
        else:
            out = out[0:max_time_steps]
            out_quantized = torch.from_numpy(np.asarray(np.eye(num_classes, dtype='float32')[out]))
            return(out,out_quantized)
    return f

def get_target(takes, provides):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(data):
        return data[0]
    return f

def get_x(takes, provides):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(data):
        return data[1]
    return f

def get_mel(takes, provides, local_conditioning):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(data):
        if local_conditioning:
            return data[2]
        else:
            return None
    return f

def get_target_length(takes, provides):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(target):
        return target.size(0)
    return f

def dataset_prep(dataset:DynamicItemDataset, hparams, tokens=None):
    """
    Prepares one or more datasets for use with wavenet.
    In order to be usable with the Wavenet model, a dataset needs to contain
    the following keys
    'wav': a file path to a .wav file containing the utterance

    Arguments
    ---------
    datasets
        a collection or datasets
    
    Returns
    -------
    the original dataset enhanced
    """
    OUTPUT_KEYS=["wav","signal","signal_cut","data","x","mel","target","target_length"]

    local_conditioning = hparams["c"] == 1

    pipeline = [
        read_wav,
        prepare_signal(
            takes ="signal",
            provides ="signal_cut",
            orig_freq=hparams["source_sample_rate"],
            fs=hparams["sample_rate"],
            threshold=hparams["trim_threshold"],
            cutoff = hparams["highpass_cutoff"],
            mu = hparams["mu"],
            silence_threshold = hparams["silence_threshold"]
            ),
        prepare_data(
            takes="signal_cut",
            provides="data",
            fs=hparams["sample_rate"],
            n_mels = hparams["num_mels"],
            hop_length = hparams["hop_length"],
            n_fft = hparams["n_fft"],
            min_level_db=hparams['min_level_db'],
            ref_level_db=hparams['ref_level_db'],
            cin_channels = hparams["cin_channels"],
            cin_pad = hparams["cin_pad"],
            max_t_sec = hparams["max_time_sec"],
            max_t_steps=hparams["max_time_steps"],
            num_classes = hparams["quantize_channels"]
            ),
        get_x(
            takes = "data",
            provides = "x"
            ),
        get_mel(
            takes = "data",
            provides = "mel",
            local_conditioning = local_conditioning
            ),
        get_target(
            takes = "data",
            provides = "target"
            ),
        get_target_length(
            takes = "target",
            provides = "target_length"
            )
        ]
    
    for element in pipeline:
        dataset.add_dynamic_item(element)

    dataset.set_output_keys(OUTPUT_KEYS)

    return dataset

def dataio_prep(hparams):
    result = {}
    for name, dataset_params in hparams['datasets'].items():
        # TODO: Add support for multiple datasets by instantiating from hparams - this is temporary
        vctk = VCTK(dataset_params['path']).to_dataset()
        result[name] = dataset_prep(vctk,hparams)
    return result