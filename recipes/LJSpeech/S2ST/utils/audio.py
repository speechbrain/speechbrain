import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
from librosa.util import normalize
import numpy as np
import parselmouth

MAX_WAV_VALUE = 32768.0

def extract_f0(wav, sr=16000, extractor="pyaapt", interp=False):
    # wav = wav / MAX_WAV_VALUE
    # wav = normalize(wav) * 0.95

    if extractor == "pyaapt":
        frame_length = 20.0
        pad = int(frame_length / 1000 * sr) // 2
        wav = np.pad(wav.squeeze(), (pad, pad), "constant", constant_values=0)
        signal = basic.SignalObj(wav, sr)
        pitch = pYAAPT.yaapt(
                signal,
                **{
                    'frame_length': frame_length,
                    'frame_space': 5.0,
                    'nccf_thresh1': 0.25,
                    'tda_frame_length': 25.0
                })
        pitch = pitch.samp_interp[None, None, :] if interp else pitch.samp_values[None, None, :]
        pitch = pitch[0, 0]
        return pitch

    elif extractor == "parselmouth":
        frame_length = 256/sr
        pad = int(frame_length / 1000 * sr) // 2
        wav = np.pad(wav.squeeze(), (pad, pad), "constant", constant_values=0)
        x = wav.astype(np.double)
        snd = parselmouth.Sound(values=x, sampling_frequency=sr)
        pitch  = snd.to_pitch(time_step=frame_length, pitch_floor=40, pitch_ceiling=600).selected_array['frequency']
        return pitch