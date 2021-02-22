"""
This file contains functions to create mixtures given json metadata.
The mixtures simulate a multi-party conversation in a noisy scenario.

Author
------
Samuele Cornell, 2020
"""


import os
import torch
import json
import numpy as np
import torchaudio
from speechbrain.processing.signal_processing import rescale, reverberate


def create_mixture(session_n, output_dir, params, metadata):
    os.makedirs(os.path.join(output_dir, session_n), exist_ok=True)

    session_meta = {}
    speakers = [x for x in metadata.keys() if x not in ["noises", "background"]]

    tot_length = int(
        np.ceil(metadata["background"]["stop"] * params["samplerate"])
    )
    mixture = torch.zeros(tot_length)  # total mixture file
    assert len(mixture) > 0, "Mixture has length 0, please raise max_length."
    # step 1
    for spk in speakers:
        session_meta[spk] = []
        # we create mixture for each speaker and we optionally save it.
        if params["save_dry_sources"]:
            dry = torch.zeros(tot_length)
        if params["save_wet_sources"]:
            wet = torch.zeros(tot_length)

        for utt in metadata[spk]:
            c_audio, fs = torchaudio.load(
                os.path.join(params["librispeech_root"], utt["file"])
            )
            assert fs == params["samplerate"]
            if len(c_audio.shape) > 1:  # multichannel
                c_audio = c_audio[utt["channel"], :]
                c_audio = c_audio - torch.mean(c_audio)
            c_audio = rescale(
                c_audio,
                c_audio.size(0),
                utt["lvl"],
                scale="dB",
                amp_type="peak",
            )
            # we save it in dry
            dry_start = int(utt["start"] * params["samplerate"])
            dry_stop = dry_start + c_audio.shape[-1]
            if params["save_dry_sources"]:
                dry[dry_start:dry_stop] += c_audio
            # we add now reverb and put it in wet
            c_rir, fs = torchaudio.load(
                os.path.join(params["rirs_noises_root"], utt["rir"])
            )
            assert fs == params["samplerate"]
            c_rir = c_rir[utt["rir_channel"], :]

            c_audio = reverberate(c_audio, c_rir, "peak")
            # tof is not accounted because in reverberate we shift by it
            wet_start = dry_start
            wet_stop = dry_stop  # + early_rev_samples
            if params["save_wet_sources"]:
                wet[wet_start : wet_start + len(c_audio)] += c_audio

            session_meta[spk].append(
                {
                    "start": np.round(wet_start / params["samplerate"], 3),
                    "stop": np.round(wet_stop / params["samplerate"], 3),
                    "lvl": utt["lvl"],
                    "words": utt["words"],
                    "file": utt["file"],
                    "channel": utt["channel"],
                    "rir": utt["rir"],
                    "rir_channels": utt["rir_channel"],
                }
            )
            # we add to mixture
            mixture[wet_start : wet_start + len(c_audio)] += c_audio

        # we allow for clipping as it occurs also in real recordings.

        # save per speaker clean sources
        if params["save_dry_sources"]:
            torchaudio.save(
                os.path.join(
                    output_dir,
                    session_n,
                    "session_{}_spk_{}_dry.wav".format(session_n, spk),
                ),
                torch.clamp(dry, min=-1, max=1),
                params["samplerate"],
            )

        if params["save_wet_sources"]:
            torchaudio.save(
                os.path.join(
                    output_dir,
                    session_n,
                    "session_{}_spk_{}_wet.wav".format(session_n, spk),
                ),
                torch.clamp(wet, min=-1, max=1),
                params["samplerate"],
            )

    with open(
        os.path.join(output_dir, session_n, "{}.json".format(session_n)), "w"
    ) as f:
        json.dump(session_meta, f, indent=4)

    # add impulsive noises
    for noise_event in metadata["noises"]:

        c_audio, fs = torchaudio.load(
            os.path.join(params["rirs_noises_root"], noise_event["file"])
        )
        assert fs == params["samplerate"]
        if len(c_audio.shape) > 1:  # multichannel
            c_audio = c_audio[noise_event["channel"], :]
            c_audio = c_audio - torch.mean(c_audio)
        c_audio = rescale(
            c_audio,
            c_audio.size(0),
            noise_event["lvl"],
            scale="dB",
            amp_type="peak",
        )

        # we save it in dry
        dry_start = int(noise_event["start"] * params["samplerate"])
        # dry_stop = dry_start + c_audio.shape[-1]
        # we add now reverb and put it in wet
        c_rir, fs = torchaudio.load(
            os.path.join(params["rirs_noises_root"], noise_event["rir"])
        )
        assert fs == params["samplerate"]
        c_rir = c_rir[noise_event["rir_channel"], :]

        c_audio = reverberate(c_audio, c_rir, "peak")

        # tof is not accounted because in reverberate we shift by it
        wet_start = dry_start
        mixture[wet_start : wet_start + len(c_audio)] += c_audio

    # add background
    if metadata["background"]["file"]:
        c_audio, fs = torchaudio.load(
            os.path.join(
                params["backgrounds_root"], metadata["background"]["file"]
            ),
            frame_offset=metadata["background"]["orig_start"],
            num_frames=mixture.shape[-1],
        )
        assert fs == params["samplerate"]
        if len(c_audio.shape) > 1:  # multichannel
            c_audio = c_audio[metadata["background"]["channel"], :]
            c_audio = c_audio - torch.mean(c_audio)
        c_audio = rescale(
            c_audio,
            c_audio.size(0),
            metadata["background"]["lvl"],
            scale="dB",
            amp_type="avg",
        )
        mixture += c_audio

    else:
        # add gaussian noise
        mixture += rescale(
            torch.normal(0, 1, mixture.shape),
            mixture.size(0),
            metadata["background"]["lvl"],
            scale="dB",
            amp_type="peak",
        )

    # save total mixture
    mixture = torch.clamp(mixture, min=-1, max=1)
    torchaudio.save(
        os.path.join(output_dir, session_n, "{}_mixture.wav".format(session_n)),
        mixture.unsqueeze(0),
        params["samplerate"],
    )
