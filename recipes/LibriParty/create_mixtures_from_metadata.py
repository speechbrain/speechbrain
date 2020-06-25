import torch
import json
import numpy as np
import torchaudio


def peakGain(
    tensor, target_peak_dB
):  # this can actually be included into signal_processing
    # also, Peter might want to check this, i normalize the peak, i am not sure if it makes sense to normalize the mean
    # amplitude.....
    target_peak_dB = 10 ** (target_peak_dB / 20)
    return tensor / target_peak_dB


# this is same function as speech_augmentation AddReverb but it does not take a csv file.
# we might want to do a more general function into processing which is re--used by addReverb.
def reverberate(dry, rir):
    pass


def get_early_rev_samples(rir, th=1000):
    # we find max value
    assert (
        len(rir.shape) == 1
    ), "multidimensional tensors not supported currently"
    max, max_indx = torch.max(torch.abs(rir), dim=0)
    first_min = torch.where(torch.abs(rir[max_indx:]) <= max / th)[-1]

    return first_min + max_indx  # add back max_indx


def create_mixture(session_n, output_dir, params, metadata):
    os.makedirs(os.path.join(output_dir, session_n), exist_ok=True)

    session_meta = {}
    speakers = [x for x in metadata.keys() if x not in ["noises", "background"]]

    tot_length = int(
        np.ceil(metadata["background"]["stop"] * params["samplerate"])
    )
    mixture = torch.zeros(tot_length)  # total mixture file
    # step 1
    for spk in speakers:

        session_meta[spk] = []

        # we create mixture for each speaker and we optionally save it.
        if params["save_dry_sources"]:
            dry = torch.zeros(tot_length)
        wet = torch.zeros(tot_length)

        for utt in metadata[spk]:
            c_audio, fs = torchaudio.load(
                os.path.join(params["librispeech_root"], utt["file"])
            )
            assert fs == params["samplerate"]
            if len(c_audio.shape) > 1:  # multichannel
                c_audio = c_audio[utt["channel"], :]
            c_audio = peakGain(c_audio, utt["lvl"])
            # we save it in dry
            dry_start = int(utt["start"] * params["samplerate"])
            dry_stop = dry_start + c_audio.shape[-1]
            if params["save_dry_sources"]:
                dry[dry_start:dry_stop] = c_audio
            # we add now reverb and put it in wet
            c_rir, fs = torchaudio.load(
                os.path.join(params["rirs_root"], utt["rir"])
            )
            assert fs == params["samplerate"]
            c_rir = c_rir[utt["rir_channel"], :]
            tof = torch.where(torch.abs(c_rir) >= 1e-8)[-1][0]
            early_rev_samples = get_early_rev_samples(c_rir)

            c_audio = reverberate(c_audio, c_rir)
            wet_start = dry_start + tof
            wet_stop = dry_stop + tof + early_rev_samples
            wet[wet_start:wet_stop] = c_audio

            session_meta[spk].append(
                {
                    "start": wet_start,
                    "stop": wet_stop,
                    "lvl": utt["lvl"],
                    "words": utt["words"],
                    "file": utt["file"],
                    "channel": utt["channel"],
                    "rir": utt["rir"],
                    "rir_channels": utt["rir_channel"],
                }
            )
        # we add to mixture
        mixture += wet

        # save files for current speaker
        os.makedirs(
            os.path.join(output_dir, "session_{}".format(), "{}".format(spk)),
            exist_ok=True,
        )
        if params["save_dry_sources"]:
            torch.save(
                dry,
                os.path.join(
                    output_dir,
                    "{}".format(spk),
                    "session_{}_spk_{}_dry.wav".format(spk),
                ),
            )
        if params["save_wet_sources"]:
            torch.save(
                wet,
                os.path.join(
                    output_dir,
                    "{}".format(spk),
                    "session_{}_spk_{}_wet.wav".format(spk),
                ),
            )

    # how to handle clipping ? we either rescale everything to avoid it or we make it happen.
    # clipping occurs on real world data so we make it happen also here.

    # TODO noise & background

    torch.save(
        mixture,
        os.path.join(
            output_dir,
            "{}".format(spk),
            "session_{}_spk_{}_wet.wav".format(spk),
        ),
    )


if __name__ == "__main__":
    import os
    import sys
    import yaml
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(
        "Creating synthetic mixtures from metadata"
    )
    parser.add_argument("params_file", type=str)
    parser.add_argument("metadata_file", type=str)
    parser.add_argument("output_dir", type=str)

    # This hack needed to import data preparation script from ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from create_mixtures_metadata import create_metadata  # noqa E402

    args = parser.parse_args()

    # we load parameters
    with open(args.params_file, "r") as f:
        params = yaml.load(f)

    with open(args.metadata_file, "r") as f:
        metadata = json.load(f)

    for session in tqdm(metadata.keys()):

        create_mixture(session, args.output_dir, params, metadata[session])
        # open metadata and create session
