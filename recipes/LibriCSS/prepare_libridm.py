import os
import csv
import torch
import torchaudio

from speechbrain.processing.dynamic_mixing import mix
from speechbrain.processing.signal_processing import reverberate


def prepare_libridm(librispeech_path, openrir_path, savepath, skip_prep=False):
    if os.path.exists(os.path.join(savepath, "audio")) and skip_prep:
        return

    sources = [
        "test-clean/3729/6852/3729-6852-0037.flac",
        "test-clean/4446/2275/4446-2275-0009.flac",
    ]
    sources = [os.path.join(librispeech_path, src) for src in sources]
    rir = os.path.join(
        openrir_path,
        "real_rirs_isotropic_noises/air_type1_air_binaural_meeting_0_1.wav",
    )
    noise = os.path.join(
        openrir_path,
        "real_rirs_isotropic_noises/RVB2014_type1_noise_largeroom2_8.wav",
    )

    csv_columns = [
        "ID",
        "duration",
        "num_samples",
        "mix_wav",
        "s1_wav",
        "s2_wav",
        "noise_wav",
    ]

    os.makedirs(os.path.join(savepath, "audio"))
    savepath = os.path.abspath(savepath)

    with open(os.path.join(savepath, "libridm.csv"), "w") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_columns)
        writer.writeheader()
        # 2 spkr case
        for ovrlp in [1.0, 0.5, -0.5]:
            for d in create_samples(*sources, noise, rir, ovrlp):
                save_data(writer, savepath, *d)

        # 1 spkr case
        src1, _ = torchaudio.load(sources[0])
        noise, _ = torchaudio.load(noise)
        rir, _ = torchaudio.load(rir)
        src1, noise, rir = [x[0] for x in [src1, noise, rir]]
        noise = noise[: src1.size(0)]

        save_data(
            writer,
            savepath,
            "LibriDM_" + "3729-6852-0037" + "_single",
            src1,
            src1,
            None,
            None,
        )
        save_data(
            writer,
            savepath,
            "LibriDM_" + "3729-6852-0037" + "_single" + "_noiseRVB2014_type1_noise_largeroom2_8",
            src1 + noise,
            src1,
            None,
            noise,
        )
        src1 = reverberate(src1, rir)
        save_data(
            writer,
            savepath,
            "LibriDM_" + "3729-6852-0037" + "_single" + "_rirair_type1_air_binaural_meeting_0_1",
            src1,
            src1,
            None,
            None,
        )
        save_data(
            writer,
            savepath,
            "LibriDM_" + "3729-6852-0037" + "_single" + "_noiseRVB2014_type1_noise_largeroom2_8" + "_rirair_type1_air_binaural_meeting_0_1",
            src1 + noise,
            src1,
            None,
            noise,
        )

        # 0 spkr case
        src1 = torch.zeros(noise.shape)
        save_data(
            writer,
            savepath,
            "LibriDM" + "_none",
            src1,
            None,
            None,
            None,
        )
        save_data(
            writer,
            savepath,
            "LibriDM" + "_none" + "_noiseRVB2014_type1_noise_largeroom2_8",
            src1 + noise,
            None,
            None,
            noise,
        )


def save_data(writer, savepath, mix_id, mixture, src1, src2, noise):
    row = {
        "ID": mix_id,
        "duration": mixture.size(0) / 16000,
        "num_samples": mixture.size(0),
        "mix_wav": os.path.join(savepath, "audio", mix_id + ".wav"),
        "s1_wav": None,
        "s2_wav": None,
        "noise_wav": None,
    }

    torchaudio.save(row["mix_wav"], mixture.unsqueeze(0), sample_rate=16000)

    if src1 is not None:
        row["s1_wav"] = os.path.join(savepath, "audio", mix_id + "_s1.wav")
        torchaudio.save(row["s1_wav"], src1.unsqueeze(0), sample_rate=16000)

    if src2 is not None:
        row["s2_wav"] = os.path.join(savepath, "audio", mix_id + "_s2.wav")
        torchaudio.save(row["s2_wav"], src2.unsqueeze(0), sample_rate=16000)

    if noise is not None:
        row["noise_wav"] = os.path.join(savepath, "audio", mix_id + "_noise.wav")
        torchaudio.save(row["noise_wav"], noise.unsqueeze(0), sample_rate=16000)
    writer.writerow(row)


def create_samples(src1_f, src2_f, noise_f, rir_f, overlap):
    src1, fs = torchaudio.load(src1_f)
    assert fs == 16000
    src2, _ = torchaudio.load(src2_f)
    rir, fs = torchaudio.load(rir_f)
    assert fs == 16000
    noise, fs = torchaudio.load(noise_f)
    assert fs == 16000
    src1, src2, rir, noise = [
        x[0] for x in [src1, src2, rir, noise]
    ]  # 1st channel

    ids = [os.path.splitext(os.path.basename(x))[0] for x in [src1_f, src2_f]]
    noise_id = os.path.splitext(os.path.basename(noise_f))[0]
    rir_id = os.path.splitext(os.path.basename(rir_f))[0]

    ovrlp_samples = int(min(src1.size(0), src2.size(0)) * overlap)
    mix_id = "LibriDM_" + "_".join(ids) + "_ovrlp" + str(overlap)
    mixture, mix_srcs, _ = mix(src1, src2, ovrlp_samples)
    orig_srcs = [src1, src2]
    src1, src2 = mix_srcs
    data = [(mix_id, mixture, src1, src2, None)]

    data.append(
        (mix_id + "_noise" + noise_id, mixture + noise[:mixture.size(0)], src1, src2, noise[:mixture.size(0)])
    )

    src1, src2 = orig_srcs
    src1, src2 = [reverberate(x, rir) for x in [src1, src2]]
    mixture, mix_srcs, _ = mix(src1, src2, ovrlp_samples)
    src1, src2 = mix_srcs
    data.append((mix_id + "_rvb" + rir_id, mixture, src1, src2, None))

    data.append(
        (
            mix_id + "_rvb" + rir_id + "_noise" + noise_id,
            mixture + noise[:mixture.size(0)],
            src1,
            src2,
            noise[:mixture.size(0)],
        )
    )

    return data


if __name__ == "__main__":
    import sys
    print(sys.argv, file=sys.stderr)
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <LibriSpeech> <OpenRIR> <savepath>", file=sys.stderr)
        sys.exit(1)
    prepare_libridm(sys.argv[1], sys.argv[2], sys.argv[3])
