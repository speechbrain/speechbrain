import glob
import os
import random

import numpy as np
import torch
import torchaudio
from scipy.signal import fftconvolve

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.processing.signal_processing import rescale

"""
The functions to implement Dynamic Mixing For SpeechSeparation

Authors
    * Samuele Cornell 2021
    * Cem Subakan 2021
    * Zijian Huang 2022
"""


def dynamic_mix_data_prep(hparams):
    """
    Dynamic mixing for WSJ0-2/3Mix and WHAM!/WHAMR!
    """

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    # we build an dictionary where keys are speakers id and entries are list
    # of utterances files of that speaker
    from recipes.WSJ0Mix.separation.dynamic_mixing import (
        build_spk_hashtable,
        get_wham_noise_filenames,
    )

    spk_hashtable, spk_weights = build_spk_hashtable(hparams)

    spk_list = [x for x in spk_hashtable.keys()]
    spk_weights = [x / sum(spk_weights) for x in spk_weights]

    if "noise" in hparams["experiment_name"]:
        noise_files = get_wham_noise_filenames(hparams)

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides(
        "mix_sig", "s1_sig", "s2_sig", "s3_sig", "noise_sig"
    )
    def audio_pipeline(
        mix_wav,
    ):  # this is dummy --> it means one epoch will be same as without dynamic mixing
        """
        This audio pipeline defines the compute graph for dynamic mixing
        """

        speakers = np.random.choice(
            spk_list, hparams["num_spks"], replace=False, p=spk_weights
        )

        if "noise" in hparams["experiment_name"]:
            noise_file = np.random.choice(noise_files, 1, replace=False)

            noise, fs_read = torchaudio.load(noise_file[0])
            noise = noise.squeeze()

        # select two speakers randomly
        sources = []
        first_lvl = None

        spk_files = [
            np.random.choice(spk_hashtable[spk], 1, False)[0]
            for spk in speakers
        ]

        minlen = min(
            *[torchaudio.info(x).num_frames for x in spk_files],
            hparams["training_signal_len"],
        )

        for i, spk_file in enumerate(spk_files):
            # select random offset
            length = torchaudio.info(spk_file).num_frames
            start = 0
            stop = length
            if length > minlen:  # take a random window
                start = np.random.randint(0, length - minlen)
                stop = start + minlen

            tmp, fs_read = torchaudio.load(
                spk_file,
                frame_offset=start,
                num_frames=stop - start,
            )

            tmp = tmp[0]  # * peak  # remove channel dim and normalize

            if i == 0:
                gain = np.clip(random.normalvariate(-27.43, 2.57), -45, 0)
                tmp = rescale(tmp, torch.tensor(len(tmp)), gain, scale="dB")
                # assert not torch.all(torch.isnan(tmp))
                first_lvl = gain
            else:
                gain = np.clip(
                    first_lvl + random.normalvariate(-2.51, 2.66), -45, 0
                )
                tmp = rescale(tmp, torch.tensor(len(tmp)), gain, scale="dB")

            if "reverb" in hparams["experiment_name"]:
                tmp = torch.stack((tmp, tmp), 1)
                reverb_time = np.random.choice(
                    ["0_1s", "0_2s", "0_4s", "0_7s", "0_8s"]
                )
                azimuth = np.random.choice(list(range(-90, 91, 5)))
                hrtf_file = os.path.join(
                    hparams["hrtf_wav_path"],
                    reverb_time,
                    "CATT_{}_{}.wav".format(reverb_time, azimuth),
                )
                hrtf, sr = torchaudio.load(hrtf_file)
                transform = torchaudio.transforms.Resample(sr, fs_read)
                hrtf = transform(hrtf)
                tmp_bi = torch.from_numpy(
                    fftconvolve(tmp.numpy(), hrtf.numpy(), mode="same")
                )
            else:
                tmp_bi = torch.FloatTensor(len(tmp), 2)  # binaural
                subject_path_list = glob.glob(
                    os.path.join(hparams["hrtf_wav_path"], "subject*")
                )
                subject_path = np.random.choice(subject_path_list)
                azimuth_list = (
                    [-80, -65, -55] + list(range(-45, 46, 5)) + [55, 65, 80]
                )
                azimuth = np.random.choice(azimuth_list)

                for i, loc in enumerate(["left", "right"]):
                    hrtf_file = os.path.join(
                        subject_path,
                        "{}az{}.wav".format(
                            azimuth.astype("str").replace("-", "neg"), loc
                        ),
                    )
                    hrtf, sr = torchaudio.load(hrtf_file)
                    transform = torchaudio.transforms.Resample(sr, fs_read)
                    hrtf = transform(hrtf[:, np.random.randint(50)])
                    tmp_bi[:, i] = torch.from_numpy(
                        fftconvolve(tmp.numpy(), hrtf.numpy(), mode="same")
                    )

            # Make relative source energy same with original
            spatial_scaling = torch.sqrt(
                torch.sum(tmp**2) * 2 / torch.sum(tmp_bi**2)
            )
            sources.append(tmp_bi * spatial_scaling)

        # we mix the sources together
        # here we can also use augmentations ! -> runs on cpu and for each
        # mixture parameters will be different rather than for whole batch.
        # no difference however for bsz=1 :)

        # padding left
        # sources, _ = batch_pad_right(sources)

        sources = torch.stack(sources)
        mixture = torch.sum(sources, 0)
        if "noise" in hparams["experiment_name"]:
            len_noise = len(noise)
            len_mix = len(mixture)
            min_len = min(len_noise, len_mix)
            noise = torch.swapaxes(noise, 0, 1)
            mixture = mixture[:min_len] + noise[:min_len]

        max_amp = max(
            torch.abs(mixture).max().item(),
            *[
                x.item()
                for x in torch.abs(sources).max(dim=-1)[0].max(dim=-1)[0]
            ],
        )
        mix_scaling = 1 / max_amp * 0.9
        sources = mix_scaling * sources
        mixture = mix_scaling * mixture

        yield mixture
        for i in range(hparams["num_spks"]):
            yield sources[i]

        # If the number of speakers is 2, yield None for the 3rd speaker
        if hparams["num_spks"] == 2:
            yield None

        if "noise" in hparams["experiment_name"]:
            mean_source_lvl = sources.abs().mean()
            mean_noise_lvl = noise.abs().mean()
            noise = (mean_source_lvl / mean_noise_lvl) * noise
            yield noise
        else:
            yield None

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline)
    sb.dataio.dataset.set_output_keys(
        [train_data],
        ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig", "noise_sig"],
    )

    train_data = torch.utils.data.DataLoader(
        train_data,
        batch_size=hparams["dataloader_opts"]["batch_size"],
        num_workers=hparams["dataloader_opts"]["num_workers"],
        collate_fn=PaddedBatch,
        worker_init_fn=lambda x: np.random.seed(
            int.from_bytes(os.urandom(4), "little") + x
        ),
    )
    return train_data
