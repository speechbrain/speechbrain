import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.processing.signal_processing import rescale

"""
The functions to implement Dynamic Mixing For SpeechSeparation

Authors
    * Samuele Cornell 2021
    * Cem Subakan 2021
"""


def build_spk_hashtable(hparams):
    """
    This function builds a dictionary of speaker-utterance pairs to be used in dynamic mixing
    """
    wsj0_utterances = glob.glob(
        os.path.join(hparams["base_folder_dm"], "**/*.wav"), recursive=True
    )

    spk_hashtable = {}
    for utt in wsj0_utterances:
        spk_id = Path(utt).stem[:3]
        assert torchaudio.info(utt).sample_rate == hparams["sample_rate"]

        # e.g. 2speakers/wav8k/min/tr/mix/019o031a_0.27588_01vo030q_-0.27588.wav
        # id of speaker 1 is 019 utterance id is o031a
        # id of speaker 2 is 01v utterance id is 01vo030q

        if spk_id not in spk_hashtable.keys():
            spk_hashtable[spk_id] = [utt]
        else:
            spk_hashtable[spk_id].append(utt)

    # calculate weights for each speaker ( len of list of utterances)
    spk_weights = [len(spk_hashtable[x]) for x in spk_hashtable.keys()]

    return spk_hashtable, spk_weights


def get_wham_noise_filenames(hparams):
    "This function lists the WHAM! noise files to be used in dynamic mixing"

    if "Libri" in hparams["data_folder"]:
        # Data folder should point to Libri2Mix folder
        if hparams["sample_rate"] == 8000:
            noise_path = "wav8k/min/train-360/noise/"
        elif hparams["sample_rate"] == 16000:
            noise_path = "wav16k/min/train-360/noise/"
        else:
            raise ValueError("Unsupported Sampling Rate")
    else:
        if hparams["sample_rate"] == 8000:
            noise_path = "wav8k/min/tr/noise/"
        elif hparams["sample_rate"] == 16000:
            noise_path = "wav16k/min/tr/noise/"
        else:
            raise ValueError("Unsupported Sampling Rate")

    noise_files = glob.glob(
        os.path.join(hparams["data_folder"], noise_path, "*.wav")
    )
    return noise_files


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
    spk_hashtable, spk_weights = build_spk_hashtable(hparams)

    spk_list = [x for x in spk_hashtable.keys()]
    spk_weights = [x / sum(spk_weights) for x in spk_weights]

    if "wham" in Path(hparams["data_folder"]).stem:
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

        if "wham" in Path(hparams["data_folder"]).stem:
            noise_file = np.random.choice(noise_files, 1, replace=False)

            noise, fs_read = torchaudio.load(noise_file[0])
            noise = noise.squeeze()
            # gain = np.clip(random.normalvariate(1, 10), -4, 15)
            # noise = rescale(noise, torch.tensor(len(noise)), gain, scale="dB").squeeze()

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
                spk_file, frame_offset=start, num_frames=stop - start
            )

            # peak = float(Path(spk_file).stem.split("_peak_")[-1])
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
                # assert not torch.all(torch.isnan(tmp))
            sources.append(tmp)

        # we mix the sources together
        # here we can also use augmentations ! -> runs on cpu and for each
        # mixture parameters will be different rather than for whole batch.
        # no difference however for bsz=1 :)

        # padding left
        # sources, _ = batch_pad_right(sources)

        sources = torch.stack(sources)
        mixture = torch.sum(sources, 0)
        if "wham" in Path(hparams["data_folder"]).stem:
            len_noise = len(noise)
            len_mix = len(mixture)
            min_len = min(len_noise, len_mix)
            mixture = mixture[:min_len] + noise[:min_len]

        max_amp = max(
            torch.abs(mixture).max().item(),
            *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],
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

        if "wham" in Path(hparams["data_folder"]).stem:
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
