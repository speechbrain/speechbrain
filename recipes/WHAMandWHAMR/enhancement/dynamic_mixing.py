import speechbrain as sb
import numpy as np
import torch
import torchaudio
import glob
import os
from pathlib import Path
import random
from speechbrain.processing.signal_processing import rescale
from speechbrain.dataio.batch import PaddedBatch

"""
The functions to implement Dynamic Mixing For SpeechSeparation

Authors
    * Samuele Cornell 2021
    * Cem Subakan 2021
"""


def build_spk_hashtable(base_folder_dm, sample_rate):
    """
    This function builds a dictionary of speaker-utterance pairs to be used in dynamic mixing

    arguments:
        base_folder_dm (str) : specifies the base folder for dynamic mixing.
        sample (int) : sampling frequency
    """

    wsj0_utterances = glob.glob(
        os.path.join(base_folder_dm, "**/*.wav"), recursive=True
    )

    spk_hashtable = {}
    for utt in wsj0_utterances:

        spk_id = Path(utt).stem[:3]
        assert torchaudio.info(utt).sample_rate == sample_rate

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


def get_wham_noise_filenames(data_root_folder, sample_rate):
    """
    This function lists the WHAM! noise files to be used in dynamic mixing

        data_root_folder (str) : specifies the system path for the top folder for the WHAM!, WHAMR! dataset
        sample_rate (int) : specifies the sample rate in Hz

    """

    if sample_rate == 8000:
        noise_path = "wav8k/min/tr/noise/"
    elif sample_rate == 16000:
        noise_path = "wav16k/min/tr/noise/"
    else:
        raise ValueError("Unsupported Sampling Rate")

    noise_files = glob.glob(os.path.join(data_root_folder, noise_path, "*.wav"))
    return noise_files


def dynamic_mix_data_prep(
    tr_csv,
    data_root_folder,
    base_folder_dm,
    sample_rate,
    num_spks,
    max_training_signal_len,
    batch_size=1,
    num_workers=1,
):
    """
    Dynamic mixing for WSJ0-2/3Mix and WHAM!/WHAMR!

        tr_csv (str) : the system path for the csv
        data_root_folder (str) : the system path for the root folder of the WHAM! / WHAMR! dataset
        base_folder_dm (str) : the system path for the wsj0 root folder
        sample_rate (int) : sampling frequency in Hz
        num_spks (int) : number of speakers (2 or 3)
        max_training_signal_len (int) : upper limit for the max_training_signal_len (in number of samples)
        batch_size (int) : batch_size
        num_workers (int) : number of workers for the dataloader
    """

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=tr_csv, replacements={"data_root": data_root_folder},
    )

    # we build an dictionary where keys are speakers id and entries are list
    # of utterances files of that speaker
    spk_hashtable, spk_weights = build_spk_hashtable(
        base_folder_dm=base_folder_dm, sample_rate=sample_rate
    )

    spk_list = [x for x in spk_hashtable.keys()]
    spk_weights = [x / sum(spk_weights) for x in spk_weights]

    if "wham" in Path(data_root_folder).stem:
        noise_files = get_wham_noise_filenames(data_root_folder, sample_rate)

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
            spk_list, num_spks, replace=False, p=spk_weights
        )

        if "wham" in Path(data_root_folder).stem:
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
            max_training_signal_len,
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
                spk_file, frame_offset=start, num_frames=stop - start,
            )

            tmp = tmp[0]  # * peak  # remove channel dim and normalize

            if i == 0:
                gain = np.clip(random.normalvariate(-27.43, 2.57), -45, 0)
                tmp = rescale(tmp, torch.tensor(len(tmp)), gain, scale="dB")
                first_lvl = gain
            else:
                gain = np.clip(
                    first_lvl + random.normalvariate(-2.51, 2.66), -45, 0
                )
                tmp = rescale(tmp, torch.tensor(len(tmp)), gain, scale="dB")
            sources.append(tmp)

        # we mix the sources together
        sources = torch.stack(sources)
        mixture = torch.sum(sources, 0)
        if "wham" in Path(data_root_folder).stem:
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
        for i in range(num_spks):
            yield sources[i]

        for i in range(num_spks, 3):
            yield None

        # If the number of speakers is 2, yield None for the 3rd speaker
        if "wham" in Path(data_root_folder).stem:
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
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=PaddedBatch,
        worker_init_fn=lambda x: np.random.seed(
            int.from_bytes(os.urandom(4), "little") + x
        ),
    )
    return train_data
