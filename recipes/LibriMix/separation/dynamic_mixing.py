import speechbrain as sb
import numpy as np
import torch
import torchaudio
import glob
import os
from speechbrain.dataio.batch import PaddedBatch
from tqdm import tqdm
import warnings
import pyloudnorm
import random

"""
The functions to implement Dynamic Mixing For SpeechSeparation

Authors
    * Samuele Cornell 2021
    * Cem Subakan 2021
"""


def build_spk_hashtable_librimix(hparams):
    """
    This function builds a dictionary of speaker-utterance pairs to be used in dynamic mixing
    """
    libri_utterances = glob.glob(
        os.path.join(hparams["base_folder_dm"], "**/*.wav"), recursive=True
    )

    spk_hashtable = {}

    # just for one file check if the sample rate is correct
    assert (
        torchaudio.info(libri_utterances[0]).sample_rate
        == hparams["sample_rate"]
    )
    for utt in tqdm(libri_utterances):

        path = os.path.normpath(utt)
        path_list = path.split(os.sep)
        spk_id = path_list[-3]

        # e.g. LibriSpeech/train-clean-100/441/128988/441-128988-0014.flac
        # id of speaker is 441 utterance is 128988-0014

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


def dynamic_mix_data_prep_librimix(hparams):
    """
    Dynamic mixing for LibriMix
    """

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    # we build an dictionary where keys are speakers id and entries are list
    # of utterances files of that speaker

    print("Building the speaker hashtable for dynamic mixing")
    spk_hashtable, spk_weights = build_spk_hashtable_librimix(hparams)

    spk_list = [x for x in spk_hashtable.keys()]
    spk_weights = [x / sum(spk_weights) for x in spk_weights]

    if hparams["use_wham_noise"]:
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

        if hparams["use_wham_noise"]:
            noise_file = np.random.choice(noise_files, 1, replace=False)

            noise, fs_read = torchaudio.load(noise_file[0])
            noise = noise.squeeze()

        # select two speakers randomly
        sources = []
        spk_files = [
            np.random.choice(spk_hashtable[spk], 1, False)[0]
            for spk in speakers
        ]

        minlen = min(
            *[torchaudio.info(x).num_frames for x in spk_files],
            hparams["training_signal_len"],
        )

        meter = pyloudnorm.Meter(hparams["sample_rate"])

        MAX_AMP = 0.9
        MIN_LOUDNESS = -33
        MAX_LOUDNESS = -25

        def normalize(signal, is_noise=False):
            """
            This function normalizes the audio signals for loudness
            """
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_loudness = meter.integrated_loudness(signal)
                if is_noise:
                    target_loudness = random.uniform(
                        MIN_LOUDNESS - 5, MAX_LOUDNESS - 5
                    )
                else:
                    target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
                signal = pyloudnorm.normalize.loudness(
                    signal, c_loudness, target_loudness
                )

                # check for clipping
                if np.max(np.abs(signal)) >= 1:
                    signal = signal * MAX_AMP / np.max(np.abs(signal))

            return torch.from_numpy(signal)

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
            tmp = tmp[0].numpy()
            tmp = normalize(tmp)
            sources.append(tmp)

        sources = torch.stack(sources)
        mixture = torch.sum(sources, 0)
        if hparams["use_wham_noise"]:
            len_noise = len(noise)
            len_mix = len(mixture)
            min_len = min(len_noise, len_mix)
            noise = normalize(noise.numpy(), is_noise=True)
            mixture = mixture[:min_len] + noise[:min_len]

        # check for clipping
        max_amp_insig = mixture.abs().max().item()
        if max_amp_insig > MAX_AMP:
            weight = MAX_AMP / max_amp_insig
        else:
            weight = 1

        sources = weight * sources
        mixture = weight * mixture

        yield mixture
        for i in range(hparams["num_spks"]):
            yield sources[i]

        # If the number of speakers is 2, yield None for the 3rd speaker
        if hparams["num_spks"] == 2:
            yield None

        if hparams["use_wham_noise"]:
            noise = noise * weight
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
