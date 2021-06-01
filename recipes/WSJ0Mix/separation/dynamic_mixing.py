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
from tqdm import tqdm
import warnings
import pyloudnorm

"""
The functions to implement Dynamic Mixing For SpeechSeparation

Authors
    * Samuele Cornell 2021
    * Cem Subakan 2021
"""


def build_spk_hashtable(hparams):

    wsj0_utterances = glob.glob(
        os.path.join(hparams["wsj0_tr"], "**/*.wav"), recursive=True
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


def build_spk_hashtable_librimix(hparams):

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
    @sb.utils.data_pipeline.provides("mix_sig", "s1_sig", "s2_sig", "noise_sig")
    def audio_pipeline(
        mix_wav,
    ):  # this is dummy --> it means one epoch will be same as without dynamic mixing

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
                spk_file, frame_offset=start, num_frames=stop - start,
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

        if "wham" in Path(hparams["data_folder"]).stem:
            mean_source_lvl = sources.abs().mean()
            mean_noise_lvl = noise.abs().mean()
            noise = (mean_source_lvl / mean_noise_lvl) * noise
            yield noise
        else:
            yield None

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline)
    sb.dataio.dataset.set_output_keys(
        [train_data], ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig"]
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


def dynamic_mix_data_prep_3mix(hparams):

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

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig", "s1_sig", "s2_sig", "s3_sig")
    def audio_pipeline(
        mix_wav,
    ):  # this is dummy --> it means one epoch will be same as without dynamic mixing

        speakers = np.random.choice(
            spk_list, hparams["num_spks"], replace=False, p=spk_weights
        )
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
                spk_file, frame_offset=start, num_frames=stop - start,
            )

            # peak = float(Path(spk_file).stem.split("_peak_")[-1])
            tmp = tmp[0]  # * peak  # remove channel dim and normalize

            if i == 0:
                gain = np.clip(random.normalvariate(-27.43, 2.57), -45, 0)
                tmp = rescale(tmp, torch.tensor(len(tmp)), gain, scale="dB")
                # assert not torch.all(torch.isnan(tmp))
                first_lvl = gain
            elif i == 1:
                gain = np.clip(
                    first_lvl + random.normalvariate(-2.51, 2.66), -45, 0
                )
                tmp = rescale(tmp, torch.tensor(len(tmp)), gain, scale="dB")
            else:
                pass
                # note that we effectively using 0dB gain for the last source

            sources.append(tmp)

        # we mix the sources together
        # here we can also use augmentations ! -> runs on cpu and for each
        # mixture parameters will be different rather than for whole batch.
        # no difference however for bsz=1 :)

        # padding left
        # sources, _ = batch_pad_right(sources)

        sources = torch.stack(sources)
        mixture = torch.sum(sources, 0)
        max_amp = max(
            torch.abs(mixture).max().item(),
            *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],
        )
        mix_scaling = 1 / max_amp * 0.9
        sources = sources * mix_scaling
        mixture = mix_scaling * mixture

        yield mixture
        for i in range(hparams["num_spks"]):
            yield sources[i]

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline)
    sb.dataio.dataset.set_output_keys(
        [train_data], ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig"]
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


def dynamic_mix_data_prep_librimix(hparams):

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
    @sb.utils.data_pipeline.provides("mix_sig", "s1_sig", "s2_sig", "noise_sig")
    def audio_pipeline(
        mix_wav,
    ):  # this is dummy --> it means one epoch will be same as without dynamic mixing

        speakers = np.random.choice(
            spk_list, hparams["num_spks"], replace=False, p=spk_weights
        )

        if hparams["use_wham_noise"]:
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

        meter = pyloudnorm.Meter(hparams["sample_rate"])

        MAX_AMP = 0.9
        MIN_LOUDNESS = -33
        MAX_LOUDNESS = -25

        def normalize(signal, is_noise=False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_loudness = meter.integrated_loudness(signal)
                if is_noise:
                    target_loudness = np.random.randint(MIN_LOUDNESS - 5, MAX_LOUDNESS - 5)
                else:
                    target_loudness = np.random.randint(MIN_LOUDNESS, MAX_LOUDNESS)
                signal = pyloudnorm.normalize.loudness(signal, c_loudness, target_loudness)

                # check for clipping
                if np.max(np.abs(src)) >= 1:
                    signal = signal * MAX_AMP / np.max(np.abs(signal))

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
            tmp = normalize(signal)
            sources.append(tmp)

        sources = torch.stack(sources)
        mixture = torch.sum(sources, 0)
        if hparams["use_wham_noise"]:
            len_noise = len(noise)
            len_mix = len(mixture)
            min_len = min(len_noise, len_mix)
            noise = normalize(noise, is_noise=True)
            mixture = mixture[:min_len] + noise[:min_len]

        # check for clipping 
        if np.max(np.abs(mixture)) > MAX_AMP:
            weight = MAX_AMP / np.max(np.abs(mixture))
        else:
            weight = 1

        max_amp = max(torch.abs(mixture).max().item())
        sources = weight * sources
        mixture = weight * mixture

        yield mixture
        for i in range(hparams["num_spks"]):
            yield sources[i]

        if hparams["use_wham_noise"]:
            noise = noise * weight
            yield noise
        else:
            yield None

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline)
    sb.dataio.dataset.set_output_keys(
        [train_data], ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig"]
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


def dynamic_mix_shuffleonly_data_prep(hparams):

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    # we draw Nspk indices
    source_wavkeys = [
        "s" + str(i) + "_wav" for i in range(1, hparams["num_spks"] + 1)
    ]

    @sb.utils.data_pipeline.takes("s1_wav", "s2_wav")
    @sb.utils.data_pipeline.provides("mix_sig", "s1_sig", "s2_sig")
    def audio_pipeline(
        s1_wav, s2_wav
    ):  # this is dummy --> it means one epoch will be same as without dynamic mixing

        # find the indices of two items to mix
        inds = list(
            np.random.random_integers(
                0, len(train_data) - 1, size=(hparams["num_spks"],)
            )
        )

        # get the lengths of these items
        lengths = []
        sourcefls = []
        for i, (ind, wavkey) in enumerate(zip(inds, source_wavkeys)):
            fl = train_data.data[str(ind)]
            sourcefl = fl[wavkey]
            sourcefls.append(sourcefl)
            lengths.append(torchaudio.info(sourcefl).num_frames)
        minlen = min(lengths)

        sources = []
        for i, (sourcefl, wavkey, length) in enumerate(
            zip(sourcefls, source_wavkeys, lengths)
        ):

            start = 0
            stop = length
            if length > minlen:  # take a random window
                start = np.random.randint(0, length - minlen)
                stop = start + minlen

            tmp, fs_read = torchaudio.load(
                sourcefl,
                frame_offset=start,
                num_frames=stop - start,
                # normalize=False,
            )

            tmp = tmp[0]  # remove channel dim
            sources.append(tmp)

        sources = torch.stack(sources)
        mixture = torch.sum(sources, 0)
        max_amp = max(
            torch.abs(mixture).max().item(),
            *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],
        )
        mix_scaling = 1 / max_amp * 0.9
        sources = sources * mix_scaling
        mixture = mix_scaling * mixture

        yield mixture
        for i in range(hparams["num_spks"]):
            yield sources[i]

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline)
    sb.dataio.dataset.set_output_keys(
        [train_data], ["id", "mix_sig", "s1_sig", "s2_sig"]
    )

    return train_data
