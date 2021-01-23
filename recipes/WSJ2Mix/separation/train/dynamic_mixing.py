import speechbrain as sb
import numpy as np
import torch
import torchaudio
import glob
import os
from pathlib import Path
import json


def build_spk_hashtable(hparams):

    utterances = glob.glob(os.path.join(hparams["wsj0_max_tr"], "s1", "*.wav"))

    utterances = [Path(u).stem for u in utterances]

    with open("/tmp/scalings.json", "r") as f:
        scaling_coeffs = json.load(f)

    spk_hashtable = {}

    for utt_id in utterances:

        s1_utt = utt_id.split("_")[0]
        s1_scaling = scaling_coeffs[utt_id][0]
        s2_utt = utt_id.split("_")[0]
        s2_scaling = scaling_coeffs[utt_id][1]

        s1_id = s1_utt[:3]

        # e.g. 2speakers/wav8k/min/tr/mix/019o031a_0.27588_01vo030q_-0.27588.wav
        # id of speaker 1 is 019 utterance id is o031a
        # id of speaker 2 is 01v utterance id is 01vo030q

        # we put s1 into the hashtable together with its scaling
        if s1_id not in spk_hashtable.keys():
            spk_hashtable[s1_id] = [
                (
                    os.path.join(hparams["wsj0_max_tr"], "s1", utt_id + ".wav"),
                    s1_scaling,
                )
            ]
        else:
            spk_hashtable[s1_id].append(
                (
                    os.path.join(hparams["wsj0_max_tr"], "s1", utt_id + ".wav"),
                    s1_scaling,
                )
            )

        # same for s2
        s2_id = s2_utt[:3]
        if s2_id not in spk_hashtable.keys():
            spk_hashtable[s2_id] = [
                (
                    os.path.join(hparams["wsj0_max_tr"], "s2", utt_id + ".wav"),
                    s2_scaling,
                )
            ]
        else:
            spk_hashtable[s2_id].append(
                (
                    os.path.join(hparams["wsj0_max_tr"], "s2", utt_id + ".wav"),
                    s2_scaling,
                )
            )

    # calculate weights for each speaker ( len of list of utterances)
    spk_weights = [len(spk_hashtable[x]) for x in spk_hashtable.keys()]

    return spk_hashtable, spk_weights


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

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig", "s1_sig", "s2_sig")
    def audio_pipeline(
        mix_wav,
    ):  # this is dummy --> it means one epoch will be same as without dynamic mixing

        speakers = np.random.choice(
            spk_list, hparams["num_spks"], replace=False, p=spk_weights
        )
        # select two speakers randomly
        sources = []
        first_lvl = None
        spk_files = []
        for spk in speakers:
            c_indx = np.random.randint(0, len(spk_hashtable[spk]))
            spk_files.append(spk_hashtable[spk][c_indx])

        minlen = min(
            *[torchaudio.info(x[0]).num_frames for x in spk_files],
            hparams["training_signal_len"],
        )

        for i, spk_file in enumerate(spk_files):

            # select random offset
            length = torchaudio.info(spk_file[0]).num_frames
            start = 0
            stop = length
            if length > minlen:  # take a random window
                start = np.random.randint(0, length - minlen)
                stop = start + minlen

            tmp, fs_read = torchaudio.load(
                spk_file[0], frame_offset=start, num_frames=stop - start,
            )

            tmp = tmp[0] / spk_file[1]

            if i == 0:
                lvl = 10 ** (np.random.uniform(-2.5, 0) / 20)
                tmp = tmp * lvl
                first_lvl = lvl
            else:
                tmp = tmp * -first_lvl
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
        [train_data], ["id", "mix_sig", "s1_sig", "s2_sig"]
    )

    return train_data


def dynamic_mix_shuffleonly_data_prep(hparams):

    # 1. Define datasets
    train_data = sb.data_io.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )
    import pdb

    pdb.set_trace()

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

    sb.data_io.dataset.add_dynamic_item([train_data], audio_pipeline)
    sb.data_io.dataset.set_output_keys(
        [train_data], ["id", "mix_sig", "s1_sig", "s2_sig"]
    )
    train_data[0]

    return train_data
