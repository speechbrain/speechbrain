import speechbrain as sb
import numpy as np
import torchaudio
import torch
import glob
import os
from pathlib import Path


def build_spk_hashtable(hparams):

    wsj0_utterances = glob.glob(
        os.path.join(hparams["wsj0_tr"], "**/*.wav"), recursive=True
    )

    spk_hashtable = {}
    for utt in wsj0_utterances:

        spk_id = Path(utt).stem[:3]
        torchaudio.info(utt)
        assert torchaudio.info(utt).sample_rate == 8000

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


def dynamic_mix_data_prep(hparams):

    # 1. Define datasets
    train_data = sb.data_io.dataset.DynamicItemDataset.from_csv(
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
            tmp = tmp[0]  # remove channel dim and normalize

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

    sb.data_io.dataset.add_dynamic_item([train_data], audio_pipeline)
    sb.data_io.dataset.set_output_keys(
        [train_data], ["id", "mix_sig", "s1_sig", "s2_sig"]
    )

    from speechbrain.processing.signal_processing import compute_amplitude

    s1_amp = []
    s2_amp = []
    for i in range(len(train_data)):
        batch = train_data[i]
        s1 = batch["s1_sig"]
        s2 = batch["s2_sig"]
        s1_amp.append(compute_amplitude(s1, scale="dB", amp_type="avg").item())
        s2_amp.append(compute_amplitude(s2, scale="dB", amp_type="avg").item())

    return train_data
