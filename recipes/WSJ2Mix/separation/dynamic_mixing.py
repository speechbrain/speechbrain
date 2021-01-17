import speechbrain as sb
import numpy as np
import torchaudio
from pathlib import Path
import torch
from speechbrain.utils.data_utils import batch_pad_right
import random
from speechbrain.processing.signal_processing import rescale


def build_spk_hashtable(train_data):

    spk_hashtable = {}
    for ex_id in train_data:
        s1_wav = train_data[ex_id]["s1_wav"]
        s2_wav = train_data[ex_id]["s2_wav"]

        # e.g. 2speakers/wav8k/min/tr/mix/019o031a_0.27588_01vo030q_-0.27588.wav
        # id of speaker 1 is 019 utterance id is o031a
        # id of speaker 2 is 01v utterance id is 01vo030q
        s1_id = Path(s1_wav).stem.split("_")[0][:3]
        if s1_id not in spk_hashtable.keys():
            spk_hashtable[s1_id] = [s1_wav]
        else:
            spk_hashtable[s1_id].append(s1_wav)

        s2_id = Path(s1_wav).stem.split("_")[2][:3]

        if s2_id not in spk_hashtable.keys():
            spk_hashtable[s2_id] = [s2_wav]
        else:
            spk_hashtable[s2_id].append(s2_wav)

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

    spk_hashtable, spk_weights = build_spk_hashtable(train_data.data)
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
        for i, spk in enumerate(speakers):
            c_file = np.random.choice(spk_hashtable[spk])
            # select random offset
            length = torchaudio.info(c_file).num_frames
            start = 0
            stop = length
            if length > hparams["training_signal_len"]:  # take a random window
                start = np.random.randint(
                    0, length - hparams["training_signal_len"]
                )
                stop = start + hparams["training_signal_len"]

            tmp, _ = torchaudio.load(
                c_file, frame_offset=start, num_frames=stop - start
            )
            tmp = tmp[0]  # remove channel dim and normalize
            if i == 0:
                lvl = np.clip(random.normalvariate(-16.7, 7), -45, 0)
                tmp = rescale(tmp, torch.tensor([len(tmp)]), lvl, scale="dB")
            else:
                lvl = np.clip(random.normalvariate(2.52, 4), -45, 0)
                tmp = rescale(tmp, torch.tensor([len(tmp)]), lvl, scale="dB")
            sources.append(tmp)

        # we mix the sources together
        # here we can also use augmentations ! -> runs on cpu and for each
        # mixture parameters will be different rather than for whole batch.
        # no difference however for bsz=1 :)

        # padding left
        sources, _ = batch_pad_right(sources)
        mixture = torch.sum(sources, 0)
        peak = torch.max(torch.abs(mixture))
        if peak > 1:
            sources = sources / peak
            mixture = torch.sum(sources, 0)

        yield mixture
        for i in range(hparams["num_spks"]):
            yield sources[i]

    sb.data_io.dataset.add_dynamic_item([train_data], audio_pipeline)
    sb.data_io.dataset.set_output_keys(
        [train_data], ["id", "mix_sig", "s1_sig", "s2_sig"]
    )

    return train_data
