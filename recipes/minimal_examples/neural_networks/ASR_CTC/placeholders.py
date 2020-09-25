import torch
import torchaudio
import os.path

torchaudio.set_audio_backend("soundfile")
DATAFOLDER = "../../../../samples/audio_samples/nn_training_samples"

ASR_example_ind2label = {
    0: "aa",
    1: "ae",
    2: "ah",
    3: "ao",
    4: "aw",
    5: "ax",
    6: "ay",
    7: "b",
    8: "ch",
    9: "cl",
    10: "d",
    11: "dh",
    12: "dx",
    13: "eh",
    14: "el",
    15: "er",
    16: "ey",
    17: "f",
    18: "g",
    19: "hh",
    20: "ih",
    21: "iy",
    22: "jh",
    23: "k",
    24: "l",
    25: "m",
    26: "n",
    27: "ng",
    28: "ow",
    29: "oy",
    30: "p",
    31: "r",
    32: "s",
    33: "sh",
    34: "t",
    35: "th",
    36: "uh",
    37: "uw",
    38: "v",
    39: "w",
    40: "y",
    41: "z",
    42: "sil",
    43: "vcl",
}

ASR_example_label2ind = {
    "aa": 0,
    "ae": 1,
    "ah": 2,
    "ao": 3,
    "aw": 4,
    "ax": 5,
    "ay": 6,
    "b": 7,
    "ch": 8,
    "cl": 9,
    "d": 10,
    "dh": 11,
    "dx": 12,
    "eh": 13,
    "el": 14,
    "er": 15,
    "ey": 16,
    "f": 17,
    "g": 18,
    "hh": 19,
    "ih": 20,
    "iy": 21,
    "jh": 22,
    "k": 23,
    "l": 24,
    "m": 25,
    "n": 26,
    "ng": 27,
    "ow": 28,
    "oy": 29,
    "p": 30,
    "r": 31,
    "s": 32,
    "sh": 33,
    "t": 34,
    "th": 35,
    "uh": 36,
    "uw": 37,
    "v": 38,
    "w": 39,
    "y": 40,
    "z": 41,
    "sil": 42,
    "vcl": 43,
}


class ASRMinimalExampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filepath=os.path.join(DATAFOLDER, "train.csv"),
        audio_transform=None,
        text_transform=None,
    ):
        self.items = []
        self.audio_transform = audio_transform
        self.text_transform = text_transform
        with open(filepath) as fi:
            fiterator = iter(fi)
            header = next(fiterator)
            field_idx = {
                fieldname.strip(): i
                for i, fieldname in enumerate(header.strip().split(","))
            }
            for line in fiterator:
                if not line.strip():
                    continue
                fields = line.strip().split(",")
                ID = fields[field_idx["ID"]]
                duration = float(fields[field_idx["duration"]])
                wavpath = fields[field_idx["wav"]].replace(
                    "$data_folder", DATAFOLDER
                )
                phn_seq = fields[field_idx["phn"]].strip()
                self.items.append((ID, duration, wavpath, phn_seq))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ID, duration, wavpath, phn_seq = self.items[idx]
        wav, samplerate = torchaudio.load(wavpath)
        wav = wav.squeeze(0)  # flat tensor
        if self.audio_transform is not None:
            wav = self.audio_transform(wav)
        phn_seq = phn_seq.split()
        if self.text_transform is not None:
            phn_seq = self.text_transform(phn_seq)
        return (ID, wav, phn_seq)


class ExampleCategoricalEncoder:
    def __init__(self, label2ind, ind2label):
        self.label2ind = label2ind
        self.ind2label = ind2label

    def encode_list(self, x):
        return [self.label2ind[item] for item in x]

    def decode_list(self, x):
        return [self.ind2label[item] for item in x]


def to_int_tensor(x):
    # Takes a python list of integers and returns torch integer tensor
    return torch.tensor(x, dtype=torch.int)


class FuncPipeline:
    # Chains together functions.
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, x):
        # NOTE: Does not support keyword arguments
        if not self.funcs:
            return x
        for func in self.funcs:
            x = func(x)
        return x


def pad_and_stack(sequences, padding_value=0):
    num = len(sequences)
    lens = [s.size(0) for s in sequences]
    max_len = max(lens)
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
    relative_lens = torch.tensor(lens, dtype=torch.float) / max_len
    return out_tensor, relative_lens


def ASR_example_collation(batch):
    IDs, wavs, phns = zip(*batch)
    wavs, wav_lens = pad_and_stack(wavs)
    phns, phn_lens = pad_and_stack(phns)
    return (IDs, wavs, wav_lens), (IDs, phns, phn_lens)
