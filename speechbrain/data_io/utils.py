import os
import torch
from ruamel import yaml
from collections.abc import MutableMapping
from copy import copy


class DataCollection(MutableMapping):
    # TODO not still sure about this
    def __init__(self, path_to_yaml):

        with open(path_to_yaml, "r") as f:
            self.datacol = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.datacol[key]

    def __delitem__(self, key):
        del self.datacol[key]

    def __setitem__(self, key, value):
        if key in self:
            del self[self[key]]
        if value in self:
            del self[value]
        self.datacol[key] = value
        self.datacol[value] = key

    def __iter__(self):
        return iter(self.datacol)

    def __len__(self):
        return len(self.datacol)

    # def __repr__(self):
    #   return f"{type(self).__name__}({self.datacol})

    def sanity_check(self):
        dataset_sanity_check(self)

    def to_ASR_format(self):
        pass

    def filter(self):
        pass

    def sort(self):
        pass

    @staticmethod
    def merge(list_of_datacollections):
        pass


def dataset_sanity_check(dataset_splits):

    # dataset_structs is a list of dataset_struct
    # each dataset struct is a dict-based hierarchical annotation
    # we check for consistency between all of these e.g. between train.yaml, dev.yaml and test.yaml

    assert len(dataset_splits) > 0, "provided list of dataset splits is empty"

    # we take first supervision of first structure
    first_key = list(dataset_splits[0].keys())[0]
    first_sup_keys = copy(
        list(dataset_splits[0][first_key]["supervision"][0].keys())
    )

    for d_split in dataset_splits:
        for data_obj_id in d_split.keys():
            c_obj = d_split[data_obj_id]
            # we check that waveforms have at least one file, channels, lengths, and samplerate
            assert all(
                [
                    k in c_obj["waveforms"].keys()
                    for k in ["files", "channels", "samplerate", "lengths"]
                ]
            ), "Waveforms entries should have always files, channels, samplerate and length keys. "
            # assert are not empty and are of proper type
            assert isinstance(c_obj["waveforms"]["files"], list)
            assert isinstance(c_obj["waveforms"]["files"][0], str)
            assert isinstance(c_obj["waveforms"]["channels"], list)
            assert isinstance(c_obj["waveforms"]["channels"][0], list)
            assert isinstance(c_obj["waveforms"]["lengths"], list)
            assert isinstance(c_obj["waveforms"]["lengths"][0], int)
            assert isinstance(c_obj["waveforms"]["samplerate"], int)

            # assert files exists
            for f in c_obj["waveforms"]["files"]:
                assert os.path.exists(f), "{} does not exist".format(f)

            # we should also check for external paths here how do we specify that an entry is a path ?

            # check if there are any duplicates in supervision in the same data_obj
            seen = set()
            for sup in c_obj["supervision"]:
                t = tuple(
                    (k, str(v)) for (k, v) in sup.items()
                )  # tuplefy lists to make em hashable
                if t not in seen:
                    seen.add(t)
                else:
                    raise KeyError(
                        "Supervision for data object ID {} contains duplicates please remove them".format(
                            data_obj_id
                        )
                    )

            assert (
                len(c_obj["supervision"]) > 0
            ), "At least one supervision should be specified for each data obj"

            for sup in c_obj["supervision"]:
                assert (
                    list(sup.keys()) == first_sup_keys
                ), "All supervision must have same fields within a data object and must be ordered in same way"
                # assert all supervisions in all

                for sup_name in sup.keys():
                    assert isinstance(
                        sup[sup_name],
                        (tuple, list, float, int, bool, str, dict),
                    ), "Format not supported"

    for d_split in dataset_splits:
        for data_obj_id in d_split.keys():
            c_obj = d_split[data_obj_id]
            for sup in c_obj["supervision"]:
                # dataset have same supervisions
                # if start and stop are not specified we assume that all file is used
                # we make it explicit and take start and stop from waveforms.
                if not all(k in sup.keys() for k in ("start", "stop")):
                    sup["start"] = 0  # we modify it in place
                    sup["stop"] = min(c_obj["waveforms"]["lengths"])

                elif all(k in sup.keys() for k in ("start", "stop")):
                    pass
                else:
                    raise EnvironmentError(
                        "You can't specify only start or stop. Either specify both or none of the two"
                    )

        # how we specify external path dependencies ?


def to_ASR_format(dataset):
    """
    Converts general dataset format to ASR format where we have a list of unique utterances.
    """

    utterances = []
    for data_obj_id in dataset.keys():
        for supervision in dataset[data_obj_id]["supervision"]:
            # we "reverse" the format
            utterances.append(
                {
                    "supervision": supervision,
                    "waveforms": dataset[data_obj_id]["waveforms"],
                }
            )

    return utterances


class CategoricalEncoder:
    def __init__(
        self, data_collections: (list, dict), supervision: str, encode_to="int"
    ):

        assert encode_to in ["int", "onehot"]
        self.encode_to = encode_to

        if isinstance(data_collections, dict):
            data_collections = [data_collections]

        all_labs = set()
        for data_coll in data_collections:
            for data_obj_key in data_coll:
                data_obj = data_coll[data_obj_key]
                for sup in data_obj["supervision"]:
                    for sup_key in sup.keys():
                        if sup_key == supervision:
                            if isinstance(sup[sup_key], (list, tuple)):
                                all_labs.update(set(sup[sup_key]))
                            elif isinstance(sup[sup_key], (str)):
                                all_labs.add(sup[sup_key])
                            else:
                                raise NotImplementedError

        all_labs = sorted(list(all_labs))  # sort alphabetically just in case

        self.lab2indx = {key: index for index, key in enumerate(all_labs)}
        self.indx2lab = {key: index for key, index in enumerate(all_labs)}

    def intEncode(self, x, dtype=torch.long):
        if isinstance(x, (tuple, list)):  # x is list of strings or other things
            labels = []
            for i, elem in enumerate(x):
                labels.append(self.lab2indx[elem])
            labels = torch.tensor(labels, dtype=dtype)

        else:
            labels = torch.tensor([self.lab2indx[x]], dtype=dtype)

        return labels

    def intDecode(self, x: torch.Tensor):

        # labels are of shape batch, num_classes, arbitrary dims (e.g. sequence length)
        if x.ndim == 1:
            decoded = self.indx2lab[x.item()]
            return decoded

        elif x.ndim == 2:
            decoded = []
            indexes = torch.argmax(x, 0)
            for time_step in range(len(indexes)):
                decoded.append(self.indx2lab[indexes[time_step]])
            return decoded

        elif x.ndim == 3:  # batched tensor b, classes, steps
            decoded = []
            for batch in x.shape[0]:
                c_batch = []
                indexes = torch.argmax(x[batch], 0)
                for time_step in range(len(indexes)):
                    c_batch.append(self.indx2lab[indexes[time_step]])
                decoded.append(c_batch)
            return decoded
        else:
            raise NotImplementedError

    def oneHotEncode(self, x, dtype=torch.long):
        # TODO handle numpy arrays maybe ?
        if isinstance(x, (tuple, list)):
            # then we will return
            labels = torch.zeros(
                (len(self.lab2indx.keys()), len(x)), dtype=dtype
            )
            for i, elem in enumerate(x):
                labels[self.lab2indx[elem], i] = 1
        else:
            labels = torch.zeros((len(self.lab2indx.keys()), 1), dtype=dtype)

        return labels

    def oneHotDecode(self, x: torch.Tensor):
        # labels are of shape num_classes or num_classes, lengthofseq
        if x.ndim == 1:
            indx = torch.argmax(x)
            decoded = self.indx2lab[indx]
            return decoded

        elif x.ndim == 2:
            decoded = []
            indexes = torch.argmax(x, 0)
            for time_step in range(len(indexes)):
                decoded.append(self.indx2lab[indexes[time_step]])
            return decoded

        elif x.ndim == 3:  # batched tensor b, classes, steps
            decoded = []
            for batch in x.shape[0]:
                c_batch = []
                indexes = torch.argmax(x[batch], 0)
                for time_step in range(len(indexes)):
                    c_batch.append(self.indx2lab[indexes[time_step]])
                decoded.append(c_batch)
            return decoded
        else:
            raise NotImplementedError

    def encode(self, x):
        if self.encode_to == "int":
            return self.intEncode(x)
        elif self.encode_to == "onehot":
            return self.oneHotEncode(x)
        else:
            return NotImplementedError


def replace_entries(data_coll, replacements_dict):

    for data_obj_key in data_coll:
        data_obj = data_coll[data_obj_key]
        for sup in [*data_obj["supervision"], data_obj["waveforms"]]:
            for sup_key in sup.keys():
                if sup_key in replacements_dict.keys():
                    if isinstance(sup[sup_key], (str)):
                        mapping = replacements_dict[sup_key]
                        for map_key in mapping.keys():
                            # we replace in place
                            sup[sup_key] = sup[sup_key].replace(
                                map_key, mapping[map_key]
                            )
                    elif isinstance(sup[sup_key], (list, tuple)):
                        assert isinstance(
                            sup[sup_key][0], str
                        ), "Replacements supported only for str type, and unidimensional lists"
                        mapping = replacements_dict[sup_key]
                        for map_key in mapping.keys():
                            for indx in range(len(sup[sup_key])):
                                sup[sup_key][indx] = sup[sup_key][indx].replace(
                                    map_key, mapping[map_key]
                                )
                    else:
                        raise NotImplementedError
                    # check if we have to replace it

    return data_coll


def get_windowed_examples(dataset):
    """
    get examples for diarization, vad and other speech labelling applications
    from long files by using a sliding window with overlap.
    """
    pass


def filter_supervision(dataset):
    pass


def filter_waveforms(dataset):
    pass
