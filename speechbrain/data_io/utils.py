import os
import torch
from ruamel import yaml
from collections.abc import MutableMapping


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
    first_sup_keys = dataset_splits[0][first_key]["supervision"].keys()

    for d_split in dataset_splits:
        for data_obj_id in d_split.keys():
            c_obj = d_split[data_obj_id]
            # we check that waveforms have at least one file, channels, lengths, and samplerate
            assert all(
                k in c_obj["waveforms"].keys()
                for k in ("files", "channels", "lengths", "samplerate")
            )
            # assert are not empty and are of proper type
            assert isinstance(c_obj["waveforms"]["files"], list)
            assert isinstance(c_obj["waveforms"]["files"][0], str)
            assert isinstance(c_obj["waveforms"]["channels"], list)
            assert isinstance(c_obj["waveforms"]["channels"][0], list)
            assert isinstance(c_obj["waveforms"]["lengths"], int)
            assert isinstance(c_obj["waveforms"]["samplerate"], int)

            # assert files exists
            for f in c_obj["waveforms"]["files"]:
                assert os.path.exists(f), "{} does not exist".format(f)

            # check if there are any duplicates in supervision
            assert len(c_obj["supervision"]) == len(
                set(c_obj["supervision"])
            ), "Supervision for data object ID {} contains duplicates please remove them".format(
                data_obj_id
            )

            assert (
                len(c_obj["supervision"]) > 0
            ), "At least one supervision should be specified for each data obj"

            for sup in c_obj["supervision"]:
                assert (
                    sup.keys() == first_sup_keys
                )  # assert all supervisions in all
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
                        "You can't specify only start and stop. Either specify both or none of the two"
                    )

                # ideally we should also check for other paths to exist here
                # e.g. alignments.
                # how we specify external path dependencies ?


def to_ASR_format(dataset):
    """
    Converts general dataset format to ASR format where we have a list of unique utterances.
    """

    utterances = []
    for data_obj_id in dataset.keys():
        for supervision in dataset[data_obj_id]["supervision"]:
            utt_id = list(supervision.keys())[0]
            # we "reverse" the format
            utterances.append(
                {
                    "supervision": supervision[utt_id],
                    "waveforms": dataset[data_obj_id]["waveforms"],
                }
            )

    return utterances


class CategoricalEncoder:
    def __init__(self, data_collections: (list, dict), supervision: str):
        if isinstance(data_collections, dict):
            data_collections = [data_collections]

        all_labs = set()
        for data_coll in data_collections:
            for data_obj_key in data_coll:
                data_obj = data_coll[data_obj_key]
                for sup in data_obj["supervision"]:
                    for sup_key in sup.keys():
                        if sup_key == supervision:
                            if isinstance(
                                sup[sup_key], (list, tuple)
                            ):
                                all_labs.update(set(sup[sup_key]))
                            elif isinstance(sup[sup_key], (str)):
                                all_labs.add(sup[sup_key])
                            else:
                                raise NotImplementedError

        all_labs = sorted(list(all_labs))  # sort alphabetically just in case

        self.lab2indx = {
            key: index for index, key in enumerate(all_labs)
        }
        self.indx2lab = {
            key: index for key, index in enumerate(all_labs)
        }

    def encode_labels(self, x, dtype=torch.long):
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

    def decode_one_hot(self, x: torch.Tensor):
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


def replace_entries(data_coll, replacements_dict):

    for data_obj_key in data_coll:
        data_obj = data_coll[data_obj_key]
        for sup in data_obj["supervision"]:
            for sup_key in sup.keys():
                if sup_key in replacements_dict.keys():
                    if isinstance(sup[sup_key], (str)):
                        mapping = replacements_dict[sup_key]
                        for map_key in mapping.keys():
                            # we replace in place
                            sup[sup_key].replace(map_key, mapping[map_key])
                    elif isinstance(sup[sup_key], (list, tuple)):
                        assert isinstance(sup[sup_key][0], str), "Replacements "
                    else:
                        raise NotImplementedError
                    # check if we have to replace it




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
