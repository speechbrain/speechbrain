import os
from copy import copy


def dataset_sanity_check(dataset_splits):
    """
    This function performs sanity check over a DataCollection.
    STILL WIP BECAUSE WE MIGHT CHANGE THE DATACOLLECTION FORMAT.

    Parameters
    ----------
    dataset_splits

    Returns
    -------

    """

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
