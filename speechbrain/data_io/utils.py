def dataset_sanity_check(dataset):
    # we check that supervision entries are unique

    # we check every supervision has same fields if start and stop are not specified
    # we imply they are the min of lengths in waveforms and add it there.

    # first supervision --> all supervisions should have the same exact entries
    first_key = list(dataset.keys())[0]
    first_sup_keys = dataset[first_key]["supervision"].keys()


    for data_obj_id in dataset.keys():
        c_obj = dataset[data_obj_id]
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
        assert isinstance(c_obj["waveforms"]["lengths"], list)
        assert isinstance(c_obj["waveforms"]["lengths"], int)
        assert isinstance(c_obj["waveforms"]["samplerate"], int)

        # check if there are any duplicates in supervision
        assert len(c_obj["supervision"]) == len(
            set(c_obj["supervision"])
        ), "Supervision for data object ID {} contains duplicates please remove them".format(
            data_obj_id
        )

        assert len(c_obj["supervision"]) > 0, "At least one supervision should be specified for each data obj"

        for sup in c_obj["supervision"]:
            assert sup.keys() == first_sup_keys
            if not all(k in sup.keys() for k in ("start", "stop")):
                sup["start"] = 0  # we modify it in place
                sup["stop"] = min(c_obj["waveforms"]["lengths"])

            elif all(k in sup.keys() for k in ("start", "stop")):
                pass
            else:
                raise EnvironmentError(
                    "You can't specify only start and stop. Either specify both or none of the two"
                )


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
