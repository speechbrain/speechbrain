from torch.utils.data import Dataset


class SegmentedDataset(Dataset):

    """
    SegmentedDataset handles pre-segmented datasets.

    It takes in input a dict in which each entry is a single example.
    The example in its turn is represented by using a dict:
    e.g. {"wav_file": "/path/to/helloworld.wav" "words": ["hello", "world"]}.

    in data_fields a list of elements which one wants to be returned by this dataset class __getitem__
    method must be specifies e.g. data_fields = ["wav_file", "words"].
    Corresponding data_transformation can be specified for each element we want to return.
    Note that transformations include also reading data from a file:
    e.g. {"wav_file": read_wav} where read_wav is a suitable function we provide in data_io.py.

    Finally the specified elements are tranformed an returned from __getitem__
    in a dict where the keys corresponds to the data_fields entried e.g. "wav_file".


     ---------
    Examples : dict
        Dictionary containing single examples (e.g. utterances).
    data_fields : (list, tuple)
        The class to use for updating the modules' parameters.
    data_transforms : dict
        Dictionary where data transforms for each field is specified.
    """

    def __init__(
        self,
        examples: dict,
        data_fields: (list, tuple),
        data_transforms=None,
        length_sorting="original",
        discard_longer=None,
        discard_shorter=None,
    ):

        self.data_fields = data_fields
        self.data_transforms = data_transforms
        assert length_sorting in ["original", "ascending", "descending"]
        if length_sorting in ["ascending", "descending"]:
            assert (
                "length" in self.examples[self.examples.keys()[0]].keys()
            ), "If sorting != original then,' \
                   ' each example must have a 'length' key containing the length of the example in order to be able to sort them."
        self.sentence_sorting = length_sorting

        assert isinstance(self.data_transforms, dict)
        for k in self.data_transforms.keys():
            assert callable(
                self.data_transforms[k]
            ), "Each element in data_transforms dict must be callable"

        if discard_shorter:
            assert (
                "length" in self.examples[self.examples.keys()[0]].keys()
            ), "If discard_shorter option wants to be used, each example must have a 'length' key containing the length of the example."

            examples = {
                k: v
                for k, v in examples.items()
                if examples[k]["length"] >= discard_shorter
            }

        if discard_longer:
            assert (
                "length" in self.examples[self.examples.keys()[0]].keys()
            ), "If discard_shorter option wants to be used, each example must have a 'length' key containing the length of the example."

            examples = {
                k: v
                for k, v in examples.items()
                if examples[k]["length"] <= discard_longer
            }

        self.examples = examples
        self.ex_ids = list(self.examples.keys())

        # sorting operation
        if length_sorting == "ascending":
            self.ex_ids = sorted(
                self.ex_ids, key=lambda x: self.examples[x]["length"]
            )
        elif length_sorting == "descending":
            self.ex_ids = sorted(
                examples,
                key=lambda x: self.examples[x]["length"],
                reverse=True,
            )
        else:
            pass

    def __len__(self):
        return len(self.ex_ids)

    def __getitem__(self, item):
        ex_id = self.ex_ids[item]
        c_ex = self.examples[ex_id]
        out = {"id": ex_id}

        for k in c_ex.keys():
            if k in self.data_fields:
                out[k] = self.data_transforms[k](c_ex[k])

        return out
