from torch.utils.data import Dataset


class ASRDataset(Dataset):

    """
    DATA FORMAT AGNOSTIC, could be a base class or we can wrap it for common tasks.
    data_fields: name of entries you want to return from __getitem__
    data_transforms: dict of transform pipelines applied of corresponding
    data_filed e.g. {"phns": PhnsTF} where PhnsTF must be a function (e.g. CategoricalEncoder.encode).
    other args are from older pipeline.
    """

    def __init__(
        self,
        examples: dict,
        data_fields: (list, tuple),
        data_transforms=None,
        length_sorting="original",
        discard_longer=None,  # in seconds
        discard_shorter=None,  # in seconds
    ):

        self.data_fields = data_fields
        self.data_transforms = data_transforms
        self.sentence_sorting = length_sorting  # we will use this when wrapping

        assert isinstance(self.data_transforms, dict)
        for k in self.data_transforms.keys():
            assert callable(
                self.data_transforms[k]
            ), "Each element in data_transforms dict must be callable"

        assert length_sorting in [
            "ascending",
            "descending",
            "original",
        ]

        # filtering operation -> very easy because of how we have defined annotation
        # first filter then sort

        if discard_shorter:
            examples = filter(
                lambda x: x["stop"] - x["start"]  # these now are in ms.
                >= int(discard_shorter * 1e3),
                examples,
            )
        if discard_longer:
            examples = filter(
                lambda x: x["stop"] - x["start"] <= int(discard_longer * 1e3),
                examples,
            )

        # sorting operation
        if length_sorting == "ascending":
            examples = sorted(examples, key=lambda x: x["stop"] - x["start"],)
        elif length_sorting == "descending":
            examples = sorted(
                examples, key=lambda x: x["stop"] - x["start"], reverse=True,
            )
        else:
            pass  # original

        self.examples = examples
        self.ex_ids = list(self.examples.keys())

    def __len__(self):
        # length of dataset is simply length  of examples dict.
        return len(self.ex_ids)

    def __getitem__(self, item):
        ex_id = self.ex_ids[item]
        c_ex = self.examples[ex_id]
        out = {"id": ex_id}

        for k in c_ex.keys():
            if k in self.data_fields:
                out[k] = self.data_transforms[k](c_ex[k])

        return out
