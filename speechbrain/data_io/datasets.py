from torch.utils.data import Dataset
from .utils import to_ASR_format, dataset_sanity_check
from .data_io_new import read_audio_example


class ASRDataset(Dataset):
    def __init__(
        self,
        dataset_root_dir,
        dataset,
        sentence_sorting="original",
        discard_longer=None,
        discard_shorter=None,
    ):

        assert sentence_sorting in [
            "ascending",
            "descending",
            "original",
        ]  # note how to force that when sentence sorting is on we disable shuffling ?

        # where do we specify absolute paths ? either here or in __getitem__ i would say it is better here.
        dataset_sanity_check(
            dataset
        )  # verify dataset is consistent, paths exists etc

        examples = to_ASR_format(dataset)  # convert to utterances list


        # check specific to ASR task: verify each example at least contains words or phones
        for ex in examples:
            assert (
                "words" in ex["supervision"].keys()
                or "phones" in ex["supervision"].keys()
            ), "To perform ASR task you need to provide words or phones supervision "

        # filtering operation
        if discard_shorter:
            examples = filter(
                lambda x: ["supervision"]["stop"] - x["supervision"]["start"]
                >= int(discard_shorter * x["waveforms"]["samplerate"]),
                examples,
            )
        if discard_longer:
            examples = filter(
                lambda x: ["supervision"]["stop"] - x["supervision"]["start"]
                <= int(discard_longer * x["waveforms"]["samplerate"]),
                examples,
            )

        # sorting operation
        if sentence_sorting == "ascending":
            examples = sorted(
                examples,
                key=lambda x: x["supervision"]["stop"]
                - x["supervision"]["start"],
            )
        elif sentence_sorting == "descending":
            examples = sorted(
                examples,
                key=lambda x: x["supervision"]["stop"]
                - x["supervision"]["start"],
                reverse=True,
            )
        else:
            pass

        # label creation step
        #TODO

        self.examples = examples


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        audio = read_audio_example(self.examples[item])  # (samples, channels)

        # we read labels for this example.
        #TODO
