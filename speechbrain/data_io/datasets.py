from torch.utils.data import Dataset
from utils import to_ASR_format, dataset_sanity_check, replace_entries
from data_io_new import read_audio_example
from ruamel import yaml


class ASRDataset(
    Dataset
):  # note the user can override the methods when needed # e.g. he wants to return also the channels used
    # or for example if speech is near-field or far-field

    def __init__(
        self,
        dataset,
        supervisions,
        encoding_funcs,
        sentence_sorting="original",
        discard_longer=None,
        discard_shorter=None,
    ):

        if not isinstance(supervisions, (list, tuple)):
            supervisions = [supervisions]
        self.supervisions = supervisions
        if not isinstance(encoding_funcs, (list, tuple)):
            encoding_funcs = [encoding_funcs]
        self.encoding_funcs = encoding_funcs
        self.sentence_sorting = (
            sentence_sorting  # we will use this when wrapping
        )
        # with DataLoader to prevent shuffling when != original

        assert sentence_sorting in [
            "ascending",
            "descending",
            "original",
        ]

        self.sanity_check(
            dataset
        )  # verify dataset is consistent with the specified ASR task

        examples = to_ASR_format(
            dataset
        )  # convert to utterances list for the ASR task

        # filtering operation -> very easy because of how we have defined annotation
        # first filter then sort

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

        # label creation step is in __getitem__ now
        self.examples = examples

    def sanity_check(
        self, examples
    ):  # maybe this can be more general --> put it into a template class ?

        # TODO
        # maybe check config samplerate == samplerate of source files--> not all recipes have samplerate specified
        # i would make it mandatory because it helps avoiding stupid mistakes.

        for _, ex in examples.items():
            for supervision in ex["supervision"]:
                for req_sup in self.supervisions:
                    assert req_sup in supervision.keys(), (
                        "Requested supervision entry is not in the dataset."
                        "Available supervisions are {}".format(
                            list(ex["supervision"].keys())
                        )
                    )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        # common for every ASR task i think.. just read the audio
        audio = read_audio_example(self.examples[item])  # (channels, samples)

        labels = {k: [] for k in self.supervisions}
        # we will return supervisions in the order they are specified
        for i in range(len(self.supervisions)):
            c_sup = self.supervisions[i]
            # handles only one supervision per example --> standard ASR
            # for more complex tasks like 2 speakers ASR we will have two supervisions
            labels[c_sup].append(
                self.encoding_funcs[i].encode(
                    self.examples[item]["supervision"][c_sup]
                )
            )

        # padding of labels and examples will be handled by dataloader
        return (audio, labels)


if __name__ == "__main__":

    with open(
        "/media/sam/bx500/speechbrain_minimalVAD/speechbrain/samples/audio_samples/nn_training_samples/dev.yaml",
        "r",
    ) as f:
        devset = yaml.safe_load(f)

    # we can put this in other places than utils
    from utils import CategoricalEncoder

    encoder = CategoricalEncoder(devset, "phones")
    replacements_dict = {
        "files": {
            "DATASET_ROOT": "/media/sam/bx500/speechbrain_minimalVAD/speechbrain/samples/audio_samples/nn_training_samples"
        },
        "alignment_file": {
            "ALIGNMENT_ROOT": "/media/sam/bx500/speechbrain_minimalVAD/speechbrain/samples/audio_samples/nn_training_samples"
        },
    }
    devset = replace_entries(devset, replacements_dict)
    dataset_sanity_check([devset, devset])  # sanity check for dev
    dataset = ASRDataset(devset, "phones", encoder)
    dataset[0]
