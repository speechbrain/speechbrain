from torch.utils.data import Dataset


def dataset_sanity_check(dataset):
    # check that utterances and data_obj_ids are unique
    pass


def to_ASR_format(dataset):

    utterances = {}
    for data_obj_id in dataset.keys():
        for supervision in dataset[data_obj_id]["supervision"]:
            utt_id = list(supervision.keys())[0]
            # we "reverse" the format
            utterances[utt_id] = {
                "supervision": supervision[utt_id],
                "waveforms": dataset[data_obj_id]["waveforms"],
            }

    return utterances


class ASRDataset(Dataset):
    def __init__(self, examples):
        # examples are assumed to be a list of utterances from yaml file
        self.examples = to_ASR_format(examples)

    def __len__(self):
        return len(self.examples.keys())

    def __getitem__(self, item):
        pass
