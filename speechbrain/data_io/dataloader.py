import torch
from collections import OrderedDict


# could be handy
class PaddedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x, extra_data, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, x, padding):
        # super().__init__() # optional
        self.orig_shape = padding  # list of shapes of length batch
        assert len(padding) == x.shape[0]

    def get_mask(self):
        mask = torch.ones_like(self).to(self.device)

        return mask


# TODO not sure if it can hurt performance actually
class TensorCollection(OrderedDict):
    def __init__(self, tensor_dict=None):
        super(TensorCollection, self).__init__()
        # basically an ordered_dict with extended functionality
        if not tensor_dict:
            self.tensors = []
            self.names = {}  # maps name to position in tensors
        else:
            indx = 0
            for k, item in tensor_dict.items():
                self.tensors.append(item)
                self.names[k] = indx
                indx += 1

        # we should override recursively some Tensor methods

    def to(self, device):
        for i in range(len(self.tensors)):
            self.tensors[i] = self.tensors[i].to(device)

    def float(self):
        for i in range(len(self.tensors)):
            self.tensors[i] = self.tensors[i].float()

    def __getitem__(self, item):

        if isinstance(item, (int)):
            return self.tensors[item]
        elif isinstance(item, (str)):
            return self.tensors[self.names[item]]
        else:
            raise IndexError

    def __setitem__(self, key, value):
        self.tensors.append(value)
        self.names[key] = len(self.tensors) - 1

    def append(self, value, name):
        self.__setitem__(name, value)

    def pop(self, position):
        if isinstance(position, (int)):
            pass

        elif isinstance(position, (str)):
            pass

    def __len__(self):
        return len(self.tensors)

    def __delitem__(self, key, value):
        pass

    def pad_n_stack(self):
        # return
        pass

    def trunc_n_stack(self):
        # return
        pass


def pad_tensor_dim(tensor, dim, left, right, mode="constant", value=0.0):

    unpadded_dims_right = [0 for i in range(tensor.ndim, dim + 1, -1)] * 2
    unpadded_dims_left = [0 for i in range(dim, 0, -1)] * 2
    padding = unpadded_dims_right + [left, right] + unpadded_dims_left

    return torch.nn.functional.pad(tensor, padding, mode=mode, value=value)


def pad_examples(batch, padding_value=0.0, return_padding_mask=False):

    # each batch has metadata
    num_batches = len(batch)
    num_perceptions = len(batch[0][1])
    num_labels = len(batch[0][2])

    meta = []
    for i in range(num_batches):
        meta.append(batch[i][0])

    lens = []
    perceptions = []
    for p_indx in range(num_perceptions):
        c_p_lens = []
        c_perceptions = []
        max_seq_len = max(
            [batch[b][1][p_indx].shape[-1] for b in range(num_batches)]
        )
        for b in range(num_batches):
            c_tensor = batch[b][1][p_indx]
            c_length = batch[b][1][p_indx].shape[-1]
            padded = torch.nn.functional.pad(
                c_tensor, (0, max_seq_len - c_length)
            )
            c_perceptions.append(padded)
            c_p_lens.append(c_length / max_seq_len)
        lens.append(c_p_lens)
        perceptions.append(torch.stack(c_perceptions))

    phn_lens = []
    supervisions = []
    for p_indx in range(num_labels):
        c_p_lens = []
        c_perceptions = []
        max_seq_len = max(
            [batch[b][2][p_indx].shape[-1] for b in range(num_batches)]
        )
        for b in range(num_batches):
            c_tensor = batch[b][2][p_indx]
            c_length = batch[b][2][p_indx].shape[-1]
            padded = torch.nn.functional.pad(
                c_tensor, (0, max_seq_len - c_length)
            )
            c_perceptions.append(padded)
            c_p_lens.append(c_length / max_seq_len)
        phn_lens.append(c_p_lens)
        supervisions.append(torch.stack(c_perceptions))

    # do the same for labels

    # we pad to max seq length and do in place substitution

    return (
        [meta, perceptions, torch.Tensor(lens).float()],
        [meta, supervisions, torch.Tensor(phn_lens).float()],
    )
