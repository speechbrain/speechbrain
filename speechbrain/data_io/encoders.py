import torch
import pickle
import warnings


class CategoricalEncoder:
    """
    Categorical encoder object. It is used to encode labels to a categorical distribution.
    It is thus suitable for speaker recognition where for example you have speakers
    labels e.g. spk1, spk2, spk3 and you want corresponding labels 0, 1, 2 to be able to train a classifier with CrossEntropy loss.
    Arguments:
        init_dict: (dict, optional)
            dictionary with labels and corresponding integer encoding.
            Note must be ordered and first item must start with integer encoding 0.
    """

    def __init__(
        self, init_dict=None,
    ):

        if init_dict:
            # check that init dict has contiguos values
            max_indx = max([init_dict[x] for x in init_dict.keys()])
            min_indx = min([init_dict[x] for x in init_dict.keys()])
            assert (
                min_indx == 0
            ), "init dict first key must correspond to 0 index."
            assert [init_dict[x] for x in init_dict.keys()] == [
                x for x in range(min_indx, max_indx + 1)
            ], "Init dictionary must have contigouos categorical values."
            self.lab2indx = init_dict
            self.indx2lab = {index: key for key, index in init_dict.items()}

    def fit(self, data_collections: (list, dict), supervision: str):

        if isinstance(data_collections, dict):
            data_collections = [data_collections]

        def _recursive_helper(dictionary, supervision, labels_set):

            for k in dictionary.keys():
                if isinstance(dictionary[k], dict):
                    assert (
                        k != supervision
                    ), "desired supervision must not contain a dict, Not Supported"
                    _recursive_helper(dictionary[k], supervision, labels_set)
                else:
                    # leaf node contains no dict
                    if k == supervision:
                        if isinstance(dictionary[k], (list, tuple)):
                            all_labs.update(set(dictionary[k]))
                        elif isinstance(dictionary[k], (str)):
                            all_labs.add(dictionary[k])
                        else:
                            raise NotImplementedError

        all_labs = set()
        for data_coll in data_collections:
            _recursive_helper(data_coll, supervision, all_labs)

        if not len(self.lab2indx.keys()) and not len(self.indx2lab.keys()):
            self.lab2indx = {key: index for index, key in enumerate(all_labs)}
            self.indx2lab = {key: index for key, index in enumerate(all_labs)}
        elif len(self.lab2indx.keys()) != len(self.indx2lab.keys()):
            raise EnvironmentError(
                "Indx2lab and lab2indx should have same number of keys, something has gone wrong."
            )
        else:
            # max index till now
            max_indx_init = (
                max([self.lab2indx[x] for x in self.lab2indx.keys()]) + 1
            )
            self.lab2indx.update(
                {
                    key: index + max_indx_init
                    for index, key in enumerate(all_labs)
                }
            )
            self.indx2lab.update(
                {
                    key + max_indx_init: index
                    for key, index in enumerate(all_labs)
                }
            )

    def add_elem(self, key, elem=None):
        """
        update method (adding additional keys not present in the encoder internal state.
        Parameters
        ----------
        key: str
            new key/label that one desires to add to the label_dictionary.

        elem: int, optional
            corresponding integer value for the key/label

        Returns
        -------
        None

        """
        # if no element is provided we simply append to end of label dictionary
        assert (
            key not in self.lab2indx.keys()
        ), "Label already present in label dictionary"
        max_indx = list(self.indx2lab.keys())[-1]
        if elem is None or elem == (max_indx + 1):
            self.lab2indx[key] = max_indx + 1
            self.indx2lab[max_indx + 1] = key
        else:
            assert (
                0 <= elem <= (max_indx + 1)
            ), "invalid value specified choose between 0 and len(Encoder)"
            self.lab2indx[key] = elem
            orig_key = self.indx2lab[elem]
            self.lab2indx[orig_key] = max_indx + 1
            self.indx2lab[max_indx + 1] = orig_key
            self.indx2lab[elem] = key

    def _index_label_dict(self, label_dict, k):

        return label_dict[k]

    def encode_int(self, x: (tuple, list, str)):
        """
        Parameters
        ----------
        x: (list, tuple, str)
            list, tuple of strings or either a single string which one wants to encode.

        Returns
        -------
        labels: torch.Tensor
            tensor containing encoded value.

        """
        if isinstance(x, (tuple, list)):
            labels = []
            for i, elem in enumerate(x):
                labels.append(self._index_label_dict(self.lab2indx, elem))
            labels = torch.tensor(labels, dtype=torch.long)

        elif isinstance(x, str):
            labels = torch.tensor(
                [self._index_label_dict(self.lab2indx, x)], dtype=torch.long
            )
        else:
            raise NotImplementedError
        return labels

    def decode_int(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x: torch.Tensor
         tensor containing int values which has to be decoded to original labels strings.
         Accepts either 1d tensor or 2d tensors where first dimension is batch.

        Returns
        -------
        decoded: list
            list containing original labels (strings).
        """

        decoded = []
        if x.ndim == 1:  # 1d tensor
            for time_step in range(len(x)):
                decoded.append(self.indx2lab[x[time_step].item()])
            return decoded

        elif x.ndim == 2:
            batch = []  # classes, steps
            batch_indx = x.size(0)
            for b in range(batch_indx):
                c_example = []
                for time_step in range(len(x[b])):
                    c_example.append(self.indx2lab[x[b, time_step].item()])
                batch.append(c_example)
            return batch

        else:
            raise NotImplementedError

    def decode_one_hot(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x: torch.Tensor
         one-hot encodeing tensor which has to be decoded to original labels strings.
         Accepts 1d tensor of shape (n_classes) 2d tensor of shape (n_classes, time_steps)
         and 3d tensor of shape (batch, n_classes, time_steps)

        Returns
        -------
        decoded: list
            list containing original labels (strings).
        """

        if x.ndim == 1:
            indx = torch.argmax(x)
            decoded = self.indx2lab[indx.item()]
            return decoded

        elif x.ndim == 2:
            decoded = []
            indexes = torch.argmax(x, 0)  # classes, steps
            for time_step in range(len(indexes)):
                decoded.append(self.indx2lab[indexes[time_step].item()])
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

    def save(self, path):
        warnings.warn(
            "Using pickle right now but we may want to move to something better for big datasets"
        )
        with open(path, "wb") as f:
            pickle.dump((self.lab2indx, self.indx2lab), f)

    def load(self, path):

        with open(path, "rbb") as f:
            self.lab2indx, self.indx2lab = pickle.load(f)


class TextEncoder(CategoricalEncoder):
    def __init__(
        self,
        blank_symbol=("<blank>", 0),
        unknown_symbol=("<unknown>", 1),
        bos_symbol=None,
        eos_symbol=None,
        init_dict=None,
    ):

        init_dict = {}
        self.blank_symbol = blank_symbol
        if self.blank_symbol:
            init_dict[self.blank_symbol[0]] = self.blank_symbol[1]
        self.unknown_symbol = unknown_symbol
        if self.unknown_symbol:
            init_dict[self.unknown_symbol[0]] = self.unknown_symbol[1]
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol

        if bool(bos_symbol) != bool(eos_symbol):  # xor
            print("BOS and EOS symbols are set equal if only one is specified.")
            tmp = self.bos_symbol or self.eos_symbol
            self.eos_symbol = tmp
            self.bos_symbol = tmp
            init_dict[self.eos_symbol[0]] = self.eos_symbol[1]
            init_dict[self.bos_symbol[0]] = self.bos_symbol[1]
        elif bool(bos_symbol) and bool(eos_symbol):
            init_dict[self.eos_symbol[0]] = self.eos_symbol[1]
            init_dict[self.bos_symbol[0]] = self.bos_symbol[1]

        super(TextEncoder, self).__init__(init_dict)

    def _index_label_dict(self, label_dict, k):
        if self.unknown_symbol:
            try:
                out = label_dict[k]
            except KeyError:
                out = label_dict[self.unknown_symbol[0]]
            return out
        else:
            return label_dict[k]

    def encode_int(self, x: (tuple, list, str)):
        if self.bos_symbol:
            if isinstance(x, str):
                x = [self.bos_symbol[0], x, self.eos_symbol[0]]
            else:
                x = [self.bos_symbol[0]] + [x] + [self.eos_symbol[0]]
            return super(TextEncoder, self).encode_int(x)
        else:
            return super(TextEncoder, self).encode_int(x)
