import torch
import pickle
import logging

logger = logging.getLogger(__name__)


class CategoricalEncoder(object):
    """
    Categorical encoder object. It is used to encode labels to a categorical distribution.
    It is thus suitable for speaker recognition where for example you have speakers
    labels e.g. spk1, spk2, spk3 and you want corresponding labels 0, 1, 2 to be able to train a classifier with CrossEntropy loss.
    """

    def __init__(self):

        self.lab2indx = {}
        self.indx2lab = {}

    def __len__(self):
        return len(self.lab2indx)

    def fit(
        self,
        data_collections: (list, dict),
        supervision: str,
        sup_transform=None,
    ):

        if isinstance(data_collections, dict):
            data_collections = [data_collections]

        def _recursive_helper(dictionary, supervision, labels_set):

            for k in dictionary.keys():
                if isinstance(dictionary[k], dict):
                    if k == supervision:
                        raise NotImplementedError(
                            "Desired supervision must be either list, tuple or string, Not Supported, got: {}".format(
                                type(dictionary[k])
                            )
                        )
                    _recursive_helper(dictionary[k], supervision, labels_set)
                else:
                    # leaf node contains no dict
                    if k == supervision:
                        if isinstance(dictionary[k], (list, tuple)):
                            if sup_transform is not None:
                                all_labs.update(
                                    set(sup_transform(dictionary[k]))
                                )
                            else:
                                all_labs.update(set(dictionary[k]))
                        elif isinstance(dictionary[k], (str)):
                            if sup_transform is not None:
                                transformed = sup_transform(
                                    dictionary[k]
                                )  # we make sure is hashable
                                if isinstance(transformed, (list, tuple)):
                                    all_labs.update(set(transformed))
                                else:
                                    all_labs.add(transformed)
                            else:
                                all_labs.add(dictionary[k])
                        else:
                            raise NotImplementedError(
                                "Desired supervision must be either list, tuple or string, Not Supported, got: {}".format(
                                    type(dictionary[k])
                                )
                            )

        all_labs = set()
        for data_coll in data_collections:
            _recursive_helper(data_coll, supervision, all_labs)

        if not len(self.lab2indx.keys()) and not len(self.indx2lab.keys()):
            self.lab2indx = {key: index for index, key in enumerate(all_labs)}
            self.indx2lab = {key: index for key, index in enumerate(all_labs)}
        else:
            raise EnvironmentError(
                "lab2indx and indx2lab must be empty, "
                "please use fit right after object instantiation."
            )

    def add_elem(self, label, index=None):
        """
        Update method (adding additional labels not present in the encoder internal state.

        Note
        ----------
        If no element is provided we simply append to end of label dictionary:
        e.g. there are 40 (from 0 to 39) labels already, for the added label the encoding value (index) will be 40.

        Arguments
        ----------
        label : str
            new label we want to add.

        index : int, optional
            corresponding integer value for the label.

        Returns
        -------
        None
        """
        if label in self.lab2indx.keys():
            raise KeyError("Label already present in label dictionary")

        max_indx = list(self.indx2lab.keys())[-1]
        if index is None or index == (max_indx + 1):
            self.lab2indx[label] = max_indx + 1
            self.indx2lab[max_indx + 1] = index
        else:
            if not (0 <= index <= (max_indx + 1)):
                raise IndexError(
                    "Invalid value specified choose between 0 and {}, got {}".format(
                        len(self.lab2indx), index
                    )
                )

            self.lab2indx[label] = index
            orig_key = self.indx2lab[index]
            self.lab2indx[orig_key] = max_indx + 1
            self.indx2lab[max_indx + 1] = orig_key
            self.indx2lab[index] = label

    def update(self, d: dict):
        for k, v in d.items():
            self.add_elem(k, v)

    def _index_label_dict(self, label_dict, k):

        return label_dict[k]

    def encode_label(self, x):
        """
        Parameters
        ----------
        x : str
            list, tuple of strings or either a single string which one wants to encode.

        Returns
        -------
        labels : int
            corresponding encoded int value.

        """
        if isinstance(x, str):
            labels = self._index_label_dict(self.lab2indx, x)
        else:
            raise NotImplementedError(
                "Value to encode must be a string, got {}".format(type(x))
            )
        return labels

    def encode_sequence(self, x):
        """
        Parameters
        ----------
        x : (list, tuple)
            list, tuple of strings which one wants to encode one by one e.g.
            every element of the sequence is a label and we encode it.

        Returns
        -------
        labels : torch.Tensor
            tensor containing encoded value.

        """

        if isinstance(x, (tuple, list)):
            labels = list(map(self.encode_label, x))
        else:
            raise NotImplementedError(
                "Value to encode must be a list or tuple, got {}".format(
                    type(x)
                )
            )
        return labels

    def decode_int(self, x):
        """
        Decodes a torch.Tensor or list of ints to a list of corresponding labels.

        Arguments
        ----------
        x : (torch.Tensor, list, tuple)
            Torch tensor or list containing int values which has to be decoded to original labels strings.
            Torch tensors must be 1D with shape (seq_len,) or 2D with shape (batch, seq_len).
            List and tuples must be one-dimensional or bi-dimensional.

        Returns
        -------
        decoded : list
            list containing original labels (list of str).
        """
        if not len(x):
            return []

        if isinstance(x, torch.Tensor):

            if x.ndim == 1:  # 1d tensor
                decoded = []
                for time_step in range(len(x)):
                    decoded.append(self.indx2lab[x[time_step].item()])
                return decoded

            elif x.ndim == 2:
                batch = []  # batch, steps
                batch_indx = x.size(0)
                for b in range(batch_indx):
                    c_example = []
                    for time_step in range(len(x[b])):
                        c_example.append(self.indx2lab[x[b, time_step].item()])
                    batch.append(c_example)
                return batch

            else:
                raise NotImplementedError(
                    "Only 1D and 2D tensors are supported got tensor with ndim={}".format(
                        x.ndim
                    )
                )

        elif isinstance(x, (tuple, list)):
            if isinstance(x[0], (list, tuple)):  # 2D list
                batch = []  # classes, steps
                for b in range(len(x)):
                    c_example = []
                    for time_step in range(len(x[b])):
                        c_example.append(self.indx2lab[x[b][time_step]])
                    batch.append(c_example)
                return batch
            else:  # 1D list
                decoded = []
                for time_step in range(len(x)):
                    decoded.append(self.indx2lab[x[time_step].item()])
                return decoded

        else:
            raise TypeError(
                "Input must be a torch.Tensor or list or tuple, got {}".format(
                    type(x)
                )
            )

    def decode_one_hot(self, x):
        """

        Arguments
        ----------
        x : torch.Tensor
            One-hot encoding torch.Tensor which has to be decoded to original labels strings.
            Accepts 1D tensor of shape (n_classes) 2D tensor of shape (n_classes, time_steps)
            and 3D tensor of shape (batch, n_classes, time_steps).

        Returns
        -------
        decoded : list
            list containing original labels (strings).
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Input must be a torch.Tensor, got {}".format(type(x))
            )

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
            raise NotImplementedError(
                "Only 1D, 2D and 3D tensors are supported got tensor with ndim={}".format(
                    x.ndim
                )
            )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.lab2indx, self.indx2lab), f)

    def load(self, path):
        if len(self.lab2indx) or len(self.indx2lab):
            raise RuntimeError(
                "Use load right after instantiation, "
                "lab2indx and indx2lab must be empty otherwise this operation will overwrite them."
            )

        with open(path, "rb") as f:
            self.lab2indx, self.indx2lab = pickle.load(f)


class TextEncoder(CategoricalEncoder):
    def __init__(self):

        super(TextEncoder, self).__init__()
        self.blank_token = None
        self.unk_token = None
        self.bos_token = None
        self.eos_token = None

    def add_blank(self, encoding=None, token="<blank>"):
        self.blank_token = token
        self.add_elem(token, encoding)

    def add_unk(self, encoding=None, token="<unk>"):
        self.unk_token = token
        self.add_elem(token, encoding)

    def add_bos_eos(
        self,
        bos_encoding=None,
        bos_token="<bos>",
        eos_encoding=None,
        eos_token="<eos>",
    ):
        if not eos_encoding or (eos_encoding == bos_encoding):
            logger.debug(
                "Only BOS token specified or BOS == EOS, EOS is set implictly equal to BOS",
                exc_info=True,
            )
            self.bos_token = bos_token
            self.eos_token = bos_token
            self.add_elem(bos_token, bos_encoding)
        else:
            self.update({bos_token: bos_encoding, eos_token: eos_encoding})
            self.bos_token = bos_token
            self.eos_token = eos_token

    def _index_label_dict(self, label_dict, k):
        if self.unk_token:
            try:
                out = label_dict[k]
            except KeyError:
                out = label_dict[self.unk_token]
            return out
        else:
            try:
                return label_dict[k]
            except KeyError:
                raise KeyError(
                    "Token {} can't be encoded because it is not in the encoding dictionary, "
                    "either something was meesed up during data preparation or, "
                    "if this happens in test consider using the <unk>, unknown fallback symbol".format(
                        k
                    )
                )

    def prepend_bos(self, x: list):
        if isinstance(x, list):
            return [self.bos_token] + x
        else:
            raise TypeError(
                "Curently only inputs of type: list are supported, got {}".format(
                    type(x)
                )
            )

    def append_eos(self, x: list):
        if isinstance(x, list):
            return x.append(self.eos_token)
        else:
            raise TypeError(
                "Curently only inputs of type: list are supported, got {}".format(
                    type(x)
                )
            )


class CTCTextEncoder(TextEncoder):
    def __init__(self):
        super().__init__()

    @classmethod
    def fit_from_yaml(
        cls,
        data_collections,
        *args,
        blank_encoding=0,
        blank_token="<blank>",
        unk_encoding=1,
        unk_token=None,
        **kwargs,
    ):
        enc = cls()

        enc.fit(data_collections, *args, **kwargs)
        enc.add_blank(blank_encoding, blank_token)
        if unk_token is not None:
            enc.add_unk(unk_encoding, unk_token)
        return enc
