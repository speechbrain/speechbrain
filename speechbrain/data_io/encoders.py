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

    def fit(self, data_collections: (list, dict), supervision: str):

        if isinstance(data_collections, dict):
            data_collections = [data_collections]

        def _recursive_helper(dictionary, supervision, labels_set):

            for k in dictionary.keys():
                if isinstance(dictionary[k], dict):
                    if k == supervision:
                        logger.error(
                            "desired supervision must not contain a dict, Not Supported",
                            exc_info=True,
                        )
                        raise NotImplementedError
                    _recursive_helper(dictionary[k], supervision, labels_set)
                else:
                    # leaf node contains no dict
                    if k == supervision:
                        if isinstance(dictionary[k], (list, tuple)):
                            all_labs.update(set(dictionary[k]))
                        elif isinstance(dictionary[k], (str)):
                            all_labs.add(dictionary[k])
                        else:
                            logger.error(
                                "Supervision must be either list, tuple or string, Not Supported",
                                exc_info=True,
                            )
                            raise NotImplementedError

        all_labs = set()
        for data_coll in data_collections:
            _recursive_helper(data_coll, supervision, all_labs)

        if not len(self.lab2indx.keys()) and not len(self.indx2lab.keys()):
            self.lab2indx = {key: index for index, key in enumerate(all_labs)}
            self.indx2lab = {key: index for key, index in enumerate(all_labs)}
        else:
            logger.error(
                "lab2indx and indx2lab must be empty, please use fit right after object instantiation.",
                exc_info=True,
            )
            raise EnvironmentError

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
        if key in self.lab2indx.keys():
            logger.error(
                "Label already present in label dictionary", exc_info=True
            )
            raise KeyError

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

    def update(self, d: dict):
        for k, v in d.items():
            self.add_elem(k, v)

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
            labels = labels
        elif isinstance(x, str):
            labels = [self._index_label_dict(self.lab2indx, x)]
        else:
            logger.error(
                "Value to encode must be list, tuple or string", exc_info=True
            )
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
         one-hot encoding tensor which has to be decoded to original labels strings.
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
        with open(path, "wb") as f:
            pickle.dump((self.lab2indx, self.indx2lab), f)

    def load(self, path):
        if len(self.lab2indx) or len(self.indx2lab):
            logger.error(
                "Use load right after instantiation, "
                "lab2indx and indx2lab must be empty otherwise this operation will overwrite them.",
                exc_info=True,
            )
            raise RuntimeError

        with open(path, "rb") as f:
            self.lab2indx, self.indx2lab = pickle.load(f)


class TextEncoder(CategoricalEncoder):
    def __init__(self,):

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
                logger.error(
                    "key {} can't be encoded because it is not in the encoder dictionary, "
                    "either something was meesed up during data preparation or, "
                    "if this happens in test consider using the <unkwown> fallback symbol".format(
                        k
                    ),
                    exc_info=True,
                )
                raise KeyError

    def prepend_bos(self, x: (tuple, list, str)):
        if isinstance(x, str):
            return [self.bos_token, x]
        else:
            return [self.bos_token] + x

    def append_eos(self, x: (tuple, list, str)):
        if isinstance(x, str):
            return [x, self.eos_token]
        else:
            return x + [self.eos_token]
