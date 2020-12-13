"""Encoding categorical data as integers

Authors
  * Samuele Cornell 2020
  * Aku Rouhe 2020
"""
import ast
import torch
import itertools
import logging

logger = logging.getLogger(__name__)


class CategoricalEncoder:
    """
    Encode labels of a discrete set.

    Used for encoding text as well as e.g. speaker identities in speaker
    recognition.
    """

    VALUE_SEPARATOR = " => "
    EXTRAS_SEPARATOR = "================\n"

    def __init__(self, starting_index=0):
        self.lab2ind = {}
        self.ind2lab = {}
        self.starting_index = starting_index

    def __len__(self):
        return len(self.lab2ind)

    def update_from_iterable(self, iterable, sequence_input=False):
        """Update from iterator

        Arguments
        ---------
        iterable : iterable
            Input sequence on which to operate.
        sequence_input : bool
            Whether iterable yields sequences of labels or individual labels
            directly. False by default.
        """
        if sequence_input:
            label_iterator = itertools.chain.from_iterable(iterable)
        else:
            label_iterator = iter(iterable)
        for label in label_iterator:
            if label not in self.lab2ind:
                self.add_label(label)

    def update_from_didataset(
        self, didataset, output_key, sequence_input=False
    ):
        """Update from DynamicItemDataset

        Arguments
        ---------
        didataset : DynamicItemDataset
            Dataset on which to operate.
        output_key : str
            Key in the dataset (in data or a dynamic item) to encode.
        sequence_input : bool
            Whether the data yielded with the specified key consists of
            sequences of labels or individual labels directly.
        """
        with didataset.output_keys_as([output_key]):
            self.update_from_iterable(
                (data_point[output_key] for data_point in didataset),
                sequence_input=sequence_input,
            )

    def add_label(self, label):
        """Add new label to the encoder, at the next free position.

        Arguments
        ---------
        label : hashable
            Most often labels are str, but anything that can act as dict key is
            supported. Note that default save/load only supports Python
            literals.

        Returns
        -------
        int
            The index that was used to encode this label.
        """
        if label in self.lab2ind:
            clsname = self.__class__.__name__
            raise KeyError(f"Label already present in {clsname}")
        index = self._next_index()
        self.lab2ind[label] = index
        self.ind2lab[index] = label
        return index

    def insert_label(self, label, index):
        """Add a new label, forcing its index to a specific value.

        If a label already has the specified index, it is moved to the end
        of the mapping.

        Arguments
        ---------
        label : hashable
            Most often labels are str, but anything that can act as dict key is
            supported. Note that default save/load only supports Python
            literals.
        index : int
            The specific index to use.
        """
        if label in self.lab2ind:
            clsname = self.__class__.__name__
            raise KeyError(f"Label already present in {clsname}")
        index = int(index)
        moving_label = False
        if index in self.ind2lab:
            saved_label = self.ind2lab[index]
            moving_label = True
        self.lab2ind[label] = index
        self.ind2lab[index] = label
        if moving_label:
            logger.warning(
                f"Moving label {repr(saved_label)} from index "
                f"{index}, because {repr(label)} was inserted at its place."
            )
            new_index = self._next_index()
            self.lab2ind[saved_label] = new_index
            self.ind2lab[new_index] = saved_label

    def _next_index(self):
        """The index to use for the next new label"""
        index = self.starting_index
        while index in self.ind2lab:
            index += 1
        return index

    def encode_label(self, label):
        """Encode label to int

        Arguments
        ---------
        label : hashable
            Label to encode, must exist in the mapping.

        Returns
        -------
        int
            Corresponding encoded int value.
        """
        return self.lab2ind[label]

    def encode_label_torch(self, label):
        """Encode label to torch.LongTensor

        Arguments
        ---------
        label : hashable
            Label to encode, must exist in the mapping.

        Returns
        -------
        torch.LongTensor
            Corresponding encoded int value.
            Tensor shape [1]
        """
        return torch.LongTensor([self.lab2ind[label]])

    def encode_sequence(self, sequence):
        """Encode a sequence of labels to list

        Arguments
        ---------
        x : iterable
            Labels to encode, must exist in the mapping.

        Returns
        -------
        list
            Corresponding integer labels
        """
        return [self.lab2ind[label] for label in sequence]

    def encode_sequence_torch(self, sequence):
        """Encode a sequence of labels to torch.LongTensor

        Arguments
        ---------
        x : iterable
            Labels to encode, must exist in the mapping.

        Returns
        -------
        torch.LongTensor
            Corresponding integer labels
            Tensor shape [len(sequence)]
        """
        return torch.LongTensor([self.lab2ind[label] for label in sequence])

    def decode_int(self, x):
        """
        Decodes a torch.Tensor or list of ints to a list of corresponding labels.

        Arguments
        ---------
        x : (torch.Tensor, list, tuple)
            Torch tensor or list containing int values which has to be decoded
            to original labels strings.  Torch tensors must be 1D with shape
            (seq_len,) or 2D with shape (batch, seq_len).  List and tuples must
            be one-dimensional or bi-dimensional.

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
                    decoded.append(self.ind2lab[x[time_step].item()])
                return decoded

            elif x.ndim == 2:
                batch = []  # batch, steps
                batch_ind = x.size(0)
                for b in range(batch_ind):
                    c_example = []
                    for time_step in range(len(x[b])):
                        c_example.append(self.ind2lab[x[b, time_step].item()])
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
                        c_example.append(self.ind2lab[x[b][time_step]])
                    batch.append(c_example)
                return batch
            else:  # 1D list
                decoded = []
                for time_step in range(len(x)):
                    decoded.append(self.ind2lab[x[time_step].item()])
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
            ind = torch.argmax(x)
            decoded = self.ind2lab[ind.item()]
            return decoded

        elif x.ndim == 2:
            decoded = []
            indexes = torch.argmax(x, 0)  # classes, steps
            for time_step in range(len(indexes)):
                decoded.append(self.ind2lab[indexes[time_step].item()])
            return decoded

        elif x.ndim == 3:  # batched tensor b, classes, steps
            decoded = []
            for batch in x.shape[0]:
                c_batch = []
                indexes = torch.argmax(x[batch], 0)
                for time_step in range(len(indexes)):
                    c_batch.append(self.ind2lab[indexes[time_step]])
                decoded.append(c_batch)
            return decoded
        else:
            raise NotImplementedError(
                "Only 1D, 2D and 3D tensors are supported got tensor with ndim={}".format(
                    x.ndim
                )
            )

    def save(self, path):
        """Save the categorical encoding for later use and recovery"""
        extras = self._get_extras()
        if "starting_index" in extras:
            raise ValueError(
                "The extra key starting_index is reserved by the "
                "CategoricalEncoder base class."
            )
        extras["starting_index"] = self.starting_index
        self._save_literal(path, self.lab2ind, extras)

    def load_if_possible(self, path):
        """Loads if possible, returns bool indicating if loaded or not."""
        if self.lab2ind:
            clsname = self.__class__.__name__
            raise RuntimeError(f"Load called, but {clsname} is not empty")
        try:
            lab2ind, ind2lab, extras = self._load_literal(path)
        except (FileNotFoundError, ValueError, SyntaxError):
            logger.debug(
                f"Would load categorical encoding from {path}, "
                "but could not."
            )
            return False
        self.lab2ind = lab2ind
        self.ind2lab = ind2lab
        # Starting index is stored with extras, but is part of the base class.
        self.starting_index = extras["starting_index"]
        del extras["starting_index"]
        self._set_extras(extras)
        # If we're here, load was a success!
        logger.debug(f"Loaded categorical encoding from {path}")
        return True

    def _get_extras(self):
        """Override this to provide any additional things to save"""
        return {}

    def _set_extras(self, extras):
        """Override this to e.g. load any extras needed"""
        pass

    @staticmethod
    def _save_literal(path, lab2ind, extras):
        """Save which is compatible with _load_literal"""
        with open(path, "w") as f:
            for label, ind in lab2ind.items():
                f.write(
                    repr(label)
                    + CategoricalEncoder.VALUE_SEPARATOR
                    + str(ind)
                    + "\n"
                )
            f.write(CategoricalEncoder.EXTRAS_SEPARATOR)
            for key, value in extras.items():
                f.write(
                    repr(key)
                    + CategoricalEncoder.VALUE_SEPARATOR
                    + repr(value)
                    + "\n"
                )
            f.flush()

    @staticmethod
    def _load_literal(path):
        """Load which supports Python literals as keys.

        This is considered safe for user input, as well (unlike e.g. pickle).
        """
        lab2ind = {}
        ind2lab = {}
        extras = {}
        with open(path) as f:
            # Load the label to index mapping (until EXTRAS_SEPARATOR)
            for line in f:
                if line == CategoricalEncoder.EXTRAS_SEPARATOR:
                    break
                literal, ind = line.strip().split(
                    CategoricalEncoder.VALUE_SEPARATOR
                )
                ind = int(ind)
                label = ast.literal_eval(literal)
                lab2ind[label] = ind
                ind2lab[ind] = label
            # Load the extras:
            for line in f:
                literal_key, literal_value = line.strip().split(
                    CategoricalEncoder.VALUE_SEPARATOR
                )
                key = ast.literal_eval(literal_key)
                value = ast.literal_eval(literal_value)
                extras[key] = value
        return lab2ind, ind2lab, extras


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
