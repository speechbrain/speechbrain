import torch


class CategoricalEncoder:
    """
    Categorical encoder takes in input a list or one DataCollection file:
    (yaml file), a supervision field you want to encode.
    init_dict is an optional argument in which you can pass a dict for
    initializing the label dict e.g. {"<blank>": 0}
    """

    def __init__(
        self,
        data_collections: (list, dict),
        supervision: str,
        encode_to="int",
        init_dict=None,
    ):

        # in the init the datacollections are parsed in oder to contruct the
        # label dict.

        assert encode_to in ["int", "onehot"]
        self.encode_to = encode_to

        if isinstance(data_collections, dict):
            data_collections = [data_collections]

        all_labs = set()
        for data_coll in data_collections:
            for data_obj_key in data_coll:
                data_obj = data_coll[data_obj_key]
                for sup in data_obj["supervision"]:
                    for sup_key in sup.keys():
                        if sup_key == supervision:
                            if isinstance(sup[sup_key], (list, tuple)):
                                all_labs.update(set(sup[sup_key]))
                            elif isinstance(sup[sup_key], (str)):
                                all_labs.add(sup[sup_key])
                            else:
                                raise NotImplementedError

        if init_dict:
            self.lab2indx = init_dict
            self.indx2lab = {index: key for key, index in init_dict.items()}
        else:
            self.lab2indx = {}
            self.indx2lab = {}

        self.lab2indx.update({key: index for index, key in enumerate(all_labs)})
        self.indx2lab.update({key: index for key, index in enumerate(all_labs)})

    def update(self, key, elem=None):
        """
        update method (adding additional keys not present in the data collection
        Parameters
        ----------
        key
        elem

        Returns
        -------

        """
        # if no element is provided we simply append to end of label dictionary
        max_indx = list(self.indx2lab.keys())[-1]
        if elem is None or elem == (max_indx + 1):
            self.lab2indx[key] = max_indx + 1
            self.indx2lab[max_indx + 1] = key
        else:
            # if elem is provided we have to check if the range is valid
            # from 0 to max_indx + 1
            # we enforce label indx to be continuos
            assert 0 <= elem <= (max_indx + 1)
            self.lab2indx[key] = elem
            orig_key = self.indx2lab[elem]
            self.lab2indx[orig_key] = max_indx + 1  # append old element on tail
            self.indx2lab[max_indx + 1] = orig_key
            # NOTE: this is efficient but order is not respected
            # we could have odd-looking labeldicts

    def intEncode(self, x):
        if isinstance(x, (tuple, list)):  # x is list of strings or other things
            labels = []
            for i, elem in enumerate(x):
                labels.append(self.lab2indx[elem])
            labels = torch.tensor(labels, dtype=torch.long)

        else:
            labels = torch.tensor([self.lab2indx[x]], dtype=torch.long)

        return labels

    def intDecode(self, x: torch.Tensor):

        # labels are of shape batch, num_classes, arbitrary dims (e.g. sequence length)
        # maybe we should supprt arbitrary dims.
        if x.ndim == 1:
            decoded = self.indx2lab[x.item()]
            return decoded

        elif x.ndim == 2:
            decoded = []
            indexes = torch.argmax(x, 0)
            for time_step in range(len(indexes)):
                decoded.append(self.indx2lab[indexes[time_step]])
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

    def oneHotEncode(self, x):
        # TODO handle numpy arrays maybe ?
        if isinstance(x, (tuple, list)):
            # then we will return
            labels = torch.zeros(
                (len(self.lab2indx.keys()), len(x)), dtype=torch.long
            )
            for i, elem in enumerate(x):
                labels[self.lab2indx[elem], i] = 1
        else:
            labels = torch.zeros(
                (len(self.lab2indx.keys()), 1), dtype=torch.long
            )

        return labels

    def oneHotDecode(self, x: torch.Tensor):
        # labels are of shape num_classes or num_classes, lengthofseq
        if x.ndim == 1:
            indx = torch.argmax(x)
            decoded = self.indx2lab[indx]
            return decoded

        elif x.ndim == 2:
            decoded = []
            indexes = torch.argmax(x, 0)
            for time_step in range(len(indexes)):
                decoded.append(self.indx2lab[indexes[time_step]])
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

    def encode(self, x):
        if self.encode_to == "int":
            return self.intEncode(x)
        elif self.encode_to == "onehot":
            return self.oneHotEncode(x)
        else:
            return NotImplementedError
