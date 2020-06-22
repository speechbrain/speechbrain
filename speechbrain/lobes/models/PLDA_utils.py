import numpy
import copy

STAT_TYPE = numpy.float64


class StatObject_SB:
    """A util class for PLDA class used for statistics calculations

    Arguments
    ---------
    modelset: list
        list of model IDs for each session as an array of strings
    segset: list
        the list of session IDs as an array of strings
    start: int
        index of the first frame of the segment
    stop: int
        index of the last frame of the segment
    stat0: tensor
        a ndarray of float64. Each line contains 0-order statistics
        from the corresponding session
    stat1: tensor
        a ndarray of float64. Each line contains 1-order statistics
        from the corresponding session

    """

    def __init__(
        self,
        modelset=None,
        segset=None,
        start=None,
        stop=None,
        stat0=None,
        stat1=None,
    ):

        if modelset is None:  # For creating empty stat server
            self.modelset = numpy.empty(0, dtype="|O")
            self.segset = numpy.empty(0, dtype="|O")
            self.start = numpy.empty(0, dtype="|O")
            self.stop = numpy.empty(0, dtype="|O")
            self.stat0 = numpy.array([], dtype=STAT_TYPE)
            self.stat1 = numpy.array([], dtype=STAT_TYPE)
        else:
            self.modelset = modelset
            self.segset = segset
            self.start = start
            self.stop = stop
            self.stat0 = stat0
            self.stat1 = stat1

    def __repr__(self):
        ch = "-" * 30 + "\n"
        ch += "modelset: " + self.modelset.__repr__() + "\n"
        ch += "segset: " + self.segset.__repr__() + "\n"
        ch += "seg start:" + self.start.__repr__() + "\n"
        ch += "seg stop:" + self.stop.__repr__() + "\n"
        ch += "stat0:" + self.stat0.__repr__() + "\n"
        ch += "stat1:" + self.stat1.__repr__() + "\n"
        ch += "-" * 30 + "\n"
        return ch

    def get_mean_stat1(self):
        """Return the mean of first order statistics
        """
        mu = numpy.mean(self.stat1, axis=0)
        return mu

    def get_total_covariance_stat1(self):
        """Compute and return the total covariance matrix of the first-order
            statistics.
        """
        C = self.stat1 - self.stat1.mean(axis=0)
        return numpy.dot(C.transpose(), C) / self.stat1.shape[0]

    def get_model_stat0(self, mod_id):
        """Return zero-order statistics of a given model

        Arguments
        ---------
        mod_id: str
            ID of the model which stat0 will be returned
        """
        S = self.stat0[self.modelset == mod_id, :]
        return S

    def get_model_stat1(self, mod_id):
        """Return first-order statistics of a given model

        Arguments
        ---------
        mod_id: str
            ID of the model which stat1 will be returned
        """
        return self.stat1[self.modelset == mod_id, :]

    def sum_stat_per_model(self):
        """Sum the zero- and first-order statistics per model and store them
        in a new StatServer.
        Returns a StatServer object with the statistics summed per model
            and a numpy array with session_per_model
        """
        # sts_per_model = sidekit.StatServer()
        sts_per_model = StatObject_SB()
        sts_per_model.modelset = numpy.unique(
            self.modelset
        )  # nd: get uniq spkr ids
        sts_per_model.segset = copy.deepcopy(
            sts_per_model.modelset
        )  # same as uniq spkr ids ^^
        sts_per_model.stat0 = numpy.zeros(
            (sts_per_model.modelset.shape[0], self.stat0.shape[1]),
            dtype=STAT_TYPE,
        )
        sts_per_model.stat1 = numpy.zeros(
            (sts_per_model.modelset.shape[0], self.stat1.shape[1]),
            dtype=STAT_TYPE,
        )
        sts_per_model.start = numpy.empty(
            sts_per_model.segset.shape, "|O"
        )  # ndf: restruture this
        sts_per_model.stop = numpy.empty(sts_per_model.segset.shape, "|O")

        session_per_model = numpy.zeros(numpy.unique(self.modelset).shape[0])

        # For each model sum the stats
        for idx, model in enumerate(sts_per_model.modelset):
            sts_per_model.stat0[idx, :] = self.get_model_stat0(model).sum(
                axis=0
            )
            sts_per_model.stat1[idx, :] = self.get_model_stat1(model).sum(
                axis=0
            )
            session_per_model[idx] += self.get_model_stat1(model).shape[0]
        return sts_per_model, session_per_model

    def center_stat1(self, mu):
        """Center first order statistics.

        :param mu: array to center on.
        """
        dim = self.stat1.shape[1] / self.stat0.shape[1]
        index_map = numpy.repeat(numpy.arange(self.stat0.shape[1]), dim)
        self.stat1 = self.stat1 - (
            self.stat0[:, index_map] * mu.astype(STAT_TYPE)
        )

    def rotate_stat1(self, R):
        """Rotate first-order statistics by a right-product.

        :param R: ndarray, matrix to use for right product on the first order
            statistics.
        """
        self.stat1 = numpy.dot(self.stat1, R)

    def whiten_stat1(self, mu, sigma, isSqrInvSigma=False):
        """Whiten first-order statistics
        If sigma.ndim == 1, case of a diagonal covariance
        If sigma.ndim == 2, case of a single Gaussian with full covariance
        If sigma.ndim == 3, case of a full covariance UBM

        :param mu: array, mean vector to be subtracted from the statistics
        :param sigma: narray, co-variance matrix or covariance super-vector
        :param isSqrInvSigma: boolean, True if the input Sigma matrix is the inverse of the square root of a covariance
         matrix
        """

        if sigma.ndim == 1:
            self.center_stat1(mu)
            self.stat1 = self.stat1 / numpy.sqrt(sigma.astype(STAT_TYPE))

        elif sigma.ndim == 2:
            # Compute the inverse square root of the co-variance matrix Sigma
            sqr_inv_sigma = sigma

            if not isSqrInvSigma:
                eigen_values, eigen_vectors = numpy.linalg.eigh(sigma)
                ind = eigen_values.real.argsort()[::-1]
                eigen_values = eigen_values.real[ind]
                eigen_vectors = eigen_vectors.real[:, ind]

                sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values.real)
                sqr_inv_sigma = numpy.dot(
                    eigen_vectors, numpy.diag(sqr_inv_eval_sigma)
                )
            else:
                pass

            # Whitening of the first-order statistics
            self.center_stat1(mu)  # CENTERING
            self.rotate_stat1(sqr_inv_sigma)

        elif sigma.ndim == 3:
            # we assume that sigma is a 3D ndarray of size D x n x n
            # where D is the number of distributions and n is the dimension of a single distibution
            n = self.stat1.shape[1] // self.stat0.shape[1]
            sess_nb = self.stat0.shape[0]
            self.center_stat1(mu)
            self.stat1 = (
                numpy.einsum(
                    "ikj,ikl->ilj", self.stat1.T.reshape(-1, n, sess_nb), sigma
                )
                .reshape(-1, sess_nb)
                .T
            )

        else:
            raise Exception("Wrong dimension of Sigma, must be 1 or 2")

    def align_models(self, model_list):
        """Align models of the current StatServer to match a list of models
            provided as input parameter. The size of the StatServer might be
            reduced to match the input list of models.

        :param model_list: ndarray of strings, list of models to match
        """
        indx = numpy.array(
            [numpy.argwhere(self.modelset == v)[0][0] for v in model_list]
        )
        self.segset = self.segset[indx]
        self.modelset = self.modelset[indx]
        self.start = self.start[indx]
        self.stop = self.stop[indx]
        self.stat0 = self.stat0[indx, :]
        self.stat1 = self.stat1[indx, :]

    def align_segments(self, segment_list):
        """Align segments of the current StatServer to match a list of segment
            provided as input parameter. The size of the StatServer might be
            reduced to match the input list of segments.

        :param segment_list: ndarray of strings, list of segments to match
        """
        indx = numpy.array(
            [numpy.argwhere(self.segset == v)[0][0] for v in segment_list]
        )
        self.segset = self.segset[indx]
        self.modelset = self.modelset[indx]
        self.start = self.start[indx]
        self.stop = self.stop[indx]
        self.stat0 = self.stat0[indx, :]
        self.stat1 = self.stat1[indx, :]


def diff(list1, list2):
    c = [item for item in list1 if item not in list2]
    c.sort()
    return c


def ismember(list1, list2):
    c = [item in list2 for item in list1]
    return c


class Ndx:
    """A class that encodes trial index information.  It has a list of
    model names and a list of test segment names and a matrix
    indicating which combinations of model and test segment are
    trials of interest.

    :attr modelset: list of unique models in a ndarray
    :attr segset:  list of unique test segments in a ndarray
    :attr trialmask: 2D ndarray of boolean. Rows correspond to the models
            and columns to the test segments. True if the trial is of interest.
    """

    def __init__(
        self, ndx_file_name="", models=numpy.array([]), testsegs=numpy.array([])
    ):
        """Initialize a Ndx object by loading information from a file
        in HDF5 or text format.

        :param ndx_file_name: name of the file to load
        """
        self.modelset = numpy.empty(0, dtype="|O")
        self.segset = numpy.empty(0, dtype="|O")
        self.trialmask = numpy.array([], dtype="bool")

        if ndx_file_name == "":
            modelset = numpy.unique(models)
            segset = numpy.unique(testsegs)

            trialmask = numpy.zeros(
                (modelset.shape[0], segset.shape[0]), dtype="bool"
            )
            for m in range(modelset.shape[0]):
                segs = testsegs[numpy.array(ismember(models, modelset[m]))]
                trialmask[m,] = ismember(segset, segs)  # noqa E231

            self.modelset = modelset
            self.segset = segset
            self.trialmask = trialmask
            assert self.validate(), "Wrong Ndx format"

        else:
            ndx = Ndx.read(ndx_file_name)
            self.modelset = ndx.modelset
            self.segset = ndx.segset
            self.trialmask = ndx.trialmask

    def filter(self, modlist, seglist, keep):
        """Removes some of the information in an Ndx. Useful for creating a
        gender specific Ndx from a pooled gender Ndx.  Depending on the
        value of \'keep\', the two input lists indicate the strings to
        retain or the strings to discard.

        :param modlist: a cell array of strings which will be compared with
                the modelset of 'inndx'.
        :param seglist: a cell array of strings which will be compared with
                the segset of 'inndx'.
        :param keep: a boolean indicating whether modlist and seglist are the
                models to keep or discard.

        :return: a filtered version of the current Ndx object.
        """
        if keep:
            keepmods = modlist
            keepsegs = seglist
        else:
            keepmods = diff(self.modelset, modlist)
            keepsegs = diff(self.segset, seglist)

        keepmodidx = numpy.array(ismember(self.modelset, keepmods))
        keepsegidx = numpy.array(ismember(self.segset, keepsegs))

        outndx = Ndx()
        outndx.modelset = self.modelset[keepmodidx]
        outndx.segset = self.segset[keepsegidx]
        tmp = self.trialmask[numpy.array(keepmodidx), :]
        outndx.trialmask = tmp[:, numpy.array(keepsegidx)]

        assert outndx.validate, "Wrong Ndx format"

        if self.modelset.shape[0] > outndx.modelset.shape[0]:
            print(
                "Number of models reduced from %d to %d"
                % self.modelset.shape[0],
                outndx.modelset.shape[0],
            )
        if self.segset.shape[0] > outndx.segset.shape[0]:
            print(
                "Number of test segments reduced from %d to %d",
                self.segset.shape[0],
                outndx.segset.shape[0],
            )
        return outndx

    def validate(self):
        """Checks that an object of type Ndx obeys certain rules that
        must always be true.

        :return: a boolean value indicating whether the object is valid
        """
        ok = isinstance(self.modelset, numpy.ndarray)
        ok &= isinstance(self.segset, numpy.ndarray)
        ok &= isinstance(self.trialmask, numpy.ndarray)

        ok &= self.modelset.ndim == 1
        ok &= self.segset.ndim == 1
        ok &= self.trialmask.ndim == 2

        ok &= self.trialmask.shape == (
            self.modelset.shape[0],
            self.segset.shape[0],
        )
        return ok


class Scores:
    """A class for storing scores for trials.  The modelset and segset
    fields are lists of model and test segment names respectively.
    The element i,j of scoremat and scoremask corresponds to the
    trial involving model i and test segment j.

    :attr modelset: list of unique models in a ndarray
    :attr segset: list of unique test segments in a ndarray
    :attr scoremask: 2D ndarray of boolean which indicates the trials of interest
            i.e. the entry i,j in scoremat should be ignored if scoremask[i,j] is False
    :attr scoremat: 2D ndarray of scores
    """

    def __init__(self, scores_file_name=""):
        """ Initialize a Scores object by loading information from a file HDF5 format.

        :param scores_file_name: name of the file to load
        """
        self.modelset = numpy.empty(0, dtype="|O")
        self.segset = numpy.empty(0, dtype="|O")
        self.scoremask = numpy.array([], dtype="bool")
        self.scoremat = numpy.array([])

        if scores_file_name == "":
            pass
        else:
            tmp = Scores.read(scores_file_name)
            self.modelset = tmp.modelset
            self.segset = tmp.segset
            self.scoremask = tmp.scoremask
            self.scoremat = tmp.scoremat

    def __repr__(self):
        ch = "modelset:\n"
        ch += self.modelset + "\n"
        ch += "segset:\n"
        ch += self.segset + "\n"
        ch += "scoremask:\n"
        ch += self.scoremask.__repr__() + "\n"
        ch += "scoremat:\n"
        ch += self.scoremat.__repr__() + "\n"
