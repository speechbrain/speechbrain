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
