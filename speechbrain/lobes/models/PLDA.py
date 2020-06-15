import torch  # noqa F401
import numpy  # noqa F401
import pickle
import sys
import copy

STAT_TYPE = numpy.float64


class StatObject_SB:
    """
    class for stat0 and stat1
    defining object structure
    NDToDo: Make this more generic and readable
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

    def print_serverstat_shapes(self, stat_obj):
        """
        ndf: remove this
        """
        print("MODLESET : ", stat_obj.modelset.shape)
        print("SEG-SET : ", stat_obj.segset.shape)
        print("START: ", stat_obj.start.shape)
        print("STOP: ", stat_obj.stop.shape)
        print("STAT0: ", stat_obj.stat0.shape)
        print("STAT1: ", stat_obj.stat1.shape)

    def get_mean_stat1(self):
        """Return the mean of first order statistics

        return: the mean array of the first order statistics.
        """
        mu = numpy.mean(self.stat1, axis=0)
        return mu

    def get_total_covariance_stat1(self):
        """Compute and return the total covariance matrix of the first-order
            statistics.

        :return: the total co-variance matrix of the first-order statistics
                as a ndarray.
        """
        C = self.stat1 - self.stat1.mean(axis=0)
        return numpy.dot(C.transpose(), C) / self.stat1.shape[0]

    def get_model_stat0(self, mod_id):
        """Return zero-order statistics of a given model

        param mod_id: ID of the model which stat0 will be returned

        return: a matrix of zero-order statistics as a ndarray
        """
        S = self.stat0[self.modelset == mod_id, :]
        return S

    def get_model_stat1(self, mod_id):
        """Return first-order statistics of a given model

        param mod_id: string, ID of the model which stat1 will be returned

        return: a matrix of first-order statistics as a ndarray
        """
        return self.stat1[self.modelset == mod_id, :]

    def sum_stat_per_model(self):
        """Sum the zero- and first-order statistics per model and store them
        in a new StatServer.

        return: a StatServer with the statistics summed per model
            AND numpy array with session_per_model (ND)
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

        # nd: For each model sum the stats
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
            print("sigma.ndim == 2")
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


def fa_model_loop(
    batch_start,
    mini_batch_indices,
    factor_analyser,
    stat0,
    stat1,
    e_h,
    e_hh,
    num_thread=1,
):
    """
    Methods that is called for PLDA estimation for parallelization on classes

    :param batch_start: index to start at in the list
    :param mini_batch_indices: indices of the elements in the list (should start at zero)
    :param factor_analyser: FactorAnalyser object
    :param stat0: matrix of zero order statistics
    :param stat1: matrix of first order statistics
    :param e_h: accumulator
    :param e_hh: accumulator
    :param num_thread: number of parallel process to run
    """
    rank = factor_analyser.F.shape[1]
    if factor_analyser.Sigma.ndim == 2:
        A = factor_analyser.F.T.dot(factor_analyser.F)
        inv_lambda_unique = dict()
        for sess in numpy.unique(stat0[:, 0]):
            inv_lambda_unique[sess] = numpy.linalg.inv(
                sess * A + numpy.eye(A.shape[0])
            )

    tmp = numpy.zeros(
        (factor_analyser.F.shape[1], factor_analyser.F.shape[1]),
        dtype=STAT_TYPE,
    )

    for idx in mini_batch_indices:
        if factor_analyser.Sigma.ndim == 1:
            inv_lambda = numpy.linalg.inv(
                numpy.eye(rank)
                + (factor_analyser.F.T * stat0[idx + batch_start, :]).dot(
                    factor_analyser.F
                )
            )
        else:
            inv_lambda = inv_lambda_unique[stat0[idx + batch_start, 0]]

        aux = factor_analyser.F.T.dot(stat1[idx + batch_start, :])
        numpy.dot(aux, inv_lambda, out=e_h[idx])
        e_hh[idx] = inv_lambda + numpy.outer(e_h[idx], e_h[idx], tmp)


class PLDA:
    """
    inputs: Any seg representation vectors (xvect, ivect, dvect etc)
    returns the PLDA model
    ToDo:
    > Define standard object structure
    > Get rid of dependencies
    > Add Object definer class
    > Add trainer
    > Add Utils for eig value, Chol decom
    > Add scoring module
    """

    def __init__(self, input_file_name=None, mean=None, F=None, Sigma=None):

        if input_file_name is not None:
            pass
        else:
            self.mean = None
            self.F = None
            self.Sigma = None

        if mean is not None:
            self.mean = mean
        if F is not None:
            self.F = F
        if Sigma is not None:
            self.Sigma = Sigma

    def _save_plda_model():
        # should have a common object structure
        pass

    def _read_plda_model():
        pass

    def plda(
        self,
        stat_server=None,
        rank_f=100,
        nb_iter=10,
        scaling_factor=1.0,
        output_file_name=None,
    ):

        vect_size = stat_server.stat1.shape[1]  # noqa F841

        # Initialize mean and residual covariance from the training data
        self.mean = stat_server.get_mean_stat1()
        self.Sigma = stat_server.get_total_covariance_stat1()

        model_shifted_stat, session_per_model = stat_server.sum_stat_per_model()
        class_nb = model_shifted_stat.modelset.shape[0]  # noqa F841
        model_shifted_stat.print_serverstat_shapes(model_shifted_stat)

        # Multiply statistics by scaling_factor
        model_shifted_stat.stat0 *= scaling_factor
        model_shifted_stat.stat1 *= scaling_factor
        session_per_model *= scaling_factor

        sigma_obs = stat_server.get_total_covariance_stat1()
        evals, evecs = numpy.linalg.eigh(
            sigma_obs
        )  # evals/evect are slightly different
        idx = numpy.argsort(evals)[::-1]
        evecs = evecs.real[:, idx[:rank_f]]
        self.F = evecs[:, :rank_f]

        # Estimate PLDA model by iterating the EM algorithm
        for it in range(nb_iter):
            print(f"Estimate between class covariance, it {it+1} / {nb_iter}")

            # E-step
            print("E_step")

            # Copy stats as they will be whitened with a different Sigma for each iteration
            local_stat = copy.deepcopy(model_shifted_stat)

            # Whiten statistics (with the new mean and Sigma)
            local_stat.whiten_stat1(self.mean, self.Sigma)

            # Whiten the EigenVoice matrix
            eigen_values, eigen_vectors = numpy.linalg.eigh(self.Sigma)
            ind = eigen_values.real.argsort()[::-1]
            eigen_values = eigen_values.real[ind]
            eigen_vectors = eigen_vectors.real[:, ind]
            sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values.real)
            sqr_inv_sigma = numpy.dot(
                eigen_vectors, numpy.diag(sqr_inv_eval_sigma)
            )
            self.F = sqr_inv_sigma.T.dot(self.F)

            # Replicate self.stat0
            index_map = numpy.zeros(vect_size, dtype=int)
            _stat0 = local_stat.stat0[:, index_map]

            # ND: Initialize matrices
            e_h = numpy.zeros((class_nb, rank_f))
            e_hh = numpy.zeros((class_nb, rank_f, rank_f))

            # loop on model id's
            fa_model_loop(
                batch_start=0,
                mini_batch_indices=numpy.arange(class_nb),
                factor_analyser=self,
                stat0=_stat0,
                stat1=local_stat.stat1,
                e_h=e_h,
                e_hh=e_hh,
                num_thread=1,
            )

            # Accumulate for minimum divergence step
            _R = numpy.sum(e_hh, axis=0) / session_per_model.shape[0]

            _C = e_h.T.dot(local_stat.stat1).dot(
                numpy.linalg.inv(sqr_inv_sigma)
            )
            _A = numpy.einsum("ijk,i->jk", e_hh, local_stat.stat0.squeeze())

            # M-step
            print("M-step")
            self.F = numpy.linalg.solve(_A, _C).T

            # Update the residual covariance
            self.Sigma = sigma_obs - self.F.dot(_C) / session_per_model.sum()

            # Minimum Divergence step
            self.F = self.F.dot(numpy.linalg.cholesky(_R))

            print("F: ", self.F)


if __name__ == "__main__":
    data_dir = "/Users/nauman/Desktop/Mila/nauman/Data/xvect-sdk/sb-format/"
    train_file = data_dir + "VoxCeleb1_training_rvectors.pkl"

    # read extracted vectors (xvect, ivect, dvect, etc.)
    with open(train_file, "rb") as xvectors:
        train_obj = pickle.load(xvectors)

    # Train the model
    plda = PLDA()
    plda.plda(train_obj)

    sys.exit()

    # Scoring
    enrol_file = data_dir + "VoxCeleb1_enrol_rvectors.pkl"
    test_file = data_dir + "VoxCeleb1_test_rvectors.pkl"
    with open(enrol_file, "rb") as xvectors:
        enrol_obj = pickle.load(xvectors)
    with open(test_file, "rb") as xvectors:
        test_obj = pickle.load(xvectors)

    plda.plda(enrol_obj)
    plda.plda(test_obj)
