import torch  # noqa F401
import numpy  # noqa F401
import pickle
import sys
import copy
from PLDA_StatServer import StatObject_SB  # noqa F401


def fa_model_loop(
    batch_start, mini_batch_indices, factor_analyser, stat0, stat1, e_h, e_hh,
):
    """A function that is called for PLDA estimation

    Arguments
    ---------
    batch_start: int
        index to start at in the list
    mini_batch_indices: list
        indices of the elements in the list (should start at zero)
    factor_analyser: instance of PLDA class
        PLDA class object
    stat0: tensor
        matrix of zero order statistics
    stat1: tensor
        matrix of first order statistics
    e_h: tensor
        accumulator
    e_hh: tensor (3-dim)
        accumulator
    """
    rank = factor_analyser.F.shape[1]
    if factor_analyser.Sigma.ndim == 2:
        A = factor_analyser.F.T.dot(factor_analyser.F)
        inv_lambda_unique = dict()
        for sess in numpy.unique(stat0[:, 0]):
            inv_lambda_unique[sess] = numpy.linalg.inv(
                sess * A + numpy.eye(A.shape[0])
            )

    STAT_TYPE = numpy.float64
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
    """A class to train PLDA model from ivectors/xvector

    Arguments
    ---------
    input_file_name: str
        file to read model from
    mean: 1d tensor
        mean of the vectors
    F: tensor
        Eigenvoice matrix
    Sigma: tensor
        Residual Matrix

    Example
    -------
    >>> from PLDA import *
    >>> data_dir = "/Users/nauman/Desktop/Mila/nauman/Data/xvect-sdk/sb-format/"
    >>> train_file = data_dir + "VoxCeleb1_training_rvectors.pkl"
    >>> with open(train_file, "rb") as xvectors:
    ...         train_obj = pickle.load(xvectors)
    >>> plda = PLDA()
    >>> plda.plda(train_obj)
    >>> plda.F
    array([[-0.15477624,  0.12481283, -0.11323124, ...,  0.01819566,
         0.00498689, -0.00215435],
       [ 0.0886069 ,  0.00111409,  0.08430482, ..., -0.00495019,
        -0.00952988,  0.0214457 ],
       [-0.06895772,  0.02911249,  0.23145705, ...,  0.01127238,
        -0.01799833, -0.01383415],
       ...,
       [-0.37370813,  0.11708417,  0.07775562, ...,  0.00396989,
         0.01001941, -0.0046404 ],
       [-0.0189982 , -0.83235048,  0.31378434, ...,  0.02885494,
         0.01114018,  0.01956349],
       [ 0.26698912, -0.03049991, -0.09061876, ...,  0.02371388,
         0.01258056,  0.01813147]])
    >>> plda.Sigma
    array([[ 0.7913822 , -0.09602436, -0.11224721, ...,  0.09969463,
        -0.10152465,  0.04684615],
       [-0.09602436,  0.77100682, -0.14149264, ...,  0.00922131,
         0.11627067, -0.08430017],
       [-0.11224721, -0.14149264,  1.20611489, ..., -0.03506288,
         0.265772  , -0.04872182],
       ...,
       [ 0.09969463,  0.00922131, -0.03506288, ...,  0.88060853,
         0.0897313 ,  0.01685596],
       [-0.10152465,  0.11627067,  0.265772  , ...,  0.0897313 ,
         6.49686837,  0.05217187],
       [ 0.04684615, -0.08430017, -0.04872182, ...,  0.01685596,
         0.05217187,  0.78122243]])
    """

    def __init__(self, input_file_name=None, mean=None, F=None, Sigma=None):

        if input_file_name is not None:
            # future
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
        # can remove scaling factor (without uncertainity propagation)
        vect_size = stat_server.stat1.shape[1]  # noqa F841

        # Initialize mean and residual covariance from the training data
        self.mean = stat_server.get_mean_stat1()
        self.Sigma = stat_server.get_total_covariance_stat1()  # global

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
        print("S: ", self.Sigma)


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

    """
    # Scoring
    enrol_file = data_dir + "VoxCeleb1_enrol_rvectors.pkl"
    test_file = data_dir + "VoxCeleb1_test_rvectors.pkl"
    with open(enrol_file, "rb") as xvectors:
        enrol_obj = pickle.load(xvectors)
    with open(test_file, "rb") as xvectors:
        test_obj = pickle.load(xvectors)

    plda.plda(enrol_obj)
    plda.plda(test_obj)
    """
