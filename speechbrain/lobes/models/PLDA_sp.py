"""A popular speaker recognition/diarization model.

Authors
 * Nauman Dawalatabad 2020
 * Anthony Larcher 2020

References
 - This implemenatation is based of following papers.

 - PLDA model Training
    * Ye Jiang et. al, "PLDA Modeling in I-Vector and Supervector Space for Speaker Verification," in Interspeech, 2012.
    * Patrick Kenny et. al, "PLDA for speaker verification with utterances of arbitrary duration," in ICASSP, 2013.

 - PLDA scoring (fast scoring)
    * Daniel Garcia-Romero et. al, “Analysis of i-vector length normalization in speaker recognition systems,” in Interspeech, 2011.
    * Weiwei-LIN et. al, "Fast Scoring for PLDA with Uncertainty Propagation," in Odyssey, 2016.
    * Kong Aik Lee et. al, "Multi-session PLDA Scoring of I-vector for Partially Open-Set Speaker Detection," in Interspeech 2013.

Credits
    Most parts of this code is directly adapted from:
    https://git-lium.univ-lemans.fr/Larcher/sidekit
"""

import numpy
import copy

# from numpy import linalg
from scipy import linalg
from speechbrain.utils.Xvector_PLDA_sp import StatObject_SB  # noqa F401
from speechbrain.utils.Xvector_PLDA_sp import Ndx  # noqa F401
from speechbrain.utils.Xvector_PLDA_sp import Scores  # noqa F401


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
    e_hh: tensor
        accumulator
    """
    rank = factor_analyser.F.shape[1]
    if factor_analyser.Sigma.ndim == 2:
        A = factor_analyser.F.T.dot(factor_analyser.F)
        inv_lambda_unique = dict()
        for sess in numpy.unique(stat0[:, 0]):
            inv_lambda_unique[sess] = linalg.inv(
                sess * A + numpy.eye(A.shape[0])
            )

    tmp = numpy.zeros(
        (factor_analyser.F.shape[1], factor_analyser.F.shape[1]),
        dtype=numpy.float64,
    )

    for idx in mini_batch_indices:
        if factor_analyser.Sigma.ndim == 1:
            inv_lambda = linalg.inv(
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


def _check_missing_model(enroll, test, ndx):
    # Remove missing models and test segments
    clean_ndx = ndx.filter(enroll.modelset, test.segset, True)

    # Align StatServers to match the clean_ndx
    enroll.align_models(clean_ndx.modelset)
    test.align_segments(clean_ndx.segset)

    return clean_ndx


def fast_PLDA_scoring(
    enroll,
    test,
    ndx,
    mu,
    F,
    Sigma,
    test_uncertainty=None,
    Vtrans=None,
    p_known=0.0,
    scaling_factor=1.0,
    check_missing=True,
):
    """Compute the PLDA scores between to sets of vectors. The list of
    trials to perform is given in an Ndx object. PLDA matrices have to be
    pre-computed. i-vectors/x-vectors are supposed to be whitened before.

    Arguments
    ---------
    enroll: speechbrain.utils.Xvector_PLDA_sp.StatObject_SB
        a StatServer in which stat1 are xvectors
    test: speechbrain.utils.Xvector_PLDA_sp.StatObject_SB
        a StatServer in which stat1 are xvectors
    ndx: speechbrain.utils.Xvector_PLDA_sp.Ndx
        an Ndx object defining the list of trials to perform
    mu: double
        the mean vector of the PLDA gaussian
    F: tensor
        the between-class co-variance matrix of the PLDA
    Sigma: tensor
        the residual covariance matrix
    p_known: float
        probability of having a known speaker for open-set
        identification case (=1 for the verification task and =0 for the
        closed-set case)
    check_missing: bool
        if True, check that all models and segments exist
    """

    enroll_ctr = copy.deepcopy(enroll)
    test_ctr = copy.deepcopy(test)

    # If models are not unique, compute the mean per model, display a warning
    if not numpy.unique(enroll_ctr.modelset).shape == enroll_ctr.modelset.shape:
        # logging.warning("Enrollment models are not unique, average i-vectors")
        enroll_ctr = enroll_ctr.mean_stat_per_model()

    # Remove missing models and test segments
    if check_missing:
        clean_ndx = _check_missing_model(enroll_ctr, test_ctr, ndx)
    else:
        clean_ndx = ndx

    # Center the i-vectors around the PLDA mean
    enroll_ctr.center_stat1(mu)
    test_ctr.center_stat1(mu)

    # If models are not unique, compute the mean per model, display a warning
    if not numpy.unique(enroll_ctr.modelset).shape == enroll_ctr.modelset.shape:
        # logging.warning("Enrollment models are not unique, average i-vectors")
        enroll_ctr = enroll_ctr.mean_stat_per_model()

    # Compute constant component of the PLDA distribution
    invSigma = linalg.inv(Sigma)
    I_spk = numpy.eye(F.shape[1], dtype="float")

    K = F.T.dot(invSigma * scaling_factor).dot(F)
    K1 = linalg.inv(K + I_spk)
    K2 = linalg.inv(2 * K + I_spk)

    # Compute the Gaussian distribution constant
    alpha1 = numpy.linalg.slogdet(K1)[1]
    alpha2 = numpy.linalg.slogdet(K2)[1]
    plda_cst = alpha2 / 2.0 - alpha1

    # Compute intermediate matrices
    Sigma_ac = numpy.dot(F, F.T)
    Sigma_tot = Sigma_ac + Sigma
    Sigma_tot_inv = linalg.inv(Sigma_tot)

    Tmp = linalg.inv(Sigma_tot - Sigma_ac.dot(Sigma_tot_inv).dot(Sigma_ac))
    Phi = Sigma_tot_inv - Tmp
    Psi = Sigma_tot_inv.dot(Sigma_ac).dot(Tmp)

    # Compute the different parts of PLDA score
    model_part = 0.5 * numpy.einsum(
        "ij, ji->i", enroll_ctr.stat1.dot(Phi), enroll_ctr.stat1.T
    )
    seg_part = 0.5 * numpy.einsum(
        "ij, ji->i", test_ctr.stat1.dot(Phi), test_ctr.stat1.T
    )

    # Compute verification scores
    score = Scores()  # noqa F821
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask

    score.scoremat = model_part[:, numpy.newaxis] + seg_part + plda_cst
    score.scoremat += enroll_ctr.stat1.dot(Psi).dot(test_ctr.stat1.T)
    score.scoremat *= scaling_factor

    # Case of open-set identification, we compute the log-likelihood
    # by taking into account the probability of having a known impostor
    # or an out-of set class
    if p_known != 0:
        N = score.scoremat.shape[0]
        open_set_scores = numpy.empty(score.scoremat.shape)
        tmp = numpy.exp(score.scoremat)
        for ii in range(N):
            # open-set term
            open_set_scores[ii, :] = score.scoremat[ii, :] - numpy.log(
                p_known * tmp[~(numpy.arange(N) == ii)].sum(axis=0) / (N - 1)
                + (1 - p_known)
            )
        score.scoremat = open_set_scores

    return score


class PLDA:
    """A class to train PLDA model from ivectors/xvector
    The input is in speechbrain.utils.StatObject_SB format.
    Trains a simplified PLDA model no within class covariance matrix but full residual covariance matrix.

    Arguments
    ---------
    input_file_name: str
        file to read model from
    mean: 1d tensor
        mean of the vectors
    F: tensor
        Eigenvoice matrix
    Sigma: tensor
        Residual matrix

    Example
    -------
    >>> from speechbrain.utils.Xvector_PLDA_sp import StatObject_SB, Ndx, Scores
    >>> from PLDA_sp import *
    >>> data_dir = "/Users/nauman/Desktop/Mila/nauman/Data/xvect-sdk/sb-format/"
    >>> train_file = data_dir + "VoxCeleb1_training_rvectors.pkl"
    >>> with open(train_file, "rb") as xvectors:
    ...     train_obj = pickle.load(xvectors)
    ...
    >>> plda = PLDA()
    >>> plda.plda(train_obj)
    >>> enrol_file = data_dir + "VoxCeleb1_enrol_rvectors.pkl"
    >>> test_file = data_dir + "VoxCeleb1_test_rvectors.pkl"
    >>> ndx_file = data_dir + "ndx.pkl"
    >>> with open(enrol_file, "rb") as xvectors:
    ...     enrol_obj = pickle.load(xvectors)
    ...
    >>> with open(test_file, "rb") as xvectors:
    ...     test_obj = pickle.load(xvectors)
    ...
    >>> with open(ndx_file, "rb") as ndxes:
    ...     ndx_obj = pickle.load(ndxes)
    ...
    >>> scores_plda = fast_PLDA_scoring(enrol_obj, test_obj, ndx_obj, plda.mean, plda.F, plda.Sigma)
    >>> scores_plda.scoremat[:3, :3]
    array([[-2.98146610e+07,  7.81558818e+07, -5.62018466e+07],
       [ 1.80313207e+08, -4.90753877e+08,  3.39299061e+08],
       [ 1.54824606e+08, -4.21029694e+08,  2.89427688e+08]])
    """

    def __init__(self, mean=None, F=None, Sigma=None):
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
        # Just incase someone wants to save it.
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
        """Trains PLDA model with no within class covariance matrix but full residual covariance matrix.

        Arguments
        ---------
        stat_server:
            object of speechbrain.utils.Xvector_PLDA_sp.StatObject_SB
        rank_f:
            rank of the between class covariance matrix
        nb_iter:
            number of iterations to run
        scaling_factor:
            scaling factor to downscale statistics (value bewteen 0 and 1)
        output_file_name: name of the output file where to store PLDA model
            save_partial: boolean, if True, save PLDA model after each iteration
        """

        # Dimension of the vector (x-vectors stored in stat1)
        vect_size = stat_server.stat1.shape[1]  # noqa F841

        # Initialize mean and residual covariance from the training data
        self.mean = stat_server.get_mean_stat1()
        self.Sigma = stat_server.get_total_covariance_stat1()

        # Sum stat0 and stat1 for each speaker model
        model_shifted_stat, session_per_model = stat_server.sum_stat_per_model()

        # Number of speakers (classes) in training set
        class_nb = model_shifted_stat.modelset.shape[0]

        # Multiply statistics by scaling_factor
        model_shifted_stat.stat0 *= scaling_factor
        model_shifted_stat.stat1 *= scaling_factor
        session_per_model *= scaling_factor

        # Covariance for stat1
        sigma_obs = stat_server.get_total_covariance_stat1()
        evals, evecs = linalg.eigh(sigma_obs)

        # Initial F (eigen voice matrix) from rank
        idx = numpy.argsort(evals)[::-1]
        evecs = evecs.real[:, idx[:rank_f]]
        self.F = evecs[:, :rank_f]

        # Estimate PLDA model by iterating the EM algorithm
        for it in range(nb_iter):

            # E-step
            print(
                f"\nE-step: Estimate between class covariance, it {it+1} / {nb_iter}"
            )

            # Copy stats as they will be whitened with a different Sigma for each iteration
            local_stat = copy.deepcopy(model_shifted_stat)

            # Whiten statistics (with the new mean and Sigma)
            local_stat.whiten_stat1(self.mean, self.Sigma)

            # Whiten the EigenVoice matrix
            eigen_values, eigen_vectors = linalg.eigh(self.Sigma)
            ind = eigen_values.real.argsort()[::-1]
            eigen_values = eigen_values.real[ind]
            eigen_vectors = eigen_vectors.real[:, ind]
            sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values.real)
            sqr_inv_sigma = numpy.dot(
                eigen_vectors, numpy.diag(sqr_inv_eval_sigma)
            )
            self.F = sqr_inv_sigma.T.dot(self.F)

            # Replicate self.stat0
            # index_map = numpy.zeros(vect_size, dtype=int)
            # _stat0 = local_stat.stat0[:, index_map]
            index_map = numpy.zeros(vect_size, dtype=int)
            _stat0 = local_stat.stat0[:, index_map]

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

            _C = e_h.T.dot(local_stat.stat1).dot(linalg.inv(sqr_inv_sigma))
            _A = numpy.einsum("ijk,i->jk", e_hh, local_stat.stat0.squeeze())

            # M-step
            print("M-step: Updating F and Sigma")
            self.F = linalg.solve(_A, _C).T

            # Update the residual covariance
            self.Sigma = sigma_obs - self.F.dot(_C) / session_per_model.sum()

            # Minimum Divergence step
            self.F = self.F.dot(linalg.cholesky(_R))
