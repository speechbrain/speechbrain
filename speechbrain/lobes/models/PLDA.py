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
