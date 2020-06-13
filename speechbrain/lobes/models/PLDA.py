import torch  # noqa F401
import numpy  # noqa F401
import pickle


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

    def __init__(
        self,
        train_obj=None,
        rank_f=100,
        nb_iter=10,
        scaling_factor=1.0,
        output_file_name=None,
    ):

        self.train_obj = train_obj
        self.rank_f = rank_f
        self.nb_iter = nb_iter
        self.scaling_factor = scaling_factor
        self.output_file_name = output_file_name

    def _save_plda_model():
        # should have a common object structure
        pass

    def _read_plda_model():
        pass

    def _get_mean_stat1():
        pass

    def _get_total_covariance_stat1():
        pass

    def _sum_stat_per_model():
        pass

    def plda(self):
        # vect_size = stat_server.stat1.shape[1]
        print("Training PLDA started")
        pass


if __name__ == "__main__":
    data_dir = "/Users/nauman/Desktop/Mila/nauman/Data/xvect-sdk/sb-format/"
    train_file = data_dir + "VoxCeleb1_training_rvectors.pkl"
    enroll_file = data_dir + "VoxCeleb1_training_rvectors.pkl"
    test_file = data_dir + "VoxCeleb1_training_rvectors.pkl"

    # read extracted vectors (xvect, ivect, dvect, etc.)
    with open(train_file, "rb") as xvectors:
        train_obj = pickle.load(xvectors)

    plda = PLDA(train_obj)
    plda.plda()
