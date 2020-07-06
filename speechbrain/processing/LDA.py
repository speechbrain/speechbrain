import numpy  # noqa F401
import copy  # noqa F401
import pickle  # noqa F401

# from numpy import linalg
from scipy import linalg  # noqa F401


class LDA:
    """A class to perfrom Linear Discriminant Analysis (LDA) on vectorial representations.
    The input is in speechbrain.utils.StatObject_SB format.
    Return the projected vectors onto the latent space.

    Arguments
    ---------
    input_file_name: str
        file to read model from
    transformMat: tensor
        transformation matrix.
    reduced_dim: int
        reduced dimension of the space.
    """

    def __init__(self, mat=None, red_dim=2):
        self.transformMat = None
        self.reduced_dim = red_dim

        if mat is not None:
            self.transformMat = mat

    def project_vectors(self):
        pass

    def train_lda(
        self, stat_server=None,
    ):
        pass


if __name__ == "__main__":
    """
    Example (shift this to LDA)
    -------
    >>> from speechbrain.processing.PLDA import *
    >>> data_dir = "/Users/nauman/Desktop/Mila/nauman/Data/xvect-sdk/sb-format/"
    >>> train_file = data_dir + "VoxCeleb1_training_rvectors.pkl"
    >>> with open(train_file, "rb") as xvectors:
    ...     train_obj = pickle.load(xvectors)
    ...
    """

    data_dir = "/Users/nauman/Desktop/Mila/nauman/Data/xvect-sdk/sb-format/"
    train_file = data_dir + "VoxCeleb1_training_rvectors.pkl"

    # read extracted vectors (xvect, ivect, dvect, etc.)
    with open(train_file, "rb") as xvectors:
        train_obj = pickle.load(xvectors)
    print("Training started..")
