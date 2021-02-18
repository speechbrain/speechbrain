#!/usr/bin/python
import os
import pickle
import numpy
from numpy import linalg as LA
from speechbrain.processing.PLDA_LDA import StatObject_SB  # noqa F401
from speechbrain.processing.PLDA_LDA import PLDA
from speechbrain.processing.PLDA_LDA import Ndx
from speechbrain.processing.PLDA_LDA import fast_PLDA_scoring


# Load params file
experiment_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = "../../../../../samples/plda_xvect_samples/"
data_folder = os.path.abspath(experiment_dir + data_folder)

# Xvectors stored as StatObject_SB
train_file = data_folder + "/train_stat_xvect.pkl"
enrol_file = data_folder + "/enrol_stat_xvect.pkl"
test_file = data_folder + "/test_stat_xvect.pkl"
scores_file = data_folder + "/expected_plda_scores.pkl"

# Load Train
with open(train_file, "rb") as input:
    train_obj = pickle.load(input)

# Load Enrol
with open(enrol_file, "rb") as input:
    enrol_obj = pickle.load(input)

# Load Test
with open(test_file, "rb") as input:
    test_obj = pickle.load(input)

print("Training PLDA...")
plda = PLDA()
plda.plda(train_obj)

# Preparing Ndx map
models = enrol_obj.modelset
testsegs = test_obj.modelset
ndx_obj = Ndx(models=models, testsegs=testsegs)

# PLDA scoring between enrol and test
scores_plda = fast_PLDA_scoring(
    enrol_obj, test_obj, ndx_obj, plda.mean, plda.F, plda.Sigma
)
print("PLDA score matrix: (Rows: Enrol, Columns: Test)")
print(scores_plda.scoremat)

with open(scores_file, "rb") as input:
    expected_score_matrix = pickle.load(input)

print("Expected scores:\n", expected_score_matrix)

# Ensuring the scores are proper (for integration test)
dif = numpy.subtract(expected_score_matrix, scores_plda.scoremat)
f_norm = LA.norm(dif, ord="fro")


# Integration test: Ensure we get same score matrix
def test_error():
    assert f_norm < 0.1
