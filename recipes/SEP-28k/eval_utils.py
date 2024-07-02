import warnings

import torch
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")


def my_confusion_matrix(y_trues, y_preds):
    """Computes metrics for the epoch (Acc, F-score, Missrate, Confusion Matrix, Precision, Recall)"""
    # constant for classes
    y_preds = torch.round(y_preds)
    cf_matrix = confusion_matrix(
        y_trues.cpu().detach().numpy(), y_preds.cpu().detach().numpy()
    )
    TP = cf_matrix[1, 1]
    FP = cf_matrix[0, 1]
    FN = cf_matrix[1, 0]
    TN = cf_matrix[0, 0]
    fscore = 2 * TP / (2 * TP + FP + FN)
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    missrate = FN / (FN + TP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return accuracy, fscore, missrate, cf_matrix, precision, recall
