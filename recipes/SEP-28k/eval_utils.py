
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def my_confusion_matrix(y_trues, y_preds):
    # constant for classes  
    y_preds = torch.round(y_preds)
    cf_matrix= confusion_matrix(y_trues.cpu().detach().numpy(), 
                                y_preds.cpu().detach().numpy())
    TP = cf_matrix[1,1]
    FP = cf_matrix[0,1]
    FN = cf_matrix[1,0]
    TN = cf_matrix[0,0]
    if(TN!=0):
        fscores = 2*TP/(2*TP+FP+FN)
    else:
        fscores = 0
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    missrate = FN / (FN+TP)
    if(TP+FP > 0):
        precision = TP / (TP+FP)
    else:
        precision = 0
    recall = TP / (TP+FN)
    return accuracy, fscores, missrate, cf_matrix, precision, recall