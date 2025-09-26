import pandas as pd
import numpy as np

#def silhouette_Score():

#def purity():

#def accuracy():

def create_Confusion_Matrix(true, predicted):
    """

    :param true: the true labels of the data set - 0 for Setosa, 1 for Versicolor, 2 for Virginica
    :param predicted:
    :return: confusion_matrix as data frame
    """
    matrix = pd.DataFrame(columns=["predicted 0", "predicted 1", "predicted 2"], index=["true 0", "true 1", "true 2"])
    for i in np.unique(true):
        for j in np.unique(true):
            matrix.loc[f'true {i}', f'predicted {j}'] = np.sum((true == i) & (predicted == j))

"""
#def recall(confusion_matrix):
    tp =
    tn =
    fp =
    fn =


    return tp / (tp + fn)

def precision(confusion_matrix):
    tp =
    tn =
    fp =
    fn =

    return tp / (tp + fp)

def f1(confusion_matrix):
    pre = precision(confusion_matrix)
    rec = recall(confusion_matrix)
    return pre * rec * 2 / (pre + rec)
"""
