import pandas as pd
import numpy as np
from scipy.stats import mode


#def silhouette_Score(predicted, centroids):

    #return 1

def purity(true, non_adj_predicted):
    count = 0
    for i in range(len(np.unique(non_adj_predicted))):
        mask = non_adj_predicted == i
        majority = mode(true[mask])[0]
        count+= np.sum(true[mask] == majority)
    return 1 / len(true) * count

def accuracy(cMatrix):
    cm = cMatrix.to_numpy()
    return np.trace(cm)/ cm.sum()


def create_Confusion_Matrix(true, predicted):
    """

    :param true: the true labels of the data set - 0 for Setosa, 1 for Versicolor, 2 for Virginica
    :param predicted:
    :return: confusion_matrix as data frame
    """
    cMatrix = pd.DataFrame(columns=["predicted 0", "predicted 1", "predicted 2"], index=["true 0", "true 1", "true 2"])
    for i in np.unique(true):
        for j in np.unique(true):
            cMatrix.loc[f'true {i}', f'predicted {j}'] = np.sum((true == i) & (predicted == j))
    return cMatrix


def recall(confusion_matrix, c):
    """
    :param confusion_matrix: confusion_matrix as data frame
    :param c: the class that the recall is calculated for
    :return: recall score
    """
    tp = confusion_matrix.loc[f'true {c}', f'predicted {c}']
    fn = confusion_matrix.loc[f'true {c}'].sum() - tp

    return tp / (tp + fn) if(tp + fn > 0) else np.nan

def precision(confusion_matrix, c):
    """
    :param confusion_matrix: confusion_matrix as data fram
    :param c: the class that the recall is calculated for
    :return: precision score
    """
    tp = confusion_matrix.loc[f'true {c}', f'predicted {c}']
    fp = confusion_matrix.loc[:, f'predicted {c}'].sum() - tp

    return tp / (tp + fp) if(tp + fp > 0) else np.nan

def f1(confusion_matrix,c ):
    pre = precision(confusion_matrix,c)
    rec = recall(confusion_matrix,c)
    return pre * rec * 2 / (pre + rec)

