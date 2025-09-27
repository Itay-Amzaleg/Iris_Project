import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import confusion_matrix

import Data_Preprocessing as dataPre
import itertools

import calculate_Scores
import calculate_Scores as scores
from scipy.stats import mode


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

"""
First project written in python

Main goals:
1.Being well organized 
2.getting used to python syntax
3.Working with SK_learn library
4.Making simple visualizations
5.Comparing several algorithms variations and get insightful conclusions

Work Flow:

1.Adding meaningfull columns to the Iris data set calculating common statistical metrics

2.running K-means with K=3 on every 2 attributes combination and calculating the silhouette score, purity, recall, precision and f1 score
(since i know the true labels) and making conclusions about the "best" and "worst" combinations of attributes
since i already implemented K-means in the past I will use the sk_learn model

3.Visualise the results of Step 2 with graphs

4.Running K-means with K=max_attributes to see if a higher dimensional clustering will lead to a better result

5.Running KNN with different amount of neighbors (1 to max) and calculating recall, recision and f1 score to each model
since i never implemented KNN i will re-write the algorithm by myself

6.Visualise the results of Step 5 with graphs

"""
def get_Adjusted_Predictions(true, labeled):
    new_Order = np.zeros_like(labeled)
    for i in range(len(np.unique(labeled))):
        mask = (labeled == i)
        new_Order[mask] = mode(true[mask])[0]
    return new_Order

def create_Graph(x,y,labels,centroids):
    plt.scatter(X[a], X[b], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=100, color="r")
    plt.title(f'{a} VS {b}')
    plt.xlabel(a)
    plt.ylabel(b)
    plt.show()

def calc_Scores(true, predicted, centroids,c, metrics):

    confusion_matrix = calculate_Scores.create_Confusion_Matrix(true_labels, new_Labels)
    print(confusion_matrix)
    registry = {
        #"silhouette" : lambda: calculate_Scores.silhouette_Score(predicted, centroids),
        "purity" : lambda: calculate_Scores.purity(true, predicted),
        "accuracy" : lambda: calculate_Scores.accuracy(confusion_matrix),
        "recall" : lambda: calculate_Scores.recall(confusion_matrix,c),
        "precision" : lambda: calculate_Scores.precision(confusion_matrix,c),
        "f1" : lambda: calculate_Scores.f1(confusion_matrix,c)
    }
    return [round(float(registry[m]()), 3) for m in metrics]



if __name__ == '__main__':
    iris = sk.datasets.load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    dataPre.addColumns(df_iris)
    dataPre.statistics(df_iris)
    normalized_iris = dataPre.normalize(df_iris)
    true_labels = iris.target
    #normalized_iris will be used solely for K-means (we don't want data leakage for KNN)
    score_Matrix = pd.DataFrame(index=["Purity", "Accuracy", "Recall", "Precision", "f1 Score"])

    #cluster the data in 2D for every combination of attributes
    for a,b in itertools.combinations(df_iris.columns, 2):
        kMeans = sk.cluster.KMeans(n_clusters=3, random_state=40)
        X = normalized_iris[[a,b]]
        kMeans.fit(X)
        labels = kMeans.labels_
        centroids = kMeans.cluster_centers_
        create_Graph(a,b,labels,centroids)
        new_Labels = get_Adjusted_Predictions(true_labels, labels)
        score_Matrix[f'{a} and {b}'] = calc_Scores(true_labels, labels, centroids, 0,metrics=["purity","accuracy","recall","precision","f1"])
        #will calculate "recall","precision","f1" as for Setosa Arbitrary
    print(score_Matrix)


"""


   #splitting for training and testing data for KNN later
   df_iris_train = df_iris.sample(frac=0.8)
   df_iris_test = df_iris.drop(df_iris_train.index)

   #*setting up Kmeans Model
   kmeans = sk.cluster.KMeans(n_clusters=3)
   minmax_scaler = sk.preprocessing.MinMaxScaler()
   df_iris_scaled = pd.DataFrame(minmax_scaler.fit_transform(df_iris), columns=df_iris.columns)
   selected_features = ["sepal length (cm)","sepal width (cm)"]
   df_selected = df_iris_scaled[selected_features]
   kmeans.fit_transform(df_selected)
   df_iris["cluster"] = kmeans.labels_
   centroids = kmeans.cluster_centers_


   plt.scatter(df_iris_scaled["sepal length (cm)"], df_iris_scaled["sepal width (cm)"], c=df_iris["cluster"])
   plt.scatter(centroids[:,0], centroids[:,1], color='white', edgecolors='black', marker='o', s=500, alpha= 0.5)

   plt.show()
   """











