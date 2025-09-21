import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import Data_Preprocessing as dataPre
import itertools



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

1.Adding meaningfull and meaning less columns to the Iris data set calculating common statistical metrics

2.running K-means with K=3 on every 2 attributes combination and calculating the silhouette score, purity, recall, precision and f1 score
(since i know the true labels) and making conclusions about the "best" and "worst" combinations of attributes
since i already implemented K-means in the past I will use the sk_learn model

3.Visualise the results of Step 2 with graphs

4.Running K-means with K=max_attributes to see if a higher dimensional clustering will lead to a better result
since i never implemented KNN i will re-write the algorithm by myself

5.Running KNN with different amount of neighbors (1 to max) and calculating recall, recision and f1 score to each model

6.Visualise the results of Step 5 with graphs

"""


if __name__ == '__main__':
    iris = sk.datasets.load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    dataPre.addColumns(df_iris)
    dataPre.statistics(df_iris)
    normalized_iris = dataPre.normalize(df_iris)
    true_labels = iris.target
    #normalized_iris will be used solely for K-means (we don't want data leakage for KNN)


    for a,b in itertools.combinations(df_iris.columns, 2):
        kMeans = sk.cluster.KMeans(n_clusters=3, random_state=40)
        X = normalized_iris[[a,b]]
        kMeans.fit(X)
        labels = kMeans.labels_
        centroids = kMeans.cluster_centers_
        print(labels)
        break



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












