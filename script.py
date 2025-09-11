import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn as sk
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def MessWithData(data):
    df_iris["is_big"] = df_iris.apply(lambda row: 1 if (row["sepal length (cm)"] > 3 and row["sepal width (cm)"] > 3) else 0, axis=1)
    df_iris["sepal avg"] = df_iris[["sepal length (cm)", "sepal width (cm)"]].mean(axis=1)
    df_iris["petal avg"] = (df_iris["petal length (cm)"] + df_iris["petal width (cm)"]) / 2.0
    return data

def KmeansTwoFeatures(data, features):
    kmeans = KMeans(n_clusters=len(features))


if __name__ == '__main__':
    iris = sk.datasets.load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris["target"] = iris['target']
    df_iris = MessWithData(df_iris)

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











