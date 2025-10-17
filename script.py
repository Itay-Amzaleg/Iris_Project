import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import confusion_matrix

import Data_Preprocessing
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

def create_Graph(labels,centroids):
    plt.scatter(X[a], X[b], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=100, color="r")
    plt.title(f'{a} VS {b}')
    plt.xlabel(a)
    plt.ylabel(b)
    plt.show()

def calc_Scores(true, predicted,X, c, metrics):
    confusion_matrix = calculate_Scores.create_Confusion_Matrix(true_labels, new_Labels)
    print(confusion_matrix)
    registry = {
        "silhouette" : lambda: calculate_Scores.silhouette_Score(predicted,X),
        "purity" : lambda: calculate_Scores.purity(true, predicted),
        "accuracy" : lambda: calculate_Scores.accuracy(confusion_matrix),
        "recall" : lambda: calculate_Scores.recall(confusion_matrix,c),
        "precision" : lambda: calculate_Scores.precision(confusion_matrix,c),
        "f1" : lambda: calculate_Scores.f1(confusion_matrix,c)
    }
    return [round(float(registry[m]()), 3) for m in metrics]

def print_top_5(df):
    top_5 = df.apply(lambda row: row.nlargest(5), axis=1)
    for metric, row in top_5.iterrows():
        print(f'Top 5 features combinations for {metric} are:')
        row_clean = row.dropna().sort_values(ascending=False)
        for index, (name, value) in enumerate(row_clean.items(), 1):
            print(f'{index}. {name} with score of {value}')

if __name__ == '__main__':
    iris = sk.datasets.load_iris()
    """
    df_iris_kMeans = pd.DataFrame(iris.data, columns=iris.feature_names)
    dataPre.addColumns(df_iris_kMeans)
    dataPre.statistics(df_iris_kMeans)
    normalized_iris = dataPre.normalize(df_iris_kMeans)
    true_labels = iris.target
    #normalized_iris will be used solely for K-means (we don't want data leakage for KNN)
    kMeans_1 = sk.cluster.KMeans(n_clusters=3, random_state=40)
    kMeans_1.fit(normalized_iris)
    new_Labels = get_Adjusted_Predictions(true_labels,kMeans_1.labels_)
    print(calc_Scores(true_labels, kMeans_1.labels_, normalized_iris, 0, metrics=["silhouette", "purity", "accuracy", "recall", "precision", "f1"]))

    score_Matrix = pd.DataFrame(index=["silhouette","Purity", "Accuracy", "Recall", "Precision", "f1 Score"])

    #cluster the data in 2D for every combination of attributes
    for a,b in itertools.combinations(df_iris_kMeans.columns, 2):
        kMeans_2 = sk.cluster.KMeans(n_clusters=3, random_state=40)
        X = normalized_iris[[a,b]]
        kMeans_2.fit(X)
        labels = kMeans_2.labels_
        centroids = kMeans_2.cluster_centers_
        create_Graph(labels,centroids)
        new_Labels = get_Adjusted_Predictions(true_labels, labels)
        score_Matrix[f'{a} and {b}'] = calc_Scores(true_labels, labels, X, 0,metrics=["silhouette", "purity","accuracy","recall","precision","f1"])
        #will calculate "recall","precision","f1" as for Setosa Arbitrary
        break
    print(score_Matrix)
    print_top_5(score_Matrix)

    #this is the end of the kmeans section
    #Suuming up: made a script that adds non-linear features, cluster the data based on every combination of 2 features and evaluates each combination to find the most
    #"good" combinations depends on what metric intrest you the most
    # a quick check will show that there are 2d clustering that are better than the 13d(4 original + 9 syntetic)
"""


    #From this part on, the KNN algorithm will be manually written in order to practice basic understanding of it, and basic writing in python

    df_iris_KNN = pd.DataFrame(iris.data, columns=iris.feature_names)
    Data_Preprocessing.addColumns(df_iris_KNN)
    #splitting for training and testing data for KNN later
    df_iris_train = df_iris_KNN.sample(frac=0.8, random_state=40)
    iris_train_true_labels = iris.target[df_iris_train.index]
    df_iris_test = df_iris_KNN.drop(df_iris_train.index)
    iris_test_true_labels = iris.target[df_iris_test.index]


    #pre-processing - adding non-liner features (already done), normalizing (saving the min and max for each coulomn to normalize the test afterwerds
    normalized_iris_train, normalized_iris_test = Data_Preprocessing.normalize(df_iris_train,df_iris_test)

    distance_matrix = Data_Preprocessing.distance_matrix(normalized_iris_train)

    distance_matrix = distance_matrix + distance_matrix.T

    #running knn with 1<=k<=50 (assuming max 50 samples for each type) to find the best value of K
    train_predict = np.empty(len(df_iris_train))
    for k in range(1,50):
        for index in range(len(normalized_iris_train)):
            closest_k = np.nlargest(distance_matrix[index],k)











