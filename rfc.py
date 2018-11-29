import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import sklearn as skl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    X = []
    Y = []

    with open("chicago_taxi_trip_january.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i is not 0:
                data_row = [float(num) for num in row[1:7]]
                X.append(data_row)
                Y.append(row[7])

    X = np.array(X)
    Y = np.array(Y)

    X_test = []
    Y_test = []
    with open("chicago_taxi_trip_TEST.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i is not 0:
                data_row = [float(num) for num in row[1:7]]
                X_test.append(data_row)
                Y_test.append(row[7])

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)


    #create pipeline
    scores = []
    yticklabels = []
    for i in range(1, 250, 10):
        score_r  = []
        for j in range(1, 7):
            pipe = Pipeline(steps=[('PCA', PCA(n_components=j)), ('RFC', RandomForestClassifier(n_estimators=i))])
            pipe.fit(X, Y)
            predictions = pipe.predict(X_test)
            score_r.append(accuracy_score(Y_test, predictions))
        scores.append(score_r)
        yticklabels.append(i)

    scores = np.asarray(scores)

    print scores.shape

    f,ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(scores, cmap = 'Greys', annot=True, linewidths=.5, fmt= '.3f', xticklabels=[1, 2, 3, 4, 5, 6], yticklabels=yticklabels, ax=ax)
    plt.xlabel("n_components")
    plt.ylabel("n_estimators")
    plt.show()
