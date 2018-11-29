import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import sklearn as skl
from sklearn.decomposition import PCA
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

    X = (X-X.mean())/(X.max()-X.min())

    fig, axs = plt.subplots(2, 3)
    
    axs[0, 0].hist(X[:, 0], bins=50, color='#000000',
        rwidth=0.85)
    axs[0, 0].set_xlabel("trip seconds")
    axs[0, 1].hist(X[:, 1], bins=50, color='#000000',
        rwidth=0.85)
    axs[0, 1].set_xlabel("trip miles")
    axs[0, 2].hist(X[:, 2], bins=50, color='#000000',
        rwidth=0.85)
    axs[0, 2].set_xlabel("fare")
    axs[1, 0].hist(X[:, 3], bins=50, color='#000000',
        rwidth=0.85)
    axs[1, 0].set_xlabel("tips")
    axs[1, 1].hist(X[:, 4], bins=50, color='#000000',
        rwidth=0.85)
    axs[1, 1].set_xlabel("tolls")
    axs[1, 2].hist(X[:, 5], bins=50, color='#000000',
        rwidth=0.85)
    axs[1, 2].set_xlabel("trip total")

    plt.show()

    """pca = PCA()
    pca.fit(X)

    plt.figure(1, figsize=(14, 13))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_ratio_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')
    plt.show()
    print X.shape"""




