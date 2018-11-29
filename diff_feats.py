import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import sklearn as skl
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
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
    #scaler = skl.preprocessing.StandardScaler(copy=False)
    #scaler.fit_transform(X)
   
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

    X = (X-X.mean())/(X.max()-X.min())


    """scores = []
    print Y.shape

    for i in range(6):
        score_r = []
        X_temp = X[:, :6-i]#.reshape(-1, 1)
        X_temp_t = X_test[:, :6-i]#.reshape(-1, 1)
        model = GaussianNB()
        model.fit(X_temp, Y)
        score_r.append(model.score(X_temp_t, Y_test)) 
        model = MultinomialNB()
        model.fit(X_temp, Y)
        score_r.append(model.score(X_temp_t, Y_test))
        model = ComplementNB()
        model.fit(X_temp, Y)
        score_r.append(model.score(X_temp_t, Y_test))
        model = BernoulliNB()
        model.fit(X_temp, Y)
        score_r.append(model.score(X_temp_t, Y_test))
        scores.append(score_r)

    print scores
    
    scores = np.asarray(scores)

    f,ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(scores, annot=True, linewidths=.5, fmt= '.3f',ax=ax)
    plt.show()"""

    """ TP=0
    TN=0
    FP=0
    FN=0
    total = 0
    with open("chicago_taxi_trip_TEST.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i is not 0:
                data_row = [float(num) for num in row[1:7]]
                #data_row = np.asarray(data_row).reshape(-1, 1)
                #data_row = scaler.transform(data_row)
                prediction = model.predict([data_row])[0]
                if prediction == row[7]:
                    if prediction == '1':
                        TP += 1
                    else:
                        TN += 1
                else:
                    if prediction == '1' and row[7] == '0':
                        FP += 1
                    elif prediction == '0' and row[7] == '1':
                        FN += 1
            total += 1
    accuracy = float(TP+TN)/total
    precision = float(TP)/(TP+FP)
    recall = float(TP)/(TP+FN)
    F1 = float(2*precision*recall)/(precision+recall)

    print ("Accuracy = {}".format(accuracy))
    print ("Precision = {}".format(precision))
    print ("Recall = {}".format(recall))
    print ("F1 Score = {}".format(F1))"""

    
