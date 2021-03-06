import sys
from sklearn.naive_bayes import ComplementNB
import csv
import numpy as np
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

    model = ComplementNB()
    model.fit(X, Y)


    TP=0
    TN=0
    FP=0
    FN=0
    total = 0
    with open("chicago_taxi_trip_TEST.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i is not 0:
                data_row = [float(num) for num in row[1:7]]
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
    print ("F1 Score = {}".format(F1))

    
