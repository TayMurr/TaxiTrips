import sys
from sklearn.naive_bayes import GaussianNB
import csv
import numpy as np
if __name__ == "__main__":

    X = []
    Y = []

    with open("chicago_taxi_trip_january.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i is not 0:
                print row
                data_row = [float(num) for num in row[1:7]]
                X.append(data_row)
                Y.append(row[7])
    print "features"
    print X
    print "class"
    print Y
    X = np.array(X)
    Y = np.array(Y)

    model = GaussianNB()
    model.fit(X, Y)

    predicted= model.predict([[1080.0,6.2,17.75,0.0,0.0,17.75]])
    print predicted
