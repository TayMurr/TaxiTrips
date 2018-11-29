import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

taxiData = np.genfromtxt('chicago_taxi_trip_january.csv',delimiter=',',names=True,dtype=None,encoding=None)
df = pd.DataFrame(taxiData)[["trip_seconds", "trip_miles", "fare", "tips", "tolls", "trip_total"]]
y = pd.DataFrame(taxiData)[["payment_type"]]

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)

dtree=tree.DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_predict = dtree.predict(X_test)

np_test_labels = [int(x) for x in y_test.values]
np_y = [int(x) for x in y_predict]
print np_y
print np_test_labels

TP=0
TN=0
FP=0
FN=0
total=0

for i, x in enumerate(np_y):
	if np_y[i] == np_test_labels[i]:
		if np_y[i] == 1:
			TP+=1
		else:
			TN+=1
	else:
		if np_y[i] == 1 and np_test_labels[i] == 0:
			FP+=1
		elif np_y[i] == 0 and np_test_labels[i] == 1:
			FN+=1
	total = total + 1

accuracy = float(TP+TN)/total
precision = float(TP)/(TP+FP)
recall = float(TP)/(TP+FN)
F1 = float(2*precision*recall)/(precision+recall)

print ("Accuracy = {}".format(accuracy))
print ("Precision = {}".format(precision))
print ("Recall = {}".format(recall))
print ("F1 Score = {}".format(F1))