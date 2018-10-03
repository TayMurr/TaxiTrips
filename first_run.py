import numpy as np
import scipy.stats as st
import pandas as pd

def open_clean_file(filename):
	data = np.genfromtxt(filename,delimiter=',',names=True,dtype=None,encoding=None)
	pddata = pd.DataFrame(data)[["trip_start_timestamp", "trip_end_timestamp", "trip_seconds", "trip_miles", "fare", "tips", "tolls", "trip_total","payment_type"]]
	# 1 for HOME, 0 for AWAY
	"""for i, a in enumerate(pddata["trip_start_timestamp"]):
		pddata["trip_start_timestamp"][i] = pd.to_datetime(pddata["trip_start_timestamp"][i])
		pddata["trip_end_timestamp"][i] = pd.to_datetime(pddata["trip_end_timestamp"][i])"""

	# 1 for HOME, 0 for AWAY
	for i, a in enumerate(pddata["payment_type"]):
		if pddata["payment_type"][i] == "Cash":
			pddata["payment_type"][i]=0
		elif pddata["payment_type"][i] == "Credit Card":
			pddata["payment_type"][i]=1
		elif pddata["payment_type"][i] == "Unknown":
			pddata["payment_type"][i]=2

	return pddata

if __name__ == '__main__':
	train_data = open_clean_file("data/chicago_taxi_trips_2016_01.csv")
	train_labels = train_data["payment_type"]
	train_data = train_data[["trip_start_timestamp", "trip_end_timestamp", "trip_seconds", "trip_miles", "fare", "tips", "tolls", "trip_total"]]

	attributes = list(train_data.columns.values)

	print train_data.dtypes









	# # Do an initial run with the full training dataset
	# level_max = 9
	# tree = DecisionTree(train_data, train_labels, attributes, max_level=level_max)
	# y = tree.classify(test_data)
	# np_test_labels = [int(x) for x in test_labels.values]
	# np_y = [int(x) for x in y]
	# print np_y
	# print np_test_labels
	
	# TP=0
	# TN=0
	# FP=0
	# FN=0
	# total=0

	# for i, x in enumerate(np_y):
	# 	if np_y[i] == np_test_labels[i]:
	# 		if np_y[i] == 1:
	# 			TP+=1
	# 		else:
	# 			TN+=1
	# 	else:
	# 		if np_y[i] == 1 and np_test_labels[i] == 0:
	# 			FP+=1
	# 		elif np_y[i] == 0 and np_test_labels[i] == 1:
	# 			FN+=1
	# 	total = total + 1

	# accuracy = float(TP+TN)/total
	# precision = float(TP)/(TP+FP)
	# recall = float(TP)/(TP+FN)
	# F1 = float(2*precision*recall)/(precision+recall)

	# print ("Accuracy = {}".format(accuracy))
	# print ("Precision = {}".format(precision))
	# print ("Recall = {}".format(recall))
	# print ("F1 Score = {}".format(F1))
