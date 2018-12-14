# Introduction

This repository is made up of the data and the code used for project.
The sections below present helpful info for running and understanding
the dataset.

# Dependencies and Python version
The version of python we used for this project was Python 2.7.15

The dependecies are the following:
- sys
- sklearn
- csv
- numpy
- pandas
- matplotlib
- seaborn
- scipy

# Data

The data used to train is chicago_taxi_trip_january.csv 
and the data used to test is called chicago_taxi_trip_TEST.csv 
these csv files are excerpts from the kaggle dataset Chicago Taxi 
Rides 2016. The entire dataset is much to large for us to fit on 
our computers so that is why we use a subset. The files used for 
cleaning the data and reformmating it are cleanup.py and datacleaning.py 

Note: These clean files  are included to show you how we cleaned the data. The entire Kaggle
dataset is much too big for the submission. If you download the Kaggle dataset 
and store the csv is a directory called data then you can run them.

# Classifiers

The files that train and test classifiers are the following:
- GaussianNB.py
- dtree.py (the decision tree)
- bernoulliNB.py
- multinomial_naive_bayes.py
- complement_bayes.py
- rfc.py (random forest classifier)

Any of the classifiers in this repo can be ran as follows:

python <classifier>.py

They all print the accuracy. Some are also set to print the
accuracy, precision, recall, and F1 score.
 
# Visualizations

### viz.py

viz.py displays the distributions across each feature
and it also prints the explained variances from PCA

run the program as follows:

python viz.py

### rfc_heatmap.py

rfc_heatmap.py displays the heatmap of accuracies across
n components in PCA and n estimators in the RFC.

run the program as follows:

python rfc_heatmap.py

### creditvscash.py

creditvscash.py displas the barchart for each of the payment types

run as followS:

python creditvscash.py

 
