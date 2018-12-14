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

# Classifiers

GaussianNB.py			diff_feats.py
dtree.py
bernoulliNB.py			first_run.py
multinomial_naive_bayes.py
nbm_diff_params.py
cleanup.py			rfc.py
complement_bayes.py		rfc_1.py
creditvscash.py	
datacleanup.py


Any of the classifiers in this repo can be ran as follows:

python <classifier>.py

# Visualizations

### viz.py

viz.py displays the distributions across each feature
and it also prints the explained variances from PCA

run the program as follows:

python viz.py 
