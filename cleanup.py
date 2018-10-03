import sys
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np

def open_clean_file(filename):
#with open("chicago-taxi-rides-2016/chicago_taxi_trips_2016_01.csv") as f:
    pddata = pd.read_csv(filename, usecols = ["trip_start_timestamp", "trip_end_timestamp", "trip_seconds", "trip_miles", "fare", "tips", "tolls", "trip_total","payment_type"], nrows = 1000)
    print type(pddata)
    # 1 for HOME, 0 for AWAY
    i = 0
    for data_obj in pddata["payment_type"]:
            if data_obj == "Cash":
                pddata.loc[i, "payment_type"] = 0
            elif data_obj == "Credit Card":
                 pddata.loc[i, "payment_type"] = 1
            else: 
                pddata.loc[i, "payment_type"] = 2
            i += 1

    return pddata

if __name__ == "__main__":

    train_data = open_clean_file("data/chicago_taxi_trips_2016_01.csv")
    train_data.to_csv("c.csv")



