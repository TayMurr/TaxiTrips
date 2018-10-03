import sys
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
if __name__ == "__main__":

    #with open("chicago-taxi-rides-2016/chicago_taxi_trips_2016_01.csv") as f:
    data_frame = pd.read_csv("data/chicago_taxi_trips_2016_01.csv")
    data_frame.to_csv("chicago_taxi_trip_january.csv")
    credit_tot = 0
    card_tot = 0
    unknown = 0
    nocharge = 0
    dispute = 0
    pcard = 0
    for data_obj in data_frame["payment_type"]:
        if data_obj == "Credit Card":
            credit_tot += 1
        elif data_obj == "Cash":
            card_tot += 1
        elif data_obj == "Unknown":
            unknown += 1
        elif data_obj == "No Charge":
            nocharge += 1
        elif data_obj == "Dispute":
            dispute += 1
        elif data_obj == "Pcard":
            pcard += 1
    height = [credit_tot, card_tot, unknown, nocharge, dispute, pcard]
    print len(data_frame["payment_type"])
    print unknown
    print nocharge
    print dispute
    print pcard
    fig, ax = plot.subplots()
    bars = ['credit card', 'cash', 'unknown', 'no charge', 'dispute', 'pcard']
    y_pos = np.arange(len(bars))
    ax.bar(y_pos, height)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(bars)
    ax.set_ylabel("frequency")
    ax.set_xlabel("payment type")
    ax.set_yscale('log')
    plot.show()
