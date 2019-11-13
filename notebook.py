# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(path):
    raw_data = pd.read_csv(
        path, names=["timestamp", "SSID", "BSSID", "RSSI"], delimiter=",")

    bssid_list = np.unique(raw_data.BSSID)

    # create a dictionary of dataframes s.t. bssid => Dataframe(timestamp, level, name)
    data = {}
    for i in range(bssid_list.shape[0]):
        #bssid_list[i] = bssid_list[i].lstrip()
        data[bssid_list[i]] = {"ssid": "", "rssi": np.array(
            []), "timestamp": np.array([])}

    # fill the new form of the data
    for bssid in bssid_list:
        sample_indeces = np.where(raw_data.BSSID.values == bssid)[0]
        data[bssid]["rssi"] = raw_data.RSSI.values[sample_indeces]
        data[bssid]["timestamp"] = raw_data.timestamp.values[sample_indeces]
        data[bssid]["ssid"] = raw_data.SSID.values[sample_indeces][0]

    return data, raw_data


# %%
# load wifi log
data_path = "../data/wifi.csv"
wifi_log, level_range = read_data(data_path)


# %%
