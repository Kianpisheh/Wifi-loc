import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import datetime as dt
import json
import os


def read_data(path):
    raw_data = pd.read_csv(
        path, names=["timestamp", "ssid", "bssid", "rssi"], delimiter=",")

    bssid_list = np.unique(raw_data.bssid)

    # create a dictionary of dataframes s.t. bssid => Dataframe(timestamp, level, name)
    data = {}
    for i in range(bssid_list.shape[0]):
        data[bssid_list[i]] = {"ssid": "", "rssi": np.array(
            []), "timestamp": np.array([])}

    # fill the new form of the data
    for bssid in bssid_list:
        sample_indeces = np.where(raw_data.bssid.values == bssid)[0]
        data[bssid]["rssi"] = raw_data.rssi.values[sample_indeces]
        data[bssid]["timestamp"] = raw_data.timestamp.values[sample_indeces]
        data[bssid]["ssid"] = raw_data.ssid.values[sample_indeces][0]

    return data, raw_data


def draw_rssi(wifi_log, n_rows, n_cols, labels=None):
    wifi_list = list(wifi_log.keys())
    _, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)

    # handle special dim cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array(axes).reshape(1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for r in range(n_rows):
        for c in range(n_cols):
            idx = wifi_list[r*n_cols + c]
            rssi_data = wifi_log[idx]["rssi"]
            timestamp = wifi_log[idx]["timestamp"]
            wifi_name = wifi_log[idx]["ssid"]
            if wifi_name == " ":
                wifi_name = "no_name"
            timestamp = [dt.datetime.fromtimestamp(
                round(t/1000)) for t in timestamp]
            axes[r, c].plot(timestamp, rssi_data)
            axes[r, c].xaxis.set_major_formatter(
                mdate.DateFormatter('%H:%M:%S'))
            axes[r, c].xaxis.set_tick_params(rotation=30, labelsize=8)
            axes[r, c].set_title(wifi_name, fontsize=8)
            if labels is not None:
                add_labels_indicators(axes[r, c], labels)
    plt.show()


def add_labels_indicators(ax, labels):
    colors = ["blue", "red", "purple", "green"]
    i = 0
    for label, intervals in labels.items():
        for interval in intervals:
            ax.axvline(x=dt.datetime.fromtimestamp(
                interval[0]/1000), color=colors[i])
            ax.axvline(x=dt.datetime.fromtimestamp(
                interval[1]/1000), color=colors[i])
        i += 1


def read_labels(path):
    # create a dict of intervals for each label
    all_labels = None
    print(path)
    with open(path, "r") as read_file:
        all_labels = json.load(read_file)

    # find the list of labels
    labels_list = list(set([label["label1_2"] for label in all_labels]))

    labels = {}
    for label in labels_list:
        labels[label] = []
        for label_instance in all_labels:
            if (label_instance["label1_2"] == label):
                labels[label].append(
                    (label_instance["time1_2"], label_instance["time2_1"]))

    return labels


def data_complete_feature_set(raw_wifi_data, bssid_list, time_window=2):
    """create a dataframe in which each sample has a value
        for each AP; NA when AP is not available

    Arguments:
        wifi_log {dictionary} -- a dictionary of wifi logs.
            key:bssid, values: {timestamp: np.ndarray, ssid="wifi_name", rssi: np.ndarray}
    Return:
        data {pd.DataFrame} -- a dataframe with all APs as its columns, and label
    """
    header = ["timestamp"] + bssid_list
    data = pd.DataFrame(columns=header)

    sample = {}
    for index, row in raw_wifi_data.iterrows():
        if len(sample) == 0:
            sample["timestamp"] = row.timestamp
        elif row.timestamp - sample["timestamp"] > (time_window - 0.06)*1000:
            sample_df = pd.DataFrame(
                sample, index={0}, columns=list(sample.keys()))
            data = pd.merge(data, sample_df, how="outer")
            sample = {}
        else:
            sample[row.bssid] = row.rssi

    return data


def __get_label(timestamp, labels):
    sample_label = "no label"
    for label, intervals in labels.items():
        for interval in intervals:
            if interval[0] <= timestamp <= interval[1]:
                sample_label = label

    return sample_label


def create_json_file(raw_data, time_window=2, labels=None, file_name=None, save=True):
    """ creates a list of json objects (dictionaries).

    Arguments:
        raw_data {pd.DataFrame} -- columns: timestamp, bssid, ssid, rssi

    Keyword Arguments:
        save {bool} -- it saves the result if true (default: {True})

    Returns:
        [list] -- a list of json objects: [{timestamp:***, bssid:[bssid:level], ssid:["***"]}]
    """
    data_json = []

    sample = {}
    for index, row in raw_data.iterrows():
        if len(sample) == 0:
            sample = {"timestamp": row.timestamp, "rssi": {
                row.bssid: row.rssi}, "ssid": [row.ssid]}
            if labels is not None:
                sample["label"] = __get_label(row.timestamp, labels)
        elif row.timestamp - sample["timestamp"] > (time_window - 0.06)*1000:
            data_json.append(sample)
            sample = {}
        else:
            sample["rssi"][row.bssid] = row.rssi
            sample["ssid"].append(row.ssid)

    if save and file_name is not None:
        if not os.path.exists("./output"):
            os.mkdir("./output")
        with open("./output/" + file_name + ".json", 'w') as file:
            file.write(json.dumps(data_json, indent=4))
    return data_json


def read_json_file(path, file_name):
    with open(f'{path}/{file_name}', encoding='utf-8') as file:
        data = json.loads(file.read())
    return data


def rssi_histogram(data, bin_width=3):
    histograms = {}
    for sample in data:
        rssi_dict = sample["rssi"]  # rssi => {bssid: rssi, ...}
        label = sample["label"]
        for bssid, rssi in rssi_dict.items():
            if not bssid in histograms:
                histograms[bssid] = {label: [rssi]}
            elif not label in histograms[bssid]:
                histograms[bssid][label] = [rssi]
            else:
                histograms[bssid][label].append(rssi)
    return histograms


def draw_rssi_histograms(histograms, n_rows, n_cols):
    """plots the rssi level histogram of different BSSIDs

    Arguments:
        histograms {dictionry} -- a dictionary of dictionaries 
                                    containing bssid level for each bssid
        n_rows {int} -- the number of subplot rows 
        n_cols {[type]} -- the number of subplot columns
    """
    i = 0
    for bssid, histogram in histograms.items():
        legends = []
        for label, rssi_list in histogram.items():
            ax = plt.subplot(n_rows, n_cols, i+1)
            ax.hist(np.array(rssi_list), 10)
            legends.append(label)
        ax.legend(legends)

        ax.set_title(bssid)
        i += 1
        if (i >= n_cols*n_rows):
            break
    plt.show()

    # make json files
make_json_files = False

if make_json_files:
    # load the training wifi log
    data_path = "../data"
    train_labels = read_labels(data_path + "/train_labels.json")
    wifi_log, raw_data = read_data(data_path + "/wifi_train.csv")
    train_data = create_json_file(
        raw_data, labels=train_labels, time_window=2, file_name="train_data")

    # load the test wifi log
    wifi_log, raw_data = read_data(data_path + "/wifi_test.csv")
    bssid_list = list(wifi_log.keys())
    test_data = create_json_file(
        raw_data, labels=None, time_window=2, file_name="test_data")

test_knn = True

if test_knn:
    data_path = "./output"
    train_data = read_json_file(data_path, "train_data.json")
    test_data = read_json_file(data_path, "test_data.json")
    histograms = rssi_histogram(train_data)
    draw_rssi_histograms(histograms, 6, 7)
    x = 1


# bssid_list = list(wifi_log.keys())
# train_data = data_complete_feature_set(raw_data, bssid_list, time_window=2)

# # load the test wifi log
# wifi_log, raw_data = read_data(data_path + "/wifi_test.csv")
# bssid_list = list(wifi_log.keys())
# test_data = data_complete_feature_set(raw_data, bssid_list, time_window=2)

# # save the training and test data as csv files
# if not os.path.exists("./output"):
#     os.mkdir("./output")
# train_data.to_csv("./output/data_train.csv", index=False)
# test_data.to_csv("./output/data_test.csv", index=False)


# draw time-rssi plots
# draw_rssi(wifi_log, 10, 2, labels)
