import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features_from_date


def load_dataset(
    datadir: str,
    dataname: str,
    val_split_rate: float,
    time_embedding: list = [True, 't'], # t: minute, h: hour, d: day, w: week, m: month, y: year
):
    trn_df = pd.read_csv(os.path.join(datadir, dataname, 'train.csv'))
    val_idx = int(len(trn_df) * val_split_rate) # val: val_idx ~ len(trn_df)
    trn_df, val_df = train_test_split(trn_df, test_size=val_split_rate, shuffle=False)

    test_df = pd.read_csv(os.path.join(datadir, dataname, 'test.csv'))
    test_label_df = pd.read_csv(os.path.join(datadir, dataname, 'test_label.csv'))

    if time_embedding[0]:
        trn_ts = time_features_from_date(trn_df['date'], timeenc=time_embedding[0], freq=time_embedding[1])
        trn = trn_df.drop(columns=['date'])
        val_ts = time_features_from_date(val_df['date'], timeenc=time_embedding[0], freq=time_embedding[1])
        val = val_df.drop(columns=['date'])
        test_ts = time_features_from_date(test_df['date'], timeenc=time_embedding[0], freq=time_embedding[1])
        test = test_df.iloc.drop(columns=['date'])
        label = test_label_df.iloc[:, 1:]
    else:
        trn_ts = trn_df.columns[0]
        trn = trn_df.iloc[:, 1:]
        val_ts = val_df.columns[0]
        val = val_df.iloc[:, 1:]
        test_ts = test_df.columns[0]
        test = test_df.iloc[:, 1:]
        label = test_label_df.iloc[:, 1:]

        print(label)

    var = trn.shape[1]

    return trn, trn_ts, val, val_ts, test, test_ts, var, label

load_dataset(
    datadir='src/dataset',
    dataname='PSM',
    val_split_rate=0.2,
    time_embedding=[False, 't'],
)